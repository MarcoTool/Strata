# gfx906: split prefill for the remote-expert decode (branch gfx906-ep)

Two MI50 (gfx906), PCIe 3.0 (card 0 x8 at 7.1 GB/s, card 1 x16 at 13.8 GB/s), Qwen3.8-Flash-Next IQ3_S.

## Why

Measured at the production settings (262K context, same prompts, 2026-10-23):

| setup | decode | 4K prompt | 32K prompt |
| --- | ---: | ---: | ---: |
| A: layer split (`--layer-split auto`) | 44.0 | 327 | 622 |
| B: card 0 dense + card 1 remote experts (`--expert-cache-device1 auto --remote-expert-opt`) | 48.3 | 267 | 396 |
| C: `--peer-device 1` (card 1 also computes its experts' prompt rows) | 47.7 | 370 | 380 |

B decodes ~10% faster than A, but reads a long prompt at 0.64 of A: with B every layer's dense work and attention
runs on card 0, while A's two cards each read half the layers, pipelined chunk by chunk (`Prefill::set_stage`).

Tensor parallelism inside a layer does not pay on this platform: an event-ordered handoff between the cards costs
~27 us (65 us round trip), and splitting a GDN layer's projections over the two cards was 0.72-0.94 of one card
(`tools/hip/tp_gdn_probe.cpp`).  Device-side flag handoffs hung (VRAM, fine-grained and host-coherent flags).

## The plan

Read the prompt as A does, decode as B does (`STRATA_GFX906_SPLIT_PREFILL=1`, off by default):

1. Card 1 gets a prefill-only stage for layers [L1, 48): its own copy of those layers' dense weights (~2.1 GiB),
   a `SessionState` carved to [L1, 48) and a `Prefill` with `set_stage(L1, 48, nullptr)`.  Its experts: the remote
   tier's cache on card 1 (`RemoteExperts::cache_`), the rest streamed from the pinned host arena as usual.
2. Card 0's prompt path becomes stage 0: `set_stage(0, L1, &card1_prefill)` before `init`.
3. After `run()`: the layers [L1, 48) of card 1's session move into card 0's session (which B carves for all 48):
   GDN rows (`gdn_state`, global GDN ordinals) by peer copy; QSA K/V by `conversation_kv_save` /
   `conversation_kv_restore` per QSA layer (Strata's own int8 / resident-ring handling), or a raw copy when the two
   states have the same layout.
4. MTP: `on_chunk` runs on the last stage (card 1) with the chunk's residual rows on card 1; the drafter is on card 0,
   so the rows are copied over before `draft_kv` (T x hc x n_embd floats a chunk).
5. Mid-prompt checkpoints (`on_stage_chunk`) and the conversation cache: off in this mode at first.

## Open questions

- card 1's VRAM: remote experts (21.8 GiB) + dense [L1, 48) (~2.1 GiB) + session [L1, 48) + prompt buffers.
- whether `Prefill` on card 1 can take `RemoteExperts::cache_` as its resident cache, or needs its own.
- the handoff's cost at 32K (QSA K/V of 6 layers, int8, resident 32768 cells).

## Prototype results (2026-10-24, `strata generate`, Flash-Next IQ3_S, B config, greedy, `--spec 4`)

`STRATA_GFX906_SPLIT_PREFILL=24`: CUDA0 reads layers 0-23 of each chunk, CUDA1 layers 24-47, pipelined
(`Prefill::set_stage`); the later layers' state then moves into CUDA0's session (`split_handoff_layers`,
docs/fn/split_handoff_layers.md) and the decode runs as plain B.  Greedy tokens equal plain B in every case.

| prompt / chunk | plain B | split L=24 | note |
|---|---|---|---|
| 8K / 8192 (1 chunk) | 384 tok/s | 380 | one chunk: the stages run one after the other |
| 8K / 2048 (4 chunks) | 149 | 316 | small chunks re-stream every expert per chunk |
| 32K / 8192 (4 chunks), CUDA1 experts streamed | 403 | 616 | |
| 32K / 8192, CUDA1 reads the remote tier's resident blobs | 403 | 625 | default |
| 32K / 4096 (8 chunks) | - | 558 | per-chunk efficiency and 2x streaming outweigh the shorter fill |

`STRATA_PREFILL_TIMING=1`, 32K / 8192: both stages are compute bound and balanced - CUDA0 ~10.5 s per chunk,
CUDA1 ~9.8 s, expert copy waits 79 ms in all.  Wall = CUDA0's 4 chunks + CUDA1's last (~51 s); the pipeline
fill is one chunk.  GPU time: hc read 20%, gemm gate/up 16%, GDN 13%, QSA attention 10%, gemm down 9%.
Handoff: 273 MiB in ~0.3 s at 32K.

Open: TTFT exceeds the prefill by ~7.5 s with the split (~1 s plain); serve-mode port; kv_mode 1 (>32K) handoff.
`STRATA_GFX906_SPLIT_PREFILL_STREAM=1` keeps CUDA1 on streamed experts (A/B).
