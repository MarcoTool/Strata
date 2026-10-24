# `split_handoff_layers`

## Purpose

Moves the prompt state of layers `[lb, le)` from `src_dev` after a split prefill into the decode session on `dst_dev`.
GDN layers copy their recurrent and conv state layer by layer. QSA layers move the K/V and pooled index of the first
`n_tokens` positions through the conversation snapshot API, and peer-copy the indexer's tail, dead keys and block
positions separately.

## Signature

```cpp
bool split_handoff_layers(const core::SessionState& src, int src_dev, void* src_stream,
                          const core::SessionState& dst, int dst_dev, void* dst_stream,
                          const core::ModelGeometry& g, int64_t lb, int64_t le, int64_t n_tokens,
                          HandoffStats& stats, std::string& err);
```

## Inputs

- `src`: source `SessionState`, a host-side descriptor; not owned. Its state pointers point to device memory on
  `src_dev` and must own the QSA/GDN ordinals of `[lb, le)`.
- `src_dev`: source CUDA device ordinal; must be non-negative.
- `src_stream`: a `cudaStream_t` on the source device, passed as `void*`; not owned, must not be null.
- `dst`: destination `SessionState`, a host-side descriptor; not owned. Its state pointers point to device memory on
  `dst_dev` and must own the QSA/GDN ordinals of `[lb, le)`.
- `dst_dev`: destination CUDA device ordinal; must be non-negative.
- `dst_stream`: a `cudaStream_t` on the destination device, passed as `void*`; not owned, must not be null.
- `g`: model geometry in host memory, read-only, not owned; `n_layers` and `qsa_interval` determine each layer's type
  and global ordinal.
- `lb`: first global layer index to move, inclusive.
- `le`: end of the range, exclusive.
- `n_tokens`: prompt length to save and restore, in tokens; must not exceed `max_cells` of either session.
- `stats`: host output owned by the caller; zeroed on entry.
- `err`: host error string owned by the caller; cleared on entry.

## Outputs

Returns `true` on success and `false` on failure. `stats.gdn_layers` and `stats.qsa_layers` count the layers moved
completely. `stats.ms` is the time in milliseconds from the first stream synchronization until both devices have
synchronized. `stats.bytes` follows the inline reference's accounting: it counts the GDN rows and
`ConversationKv::bytes()`, not the extra per-QSA peer copies of `tail`, `dead` and `block_pos`.

The destination session's state for `[lb, le)` is overwritten. A failure can happen after some layers have been moved;
the destination session must then not be treated as a complete handoff.

## Preconditions

- `0 <= lb < le <= g.n_layers`, and `g.qsa_interval > 0`.
- `n_tokens > 0`, and no larger than `src.max_cells` or `dst.max_cells`.
- `qsa_ord0/qsa_alloc` and `gdn_ord0/gdn_alloc` of both sessions cover the range.
- The required `gdn_state`, `qsa_states` and QSA index state pointers are valid and on the declared devices.
- Peer access and the resources the conversation snapshot API needs are set up by the caller.
- No other thread or stream modifies the source or destination state during the handoff.

## Errors

This function writes the following to `err` directly:

- `split_handoff_layers: invalid layer range`
- `split_handoff_layers: qsa_interval must be positive`
- `split_handoff_layers: n_tokens must be positive`
- `split_handoff_layers: n_tokens exceeds source session capacity`
- `split_handoff_layers: n_tokens exceeds destination session capacity`
- `split_handoff_layers: src_dev must be non-negative`
- `split_handoff_layers: dst_dev must be non-negative`
- `split_handoff_layers: src_stream is null`
- `split_handoff_layers: dst_stream is null`
- `split_handoff_layers: source session does not own QSA range`
- `split_handoff_layers: destination session does not own QSA range`
- `split_handoff_layers: source session does not own GDN range`
- `split_handoff_layers: destination session does not own GDN range`
- `split_handoff_layers: source gdn_state is null`
- `split_handoff_layers: destination gdn_state is null`
- `split_handoff_layers: source qsa_states is null`
- `split_handoff_layers: destination qsa_states is null`
- `split_handoff_layers: conversation state sizing failed`
- `split_handoff_layers: invalid GDN state size`
- `split_handoff_layers: source stream synchronization failed`
- `split_handoff_layers: destination stream synchronization failed`
- `split_handoff_layers: GDN peer copy failed`
- `split_handoff_layers: QSA index state pointer is null`
- `split_handoff_layers: QSA save failed`
- `split_handoff_layers: QSA restore failed`
- `split_handoff_layers: QSA tail peer copy failed`
- `split_handoff_layers: QSA dead peer copy failed`
- `split_handoff_layers: QSA block-position peer copy failed`
- `split_handoff_layers: source device synchronization failed`
- `split_handoff_layers: destination device synchronization failed`

When `conversation_state_sizes`, `conversation_kv_save` or `conversation_kv_restore` provides a more specific error
string, it is kept as is; the sizing, save and restore messages above are used only when the lower layer provides none.

## Threading/streams

The function first synchronizes `src_stream` and `dst_stream`, each in its own device context, then runs the
synchronous host-mediated snapshot operations and `cudaMemcpyPeer`. Before returning it calls `cudaDeviceSynchronize`
once on `src_dev` and once on `dst_dev`, and attempts both synchronizations even when a transfer has failed. The
function provides no concurrency protection for a session.

## Cost

Let `G` be the number of GDN layers in the range, `Q` the number of QSA layers, and
`gdn_row = sizes.gdn / g.n_gdn_layers()`. GDN peer traffic is `G * gdn_row` bytes. Each QSA layer moves the K/V and
pooled index of the first `n_tokens` positions through a host snapshot, plus `sizes.tail + sizes.dead +
sizes.block_pos` bytes of peer traffic. There are two stream synchronizations and two device synchronizations in
total; the time grows roughly linearly with `G`, `Q` and `n_tokens`.

## Example call

```cpp
strata::gfx906::HandoffStats stats;
std::string err;
if (!strata::gfx906::split_handoff_layers(remote, 1, remote_stream,
                                           decode, 0, decode_stream,
                                           geometry, split_layer, geometry.n_layers,
                                           prompt_tokens, stats, err)) {
    report(err);
}
```
