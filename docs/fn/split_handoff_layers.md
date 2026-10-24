# `split_handoff_layers`

## Purpose

把 split-prefill 後位於 `src_dev`、層範圍 `[lb, le)` 的 prompt 狀態搬到 `dst_dev` 的 decode session。GDN 逐層複製 recurrent/conv state；QSA 透過 conversation snapshot API 搬移前 `n_tokens` 個位置的 K/V 與 pooled index，並另外複製 indexer 的 tail、dead key 與 block position。

## Signature

```cpp
bool split_handoff_layers(const core::SessionState& src, int src_dev, void* src_stream,
                          const core::SessionState& dst, int dst_dev, void* dst_stream,
                          const core::ModelGeometry& g, int64_t lb, int64_t le, int64_t n_tokens,
                          HandoffStats& stats, std::string& err);
```

## Inputs

- `src`：來源 `SessionState`；host 端描述物件，本函式不取得所有權。其狀態指標指向 `src_dev` device memory，且必須擁有 `[lb, le)` 對應的 QSA/GDN ordinal。
- `src_dev`：來源 CUDA device ordinal，無單位；必須為非負值。
- `src_stream`：來源 device 上的 `cudaStream_t`，以 `void*` 傳入；不取得所有權，不可為 null。
- `dst`：目的 `SessionState`；host 端描述物件，本函式不取得所有權。其狀態指標指向 `dst_dev` device memory，且必須擁有 `[lb, le)` 對應的 QSA/GDN ordinal。
- `dst_dev`：目的 CUDA device ordinal，無單位；必須為非負值。
- `dst_stream`：目的 device 上的 `cudaStream_t`，以 `void*` 傳入；不取得所有權，不可為 null。
- `g`：model geometry，host memory、唯讀、不取得所有權；`n_layers` 與 `qsa_interval` 決定層型別及 global ordinal。
- `lb`：第一個要搬移的 global layer index，單位為 layer，包含此層。
- `le`：範圍終點，單位為 layer，不包含此層。
- `n_tokens`：要保存及還原的 prompt 長度，單位為 token；不可超過兩個 session 的 `max_cells`。
- `stats`：host 端輸出物件，由呼叫者持有；進入函式時會清零。
- `err`：host 端錯誤字串，由呼叫者持有；進入函式時會清空。

## Outputs

成功回傳 `true`；失敗回傳 `false`。`stats.gdn_layers` 與 `stats.qsa_layers` 是已完整搬移的層數，`stats.ms` 是從首次 stream synchronization 到兩端 device synchronization 完成的毫秒數。`stats.bytes` 沿用 inline reference 的統計方式：計入 GDN rows 與 `ConversationKv::bytes()`，不計入每個 QSA 額外 peer-copy 的 `tail`、`dead`、`block_pos`。

目的 session 的 `[lb, le)` 狀態會被覆寫。失敗可能發生在部分層已搬移之後；此時目的 session 不可視為完整 handoff。

## Preconditions

- `0 <= lb < le <= g.n_layers`，且 `g.qsa_interval > 0`。
- `n_tokens > 0`，並且不大於 `src.max_cells`、`dst.max_cells`。
- 兩個 session 的 `qsa_ord0/qsa_alloc` 與 `gdn_ord0/gdn_alloc` 必須涵蓋指定範圍。
- 所需的 `gdn_state`、`qsa_states` 與 QSA index state 指標均有效，且位於所宣告的 device。
- peer access 與 conversation snapshot API 所需資源已由呼叫者設定。
- handoff 期間沒有其他執行緒或 stream 修改來源或目的狀態。

## Errors

本函式直接產生下列 `err`：

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

`conversation_state_sizes`、`conversation_kv_save` 或 `conversation_kv_restore` 若提供更具體的錯誤字串，本函式會原樣保留；上述 sizing/save/restore 字串只在下層未提供訊息時使用。

## Threading/streams

函式先在各自 device context 同步 `src_stream` 與 `dst_stream`，再執行同步的 host-mediated snapshot 操作及 `cudaMemcpyPeer`。結束前會在 `src_dev` 與 `dst_dev` 各呼叫一次 `cudaDeviceSynchronize`，即使中途搬移失敗也會嘗試同步兩端。本函式本身不提供同一 session 的併行保護。

## Cost

令 `G` 為範圍內 GDN 層數、`Q` 為 QSA 層數，`gdn_row = sizes.gdn / g.n_gdn_layers()`。GDN peer traffic 為 `G * gdn_row` bytes。每個 QSA 會經 host snapshot 搬移前 `n_tokens` 的 K/V 與 pooled index，另有 `sizes.tail + sizes.dead + sizes.block_pos` bytes 的 peer traffic。總同步次數包含兩次 stream sync 與兩次 device sync；時間大致隨 `G`、`Q` 與 `n_tokens` 線性增加。

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
