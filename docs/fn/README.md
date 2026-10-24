# gfx906 function index

| Function / class | Header | Documentation | Summary |
|---|---|---|---|
| `split_handoff_layers` / `HandoffStats` | `include/strata/gfx906/split_handoff.hpp` | [`split_handoff_layers.md`](split_handoff_layers.md) | 將 split-prefill 的 GDN 與 QSA prompt state 搬回 decode device，並回報傳輸統計。 |
| `RowRelay` / `ChunkFn` | `include/strata/gfx906/row_relay.hpp` | [`RowRelay.md`](RowRelay.md) | 以固定大小的 destination-device buffer 分段 relay rows，再呼叫目的端 callback。 |
