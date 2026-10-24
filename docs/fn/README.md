# gfx906 function index

| Function / class | Header | Documentation | Summary |
|---|---|---|---|
| `split_handoff_layers` / `HandoffStats` | `include/strata/gfx906/split_handoff.hpp` | [`split_handoff_layers.md`](split_handoff_layers.md) | Moves the GDN and QSA prompt state of a split prefill back to the decode device and reports transfer statistics. |
| `RowRelay` / `ChunkFn` | `include/strata/gfx906/row_relay.hpp` | [`RowRelay.md`](RowRelay.md) | Relays rows in pieces through a fixed-size buffer on the destination device, then calls a callback there. |
