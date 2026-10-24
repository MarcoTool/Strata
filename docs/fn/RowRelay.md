# `RowRelay`

## Purpose

`RowRelay` owns a device buffer of `piece * row_floats` floats on `dst_dev`. The callback returned by `wrap(fn)`
peer-copies rows that live on `src_dev` into this buffer piece by piece, and calls `fn` for each piece in the
`dst_dev` context. The class is move-only; the destructor frees the buffer.

## Signature

```cpp
using ChunkFn = std::function<bool(const float* R_rows, int64_t T,
                                   int64_t pos0, std::string& err)>;

class RowRelay {
public:
    RowRelay() = default;
    ~RowRelay();
    RowRelay(const RowRelay&) = delete;
    RowRelay& operator=(const RowRelay&) = delete;
    RowRelay(RowRelay&& other) noexcept;
    RowRelay& operator=(RowRelay&& other) noexcept;

    bool init(int src_dev, int dst_dev, int64_t row_floats,
              int64_t piece, std::string& err);
    ChunkFn wrap(ChunkFn fn) const;
};
```

## Inputs

`RowRelay::init`:

- `src_dev`: CUDA device ordinal that holds `R_rows`.
- `dst_dev`: CUDA device ordinal of the relay buffer, where the wrapped callback runs.
- `row_floats`: floats per row.
- `piece`: maximum rows moved per copy; `1024` in the gfx906 split prefill.
- `err`: host error string owned by the caller; cleared on entry.

`RowRelay::wrap`:

- `fn`: a `ChunkFn` supplied by the caller and captured by value. Each piece receives the staged rows on `dst_dev`,
  the number of rows in the piece and the adjusted start position. The callback may keep its own host state between
  calls but must not take ownership of the relay buffer.

The wrapped `ChunkFn`:

- `R_rows`: device pointer on `src_dev` to `T * row_floats` contiguous floats; not owned. May be null when `T == 0`.
- `T`: number of input rows; must be non-negative.
- `pos0`: logical token position of the first row; the piece starting at row `a` passes `pos0 + a` to `fn`.
- `err`: host error string owned by the caller; cleared when the wrapped callback is entered, then written by the
  relay or by `fn`.

## Outputs

`init` returns `true` when the buffer is allocated and `false` on failure, keeping any buffer from an earlier
successful `init`. A second successful `init` frees the old buffer and replaces it. `wrap` returns a new `ChunkFn`. The
wrapped callback returns `true` when every row is processed and `false` when a copy or the downstream callback fails.

Move construction and move assignment transfer ownership of the device buffer; the moved-from object returns to the
uninitialized state. The destructor and move assignment call `cudaFree` in the original `dst_dev` context. The buffer
contents are overwritten by the current piece before each call to `fn`.

## Preconditions

- `src_dev >= 0`, `dst_dev >= 0`, `row_floats > 0`, `piece > 0`, and their product fits in `size_t`.
- `init` has succeeded and `fn` is not empty before the wrapped callback is called.
- The allocation behind `R_rows` holds at least `T * row_floats * sizeof(float)` bytes on `src_dev`.
- `T * row_floats` fits in `size_t`, and for `T > 0`, `pos0 + T - 1` fits in `int64_t`.
- CUDA peer access is enabled.
- The `RowRelay` that owns the buffer outlives every wrapper created from it that may still be called. A move does not
  change the buffer address, but no wrapper may be used after the owner is destroyed.

## Errors

This component writes the following to `err`:

- `RowRelay::init: src_dev must be non-negative`
- `RowRelay::init: dst_dev must be non-negative`
- `RowRelay::init: row_floats must be positive`
- `RowRelay::init: piece must be positive`
- `RowRelay::init: buffer size overflow`
- `RowRelay::init: cudaMalloc failed`
- `RowRelay::wrap: relay is not initialized`
- `RowRelay::wrap: callback is empty`
- `RowRelay::wrap: T must be non-negative`
- `RowRelay::wrap: R_rows is null`
- `RowRelay::wrap: input size overflow`
- `RowRelay::wrap: position overflow`
- `RowRelay::wrap: peer copy failed`
- `RowRelay::wrap: callback failed`

When the downstream `fn` returns `false` and has set `err`, the relay keeps that message; it writes
`RowRelay::wrap: callback failed` only when `err` is empty. The destructor and move cleanup cannot report `cudaFree`
errors.

## Threading/streams

The wrapped callback uses the synchronous `cudaMemcpyPeer`; it takes no stream and does no stream synchronization of
its own. The copy and `fn` run in order inside a `core::OnDevice(dst_dev)` scope. A `RowRelay` has a single shared
buffer, so its wrappers must not be called concurrently; use one relay per concurrent path.

Caller responsibilities:

- The callback returned by `wrap()` captures only a raw pointer to the buffer, so **the `RowRelay` must outlive every
  wrapper**. Calling a wrapper after the relay is destroyed is undefined behavior.
- If `R_rows` is produced on a non-blocking stream, `cudaMemcpyPeer` on the null stream does **not** wait for it; the
  producing stream must be synchronized before the wrapper is called. Strata's prefill calls
  `cudaStreamSynchronize(m.cs)` before `on_chunk` (`src/prefill/prefill.cpp`, at `on_chunk(m.R, T, p0, err)`), so no
  extra synchronization is needed when the wrapper is used as `Prefill::on_chunk`.

## Cost

Resident device memory is `piece * row_floats * sizeof(float)` bytes. Processing `T` rows peer-copies
`T * row_floats * sizeof(float)` bytes in `ceil(T / piece)` copies, with the same number of calls to `fn`. For example,
with `piece = 1024` and `row_floats = 10240` the buffer is about 40 MiB, and 2,500 rows are processed as pieces of
1024, 1024 and 452.

## Example call

```cpp
strata::gfx906::RowRelay relay;
std::string err;
if (!relay.init(1, 0, geometry.hc_dim(), 1024, err)) {
    report(err);
}

auto on_dst = relay.wrap(process_rows_on_decode_gpu);
if (!on_dst(rows_on_gpu1, token_count, first_position, err)) {
    report(err);
}
```
