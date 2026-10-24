# `RowRelay`

## Purpose

`RowRelay` 在 `dst_dev` 擁有一塊 `piece * row_floats` 的 device buffer。`wrap(fn)` 產生的 callback 會把位於 `src_dev` 的 rows 分段 peer-copy 到這塊 buffer，並在 `dst_dev` context 下依序呼叫 `fn`。類別不可複製、可以 move，析構時釋放 buffer。

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

`RowRelay::init`：

- `src_dev`：`R_rows` 所在的來源 CUDA device ordinal，無單位。
- `dst_dev`：relay buffer 與 wrapped callback 執行所在的 CUDA device ordinal，無單位。
- `row_floats`：每列的 `float` 數，單位為 float element。
- `piece`：單次最多搬移的列數，單位為 row；gfx906 split-prefill 的典型值為 `1024`。
- `err`：host 端錯誤字串，由呼叫者持有；進入 `init` 時會清空。

`RowRelay::wrap`：

- `fn`：由呼叫者提供並以 value 捕捉的 `ChunkFn`；每段會收到位於 `dst_dev` 的暫存 rows、該段列數與修正後的起始位置。callback 可在呼叫間保存自身 host state，但不得取得 relay buffer 的所有權。

wrapped `ChunkFn`：

- `R_rows`：位於 `src_dev` 的 device pointer，資料排列為 `T * row_floats` 個連續 `float`；不取得所有權。`T == 0` 時可為 null。
- `T`：輸入列數，單位為 row，必須非負。
- `pos0`：第一列的 logical token position，單位為 position；第 `a` 段傳給 `fn` 的值為 `pos0 + a`。
- `err`：host 端錯誤字串，由呼叫者持有；wrapped callback 進入時會清空，之後 relay 或 `fn` 可寫入。

## Outputs

`init` 成功配置 buffer 時回傳 `true`；失敗回傳 `false`，並保留原本已初始化的 buffer。再次成功 `init` 會釋放舊 buffer並換成新配置。`wrap` 回傳新的 `ChunkFn`。wrapped callback 成功處理所有列時回傳 `true`，copy 或下游 callback 失敗時回傳 `false`。

move construction/assignment 轉移 device buffer 所有權；moved-from 物件回到未初始化狀態。析構與 move assignment 會在原 `dst_dev` context 呼叫 `cudaFree`。每次呼叫下游 `fn` 前，relay buffer 的內容會被當前 chunk 覆寫。

## Preconditions

- `src_dev >= 0`、`dst_dev >= 0`、`row_floats > 0`、`piece > 0`，且乘積可由 `size_t` 表示。
- 呼叫 wrapped callback 前 `init` 已成功，且 `fn` 非空。
- `R_rows` 的 device allocation 至少包含 `T * row_floats * sizeof(float)` bytes，並位於 `src_dev`。
- `T * row_floats` 必須可由 `size_t` 表示，且 `T > 0` 時 `pos0 + T - 1` 必須可由 `int64_t` 表示。
- CUDA peer access 已設定。
- `RowRelay` 的 buffer owner 必須活得比所有由它建立且仍會被呼叫的 wrappers 更久；move 不改變 buffer address，但 owner 析構後 wrapper 即不可再使用。

## Errors

此元件產生下列 `err`：

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

若下游 `fn` 回傳 `false` 且已填入 `err`，relay 會保留該訊息；只有在 `err` 為空時才寫入 `RowRelay::wrap: callback failed`。析構與 move cleanup 無法回報 `cudaFree` 錯誤。

## Threading/streams

wrapped callback 使用同步 `cudaMemcpyPeer`，沒有接受 stream，也不會自行做 stream synchronization。copy 與 `fn` 都在 `core::OnDevice(dst_dev)` scope 內依序執行。單一 `RowRelay` 只有一塊共用 buffer，因此其 wrappers 不可同時呼叫；若需併行，應為每條執行路徑建立獨立 relay。


**呼叫端責任（整合時補註）：**

- `wrap()` 回傳的 callback 只捕捉 buffer 的裸指標，**`RowRelay` 物件必須活得比所有 wrapper 久**；relay 先被解構後再呼叫 wrapper 是未定義行為。
- `R_rows` 若是在 non-blocking stream 上產生，`cudaMemcpyPeer` 走 null stream **不會**等那條 stream；呼叫 wrapper 前，產生 rows 的 stream 必須已經同步。Strata prefill 呼叫 `on_chunk` 前會先 `cudaStreamSynchronize(m.cs)`（`src/prefill/prefill.cpp`，`on_chunk(m.R, T, p0, err)` 那段），因此用作 `Prefill::on_chunk` 時不需額外同步。

## Cost

常駐 device memory 為 `piece * row_floats * sizeof(float)` bytes。處理 `T` 列會 peer-copy `T * row_floats * sizeof(float)` bytes，執行 `ceil(T / piece)` 次 copy 與相同次數的 `fn`。例如 `piece = 1024`、`row_floats = 10240` 時，buffer 約為 40 MiB；處理 2500 列會分成 1024、1024、452 三段。

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
