#pragma once

#include <cstdint>
#include <functional>
#include <string>

namespace strata::gfx906 {

using ChunkFn = std::function<bool(const float* R_rows, int64_t T, int64_t pos0, std::string& err)>;

// doc: docs/fn/RowRelay.md
class RowRelay {
public:
    // doc: docs/fn/RowRelay.md
    RowRelay() = default;
    // doc: docs/fn/RowRelay.md
    ~RowRelay();
    // doc: docs/fn/RowRelay.md
    RowRelay(const RowRelay&) = delete;
    // doc: docs/fn/RowRelay.md
    RowRelay& operator=(const RowRelay&) = delete;
    // doc: docs/fn/RowRelay.md
    RowRelay(RowRelay&& other) noexcept;
    // doc: docs/fn/RowRelay.md
    RowRelay& operator=(RowRelay&& other) noexcept;

    // doc: docs/fn/RowRelay.md
    bool init(int src_dev, int dst_dev, int64_t row_floats, int64_t piece, std::string& err);
    // doc: docs/fn/RowRelay.md
    ChunkFn wrap(ChunkFn fn) const;

private:
    void release() noexcept;

    int src_dev_ = -1;
    int dst_dev_ = -1;
    int64_t row_floats_ = 0;
    int64_t piece_ = 0;
    float* rows_ = nullptr;
};

}  // namespace strata::gfx906
