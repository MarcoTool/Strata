#include "strata/gfx906/row_relay.hpp"

#include "strata/core/on_device.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <utility>

namespace strata::gfx906 {

// doc: docs/fn/RowRelay.md
RowRelay::~RowRelay() {
    release();
}

// doc: docs/fn/RowRelay.md
RowRelay::RowRelay(RowRelay&& other) noexcept
    : src_dev_(std::exchange(other.src_dev_, -1)),
      dst_dev_(std::exchange(other.dst_dev_, -1)),
      row_floats_(std::exchange(other.row_floats_, 0)),
      piece_(std::exchange(other.piece_, 0)),
      rows_(std::exchange(other.rows_, nullptr)) {}

// doc: docs/fn/RowRelay.md
RowRelay& RowRelay::operator=(RowRelay&& other) noexcept {
    if (this == &other) return *this;
    release();
    src_dev_ = std::exchange(other.src_dev_, -1);
    dst_dev_ = std::exchange(other.dst_dev_, -1);
    row_floats_ = std::exchange(other.row_floats_, 0);
    piece_ = std::exchange(other.piece_, 0);
    rows_ = std::exchange(other.rows_, nullptr);
    return *this;
}

// doc: docs/fn/RowRelay.md
bool RowRelay::init(int src_dev, int dst_dev, int64_t row_floats, int64_t piece, std::string& err) {
    err.clear();
    if (src_dev < 0) {
        err = "RowRelay::init: src_dev must be non-negative";
        return false;
    }
    if (dst_dev < 0) {
        err = "RowRelay::init: dst_dev must be non-negative";
        return false;
    }
    if (row_floats <= 0) {
        err = "RowRelay::init: row_floats must be positive";
        return false;
    }
    if (piece <= 0) {
        err = "RowRelay::init: piece must be positive";
        return false;
    }
    constexpr size_t max = std::numeric_limits<size_t>::max();
    if (static_cast<uint64_t>(row_floats) > max / sizeof(float) ||
        static_cast<uint64_t>(piece) > max / (static_cast<size_t>(row_floats) * sizeof(float))) {
        err = "RowRelay::init: buffer size overflow";
        return false;
    }

    const size_t bytes = static_cast<size_t>(piece) * static_cast<size_t>(row_floats) * sizeof(float);
    float* fresh = nullptr;
    {
        const core::OnDevice on(dst_dev);
        if (cudaMalloc(reinterpret_cast<void**>(&fresh), bytes) != cudaSuccess) {
            err = "RowRelay::init: cudaMalloc failed";
            return false;
        }
    }

    release();
    src_dev_ = src_dev;
    dst_dev_ = dst_dev;
    row_floats_ = row_floats;
    piece_ = piece;
    rows_ = fresh;
    return true;
}

// doc: docs/fn/RowRelay.md
ChunkFn RowRelay::wrap(ChunkFn fn) const {
    const int src_dev = src_dev_;
    const int dst_dev = dst_dev_;
    const int64_t row_floats = row_floats_;
    const int64_t piece = piece_;
    float* const rows = rows_;
    return [src_dev, dst_dev, row_floats, piece, rows, fn = std::move(fn)](
               const float* R_rows, int64_t T, int64_t pos0, std::string& err) mutable {
        err.clear();
        if (!rows || row_floats <= 0 || piece <= 0 || src_dev < 0 || dst_dev < 0) {
            err = "RowRelay::wrap: relay is not initialized";
            return false;
        }
        if (!fn) {
            err = "RowRelay::wrap: callback is empty";
            return false;
        }
        if (T < 0) {
            err = "RowRelay::wrap: T must be non-negative";
            return false;
        }
        if (T > 0 && !R_rows) {
            err = "RowRelay::wrap: R_rows is null";
            return false;
        }
        constexpr size_t max = std::numeric_limits<size_t>::max();
        if (static_cast<uint64_t>(T) > max / static_cast<size_t>(row_floats)) {
            err = "RowRelay::wrap: input size overflow";
            return false;
        }
        if (T > 0 && pos0 > std::numeric_limits<int64_t>::max() - (T - 1)) {
            err = "RowRelay::wrap: position overflow";
            return false;
        }

        const core::OnDevice on(dst_dev);
        for (int64_t a = 0; a < T; a += piece) {
            const int64_t n = std::min(piece, T - a);
            const size_t count = static_cast<size_t>(n) * static_cast<size_t>(row_floats);
            const float* from = R_rows + static_cast<size_t>(a) * static_cast<size_t>(row_floats);
            if (cudaMemcpyPeer(rows, dst_dev, from, src_dev, count * sizeof(float)) != cudaSuccess) {
                err = "RowRelay::wrap: peer copy failed";
                return false;
            }
            if (!fn(rows, n, pos0 + a, err)) {
                if (err.empty()) err = "RowRelay::wrap: callback failed";
                return false;
            }
        }
        return true;
    };
}

void RowRelay::release() noexcept {
    if (!rows_) return;
    const core::OnDevice on(dst_dev_);
    cudaFree(rows_);
    rows_ = nullptr;
    src_dev_ = -1;
    dst_dev_ = -1;
    row_floats_ = 0;
    piece_ = 0;
}

}  // namespace strata::gfx906
