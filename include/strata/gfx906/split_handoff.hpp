#pragma once

#include <cstdint>
#include <string>

namespace strata::core {
struct ModelGeometry;
struct SessionState;
}  // namespace strata::core

namespace strata::gfx906 {

// doc: docs/fn/split_handoff_layers.md
struct HandoffStats {
    uint64_t bytes = 0;
    double ms = 0;
    int64_t gdn_layers = 0, qsa_layers = 0;
};

// doc: docs/fn/split_handoff_layers.md
bool split_handoff_layers(const core::SessionState& src, int src_dev, void* src_stream,
                          const core::SessionState& dst, int dst_dev, void* dst_stream,
                          const core::ModelGeometry& g, int64_t lb, int64_t le, int64_t n_tokens,
                          HandoffStats& stats, std::string& err);

}  // namespace strata::gfx906
