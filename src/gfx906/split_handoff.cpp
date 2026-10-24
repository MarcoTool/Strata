#include "strata/gfx906/split_handoff.hpp"

#include "strata/core/conversation_snapshot.hpp"
#include "strata/core/on_device.hpp"
#include "strata/core/session.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cstddef>
#include <cstdint>

namespace strata::gfx906 {
namespace {

using Clock = std::chrono::steady_clock;

bool owns_ordinal(int64_t ordinal, int64_t first, int64_t count) {
    return count > 0 && ordinal >= first && ordinal - first < count;
}

void set_if_empty(std::string& err, const char* message) {
    if (err.empty()) err = message;
}

bool sync_device(int device) {
    const core::OnDevice on(device);
    return cudaDeviceSynchronize() == cudaSuccess;
}

}  // namespace

// doc: docs/fn/split_handoff_layers.md
bool split_handoff_layers(const core::SessionState& src, int src_dev, void* src_stream,
                          const core::SessionState& dst, int dst_dev, void* dst_stream,
                          const core::ModelGeometry& g, int64_t lb, int64_t le, int64_t n_tokens,
                          HandoffStats& stats, std::string& err) {
    stats = {};
    err.clear();

    if (lb < 0 || lb >= le || le > g.n_layers) {
        err = "split_handoff_layers: invalid layer range";
        return false;
    }
    if (g.qsa_interval <= 0) {
        err = "split_handoff_layers: qsa_interval must be positive";
        return false;
    }
    if (n_tokens <= 0) {
        err = "split_handoff_layers: n_tokens must be positive";
        return false;
    }
    if (n_tokens > src.max_cells) {
        err = "split_handoff_layers: n_tokens exceeds source session capacity";
        return false;
    }
    if (n_tokens > dst.max_cells) {
        err = "split_handoff_layers: n_tokens exceeds destination session capacity";
        return false;
    }
    if (src_dev < 0) {
        err = "split_handoff_layers: src_dev must be non-negative";
        return false;
    }
    if (dst_dev < 0) {
        err = "split_handoff_layers: dst_dev must be non-negative";
        return false;
    }
    if (!src_stream) {
        err = "split_handoff_layers: src_stream is null";
        return false;
    }
    if (!dst_stream) {
        err = "split_handoff_layers: dst_stream is null";
        return false;
    }

    int64_t gdn_count = 0;
    int64_t qsa_count = 0;
    for (int64_t l = lb; l < le; ++l) {
        if (core::is_qsa_layer(g, l)) {
            const int64_t qo = l / g.qsa_interval;
            if (!owns_ordinal(qo, src.qsa_ord0, src.qsa_alloc)) {
                err = "split_handoff_layers: source session does not own QSA range";
                return false;
            }
            if (!owns_ordinal(qo, dst.qsa_ord0, dst.qsa_alloc)) {
                err = "split_handoff_layers: destination session does not own QSA range";
                return false;
            }
            ++qsa_count;
        } else {
            const int64_t go = l - l / g.qsa_interval;
            if (!owns_ordinal(go, src.gdn_ord0, src.gdn_alloc)) {
                err = "split_handoff_layers: source session does not own GDN range";
                return false;
            }
            if (!owns_ordinal(go, dst.gdn_ord0, dst.gdn_alloc)) {
                err = "split_handoff_layers: destination session does not own GDN range";
                return false;
            }
            ++gdn_count;
        }
    }
    if (gdn_count && !src.gdn_state) {
        err = "split_handoff_layers: source gdn_state is null";
        return false;
    }
    if (gdn_count && !dst.gdn_state) {
        err = "split_handoff_layers: destination gdn_state is null";
        return false;
    }
    if (qsa_count && !src.qsa_states) {
        err = "split_handoff_layers: source qsa_states is null";
        return false;
    }
    if (qsa_count && !dst.qsa_states) {
        err = "split_handoff_layers: destination qsa_states is null";
        return false;
    }

    core::ConversationStateSizes sizes;
    if (!core::conversation_state_sizes(g, sizes, err)) {
        set_if_empty(err, "split_handoff_layers: conversation state sizing failed");
        return false;
    }
    if (gdn_count && (g.n_gdn_layers() <= 0 || sizes.gdn % static_cast<size_t>(g.n_gdn_layers()) != 0)) {
        err = "split_handoff_layers: invalid GDN state size";
        return false;
    }

    const uint64_t gdn_row = g.n_gdn_layers() > 0
        ? static_cast<uint64_t>(sizes.gdn) / static_cast<uint64_t>(g.n_gdn_layers())
        : 0;
    const Clock::time_point started = Clock::now();

    bool ok = true;
    {
        const core::OnDevice on(src_dev);
        if (cudaStreamSynchronize(static_cast<cudaStream_t>(src_stream)) != cudaSuccess) {
            err = "split_handoff_layers: source stream synchronization failed";
            ok = false;
        }
    }
    {
        const core::OnDevice on(dst_dev);
        if (cudaStreamSynchronize(static_cast<cudaStream_t>(dst_stream)) != cudaSuccess) {
            if (ok) err = "split_handoff_layers: destination stream synchronization failed";
            ok = false;
        }
    }

    for (int64_t l = lb; ok && l < le; ++l) {
        if (!core::is_qsa_layer(g, l)) {
            const int64_t go = l - l / g.qsa_interval;
            const char* from = reinterpret_cast<const char*>(src.gdn_state) +
                               static_cast<uint64_t>(go - src.gdn_ord0) * gdn_row;
            char* to = reinterpret_cast<char*>(dst.gdn_state) +
                       static_cast<uint64_t>(go - dst.gdn_ord0) * gdn_row;
            if (cudaMemcpyPeer(to, dst_dev, from, src_dev, static_cast<size_t>(gdn_row)) != cudaSuccess) {
                err = "split_handoff_layers: GDN peer copy failed";
                ok = false;
                break;
            }
            stats.bytes += gdn_row;
            ++stats.gdn_layers;
            continue;
        }

        const int64_t qo = l / g.qsa_interval;
        const core::QsaState& from = src.qsa_states[qo];
        const core::QsaState& to = dst.qsa_states[qo];
        if ((sizes.tail && (!from.idx_tail || !to.idx_tail)) ||
            (sizes.dead && (!from.idx_dead || !to.idx_dead)) ||
            (sizes.block_pos && (!from.idx_block_pos || !to.idx_block_pos))) {
            err = "split_handoff_layers: QSA index state pointer is null";
            ok = false;
            break;
        }

        core::ConversationKv image;
        {
            const core::OnDevice on(src_dev);
            ok = core::conversation_kv_save(image, from, g, n_tokens, true, err);
        }
        if (!ok) {
            set_if_empty(err, "split_handoff_layers: QSA save failed");
            break;
        }
        {
            const core::OnDevice on(dst_dev);
            ok = core::conversation_kv_restore(image, to, g, n_tokens, true, err);
        }
        stats.bytes += static_cast<uint64_t>(image.bytes());
        if (!ok) {
            set_if_empty(err, "split_handoff_layers: QSA restore failed");
            break;
        }
        if (cudaMemcpyPeer(to.idx_tail, dst_dev, from.idx_tail, src_dev, sizes.tail) != cudaSuccess) {
            err = "split_handoff_layers: QSA tail peer copy failed";
            ok = false;
            break;
        }
        if (cudaMemcpyPeer(to.idx_dead, dst_dev, from.idx_dead, src_dev, sizes.dead) != cudaSuccess) {
            err = "split_handoff_layers: QSA dead peer copy failed";
            ok = false;
            break;
        }
        if (cudaMemcpyPeer(to.idx_block_pos, dst_dev, from.idx_block_pos, src_dev, sizes.block_pos) != cudaSuccess) {
            err = "split_handoff_layers: QSA block-position peer copy failed";
            ok = false;
            break;
        }
        ++stats.qsa_layers;
    }

    const bool src_synced = sync_device(src_dev);
    const bool dst_synced = sync_device(dst_dev);
    if (!src_synced) {
        if (ok) err = "split_handoff_layers: source device synchronization failed";
        ok = false;
    }
    if (!dst_synced) {
        if (ok) err = "split_handoff_layers: destination device synchronization failed";
        ok = false;
    }
    stats.ms = std::chrono::duration<double, std::milli>(Clock::now() - started).count();
    return ok;
}

}  // namespace strata::gfx906
