// --scratch-context: a second, small session beside the main one for side requests (`lane=scratch`), so a short
// side call - a title, a tool's answer, a classification - never replaces the main conversation's cells.  The side
// session has its own prompt path, verifier (its graphs are captured over its own state), draft K/V and checkpoints;
// the weights, the expert cache, the CPU pool and the draft weights are shared.  This header holds the rules that do
// not need the engine's state (what the side session can run beside), so they can be read and tested on their own.
#pragma once

#include <cstdint>

namespace strata::program::scratch_lane {

// Why the side session cannot be made with these settings, or nullptr when it can.  It is one more session on the
// first GPU, served one request at a time beside the main one; the features below reshape or multiply the main
// session, and a second session would not follow them.
inline const char* refusal(bool serve, bool layer_split, int batch, bool kv_elastic, int peer_device,
                           int pipeline_windows, int64_t scratch_context, int64_t max_context) {
    if (!serve) return "only --serve takes lane=scratch requests";
    if (scratch_context < 1024) return "it needs at least 1024 cells";
    if (scratch_context >= max_context) return "it must be smaller than --max-context (it is the small side session)";
    if (layer_split) return "a layer split carves every session per stage";
    if (batch > 0) return "--batch already keeps several conversations";
    if (kv_elastic) return "the elastic K/V (--kv-grow) resizes every session's pools together";
    if (peer_device >= 1) return "--peer-device mirrors the main session";
    if (pipeline_windows >= 2) return "--pipeline-windows runs two verifiers over the main session";
    return nullptr;
}

}  // namespace strata::program::scratch_lane
