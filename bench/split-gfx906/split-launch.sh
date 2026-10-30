#!/usr/bin/env bash
#
# Usage: the "exe" of a split_serve model config (make_configs.py writes it there).
#
# Stock 0.2.7 serve/split.py starts both engines of a split with one executable. This
# script picks the build per role, so a CUDA card can read the prompt and gfx906 cards
# can write the answer:
#   - the prefill role carries --split-role-prefill in "prefill_args"; the marker is
#     removed and the CUDA build runs on the NVIDIA card;
#   - every other start is the decode role: the gfx906 build on the MI50s, with the
#     gfx906 opt-in switches used in the measurements.
#
# Environment (all optional):
#   STRATA_SPLIT_CUDA_EXE     CUDA build      (default: <repo>/build-cuda86/strata)
#   STRATA_SPLIT_HIP_EXE      gfx906 build    (default: <repo>/build-906/strata)
#   STRATA_SPLIT_CUDA_DEVICE  NVIDIA card for the prefill role      (default: 0)
#   STRATA_SPLIT_HIP_DEVICES  MI50s for the decode role, "0,1" or "1" (default: 0,1)
#   STRATA_SPLIT_CUDA_LIBS    directory with libcudart/libcublas 13 (default: /usr/local/cuda/lib64)
# STRATA_GFX906_TOPK_SPLIT=1 (#3987) passes through to the decode role when it is set.
set -u
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cuda_exe=${STRATA_SPLIT_CUDA_EXE:-$repo/build-cuda86/strata}
hip_exe=${STRATA_SPLIT_HIP_EXE:-$repo/build-906/strata}

args=()
prefill=0
for a in "$@"; do
  if [ "$a" = --split-role-prefill ]; then prefill=1; else args+=("$a"); fi
done

if [ $prefill = 1 ]; then
  # prefill role: CUDA build, NVIDIA card only
  unset HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES
  export CUDA_VISIBLE_DEVICES=${STRATA_SPLIT_CUDA_DEVICE:-0}
  libs=${STRATA_SPLIT_CUDA_LIBS:-/usr/local/cuda/lib64}
  export LD_LIBRARY_PATH=$libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
  exec "$cuda_exe" "${args[@]}"
fi

# decode role: gfx906 build on the MI50s
export HIP_VISIBLE_DEVICES=${STRATA_SPLIT_HIP_DEVICES:-0,1}
export HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_GFX_VERSION:-9.0.6}
export STRATA_GR_SPLIT=1 STRATA_GFX906_ATTN_QUERY_SWIZZLE=1 STRATA_GFX906_ATTN_REDUCE12=1
exec "$hip_exe" "${args[@]}"
