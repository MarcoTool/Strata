// tools/hip/tp_gdn_probe.cpp - what splitting a GDN block's projections over two gfx906 cards buys, handoffs included.
//
// Qwen3.8-Flash-Next's GDN layer (36 of 48): qkv 2560 -> 10240, gate z 2560 -> 6144, out 6144 -> 2560, Q6_K, a verify
// window of NCOL tokens.  ONE card runs the three full matrices per layer.  TWO cards each run their half of the heads
// (qkv and z: half the output rows; out: half the input columns, a partial sum), with per layer the window's input
// sent from card 0 to card 1 and card 1's partial output sent back (P2P copies ordered by events) and added on card 0.
// Synthetic weights: the time depends on the shapes and the format, not on the values.
#include "strata/kernels/native_mmvq.hpp"

#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace k = strata::kernels;
#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::printf("%s: %s\n", #x, hipGetErrorString(e_)); std::exit(1); } } while (0)

__global__ void add_into(float* __restrict__ y, const float* __restrict__ p, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) y[i] += p[i];
}

struct Mats { void *qkv, *z, *out; float* x; void* xq; float* xo; void* xoq; float *yqkv, *yz, *yout; };

static void* weights(int type, int n_in, int n_out, std::mt19937& rng) {
    const size_t b = k::native_mmvq_weight_bytes(type, n_in, n_out);
    std::vector<uint8_t> h(b);
    for (auto& v : h) v = uint8_t(rng());
    void* d; CK(hipMalloc(&d, b)); CK(hipMemcpy(d, h.data(), b, hipMemcpyHostToDevice));
    return d;
}

static Mats make(int type, int frac, int ncol, std::mt19937& rng) {   // frac 1: whole, 2: a half
    Mats m{};
    m.qkv = weights(type, 2560, 10240 / frac, rng);
    m.z = weights(type, 2560, 6144 / frac, rng);
    m.out = weights(type, 6144 / frac, 2560, rng);
    CK(hipMalloc(&m.x, sizeof(float) * 2560 * ncol));
    CK(hipMalloc(&m.xq, k::native_q8_1_bytes(2560, ncol)));
    CK(hipMalloc(&m.xo, sizeof(float) * 6144 * ncol));
    CK(hipMalloc(&m.xoq, k::native_q8_1_bytes(6144, ncol)));
    CK(hipMalloc(&m.yqkv, sizeof(float) * 10240 * ncol));
    CK(hipMalloc(&m.yz, sizeof(float) * 6144 * ncol));
    CK(hipMalloc(&m.yout, sizeof(float) * 2560 * ncol));
    std::vector<float> hx(6144 * ncol);
    std::uniform_real_distribution<float> u(-1.f, 1.f);
    for (auto& v : hx) v = u(rng);
    CK(hipMemcpy(m.x, hx.data(), sizeof(float) * 2560 * ncol, hipMemcpyHostToDevice));
    CK(hipMemcpy(m.xo, hx.data(), sizeof(float) * 6144 * ncol, hipMemcpyHostToDevice));
    return m;
}

// one layer's projections on the current card: quantize the input, qkv and z, quantize the out input, out
static void layer(const Mats& m, int type, int frac, int ncol, hipStream_t s) {
    k::native_quantize_q8_1(m.x, m.xq, 2560, ncol, s);
    k::native_mmvq(type, m.qkv, m.xq, m.yqkv, 2560, 10240 / frac, ncol, s);
    k::native_mmvq(type, m.z, m.xq, m.yz, 2560, 6144 / frac, ncol, s);
    k::native_quantize_q8_1(m.xo, m.xoq, 6144 / frac, ncol, s);
    k::native_mmvq(type, m.out, m.xoq, m.yout, 6144 / frac, 2560, ncol, s);
}

int main(int argc, char** argv) {
    const int type = argc > 1 ? std::atoi(argv[1]) : 14;   // 14 = Q6_K
    const int ncol = argc > 2 ? std::atoi(argv[2]) : 4;
    const int layers = 36, reps = 50;
    std::mt19937 rng(7);
    CK(hipSetDevice(0)); CK(hipDeviceEnablePeerAccess(1, 0));
    CK(hipSetDevice(1)); CK(hipDeviceEnablePeerAccess(0, 0));
    CK(hipSetDevice(0)); Mats full = make(type, 1, ncol, rng), half0 = make(type, 2, ncol, rng);
    float* part; CK(hipMalloc(&part, sizeof(float) * 2560 * ncol));
    hipStream_t s0; CK(hipStreamCreate(&s0));
    CK(hipSetDevice(1)); Mats half1 = make(type, 2, ncol, rng);
    hipStream_t s1; CK(hipStreamCreate(&s1));
    hipEvent_t ex[layers], ep[layers];
    CK(hipSetDevice(0)); for (auto& e : ex) CK(hipEventCreateWithFlags(&e, hipEventDisableTiming));
    CK(hipSetDevice(1)); for (auto& e : ep) CK(hipEventCreateWithFlags(&e, hipEventDisableTiming));

    auto one = [&]() {
        CK(hipSetDevice(0));
        for (int l = 0; l < layers; ++l) layer(full, type, 1, ncol, s0);
        CK(hipStreamSynchronize(s0));
    };
    auto two = [&]() {
        for (int l = 0; l < layers; ++l) {
            CK(hipSetDevice(0));
            CK(hipMemcpyPeerAsync(half1.x, 1, half0.x, 0, sizeof(float) * 2560 * ncol, s0));   // the window's input to card 1
            CK(hipEventRecord(ex[l], s0));
            layer(half0, type, 2, ncol, s0);                                                    // card 0's heads
            CK(hipSetDevice(1));
            CK(hipStreamWaitEvent(s1, ex[l], 0));
            layer(half1, type, 2, ncol, s1);                                                    // card 1's heads
            CK(hipMemcpyPeerAsync(part, 0, half1.yout, 1, sizeof(float) * 2560 * ncol, s1));   // its partial out back
            CK(hipEventRecord(ep[l], s1));
            CK(hipSetDevice(0));
            CK(hipStreamWaitEvent(s0, ep[l], 0));
            hipLaunchKernelGGL(add_into, 20, 256, 0, s0, half0.yout, part, 2560 * ncol);       // the one allreduce
        }
        CK(hipSetDevice(0)); CK(hipStreamSynchronize(s0));
    };
    for (int w = 0; w < 3; ++w) { one(); two(); }
    double t1 = 0, t2 = 0;
    for (int r = 0; r < reps; ++r) {
        auto a = std::chrono::steady_clock::now(); one();
        auto b = std::chrono::steady_clock::now(); two();
        auto c = std::chrono::steady_clock::now();
        t1 += std::chrono::duration<double, std::milli>(b - a).count();
        t2 += std::chrono::duration<double, std::milli>(c - b).count();
    }
    t1 /= reps; t2 /= reps;
    std::printf("type %d, %d tokens, %d GDN layers: one card %.2f ms, two cards (half heads each + 2 handoffs/layer) %.2f ms -> x%.2f, %.1f us/layer saved\n",
                type, ncol, layers, t1, t2, t1 / t2, 1000.0 * (t1 - t2) / layers);
    std::fflush(stdout);
    return 0;
}
