# Split serving on gfx906 and across vendors: everything to rebuild and rerun

This branch is Strata 0.2.7 (`5e8a4cca`) with two gfx906 changes and this folder:

| Commit | What |
| --- | --- |
| #3986 | gfx906 builds again (`src/core/dflash2.cpp` takes the HIP stream path) |
| #3987 | opt-in two-stage decode top-k on gfx906, `STRATA_GFX906_TOPK_SPLIT=1`; off unless set |
| this folder | the launcher, configs, client and prompts of the split-serving measurements |

With it you can rebuild both engines and rerun every setup of the four benchmark issues: two MI50s as shipped,
an RTX 3070 reading the prompt for one or two MI50s, #3987 on and off, and a consumer card of 16, 12, 10 or 8 GB.

## Hardware it was measured on

| Role | Hardware |
| --- | --- |
| Decode | 2x AMD Instinct MI50 32 GB (gfx906), HBM2 at about 1 TB/s each, no MFMA |
| Prefill | NVIDIA RTX 3070, a modified 16 GB board; FP16/BF16 issue rate raised with Zagitta/gpu-unlock, clocks 1,500 to 2,100 MHz |
| Host | AMD Threadripper 1920X, 96 GB RAM, PCIe 3.0, SATA SSD |
| Software | Ubuntu 22.04 (kernel 6.8), ROCm 7.14 with amdgpu DKMS 6.10.5, CUDA 13.0 (driver 580) |
| Model | Qwen3.8-Flash-Next IQ3_S, 256K context |

Any NVIDIA card can take the prefill role and any gfx906 card the decode role; the speeds then follow those cards.

## 1. Build both engines

Two builds from this one checkout, each in its own container (Ubuntu 24.04 based; the binaries need glibc 2.38).

**gfx906** (decode role, and the MI50-only setups), as in `docs/AMD_HIP.md`, in the community image
`mixa3607/rocm-gfx906:7.14-complete` (current ROCm releases no longer ship gfx906 libraries):

```sh
docker run --rm -it -v "$PWD":/src -w /src mixa3607/rocm-gfx906:7.14-complete bash
apt-get update && apt-get install -y cmake ninja-build git python3 build-essential
cmake -S . -B build-906 -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_HIP_GFX906=ON -DCMAKE_HIP_ARCHITECTURES=gfx906 \
  -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++
ninja -C build-906 strata
```

**CUDA, sm_86** (prefill role on the RTX 3070), in NVIDIA's CUDA 13.0 development image:

```sh
docker run --rm -it -v "$PWD":/src -w /src nvidia/cuda:13.0.0-devel-ubuntu24.04 bash
apt-get update && apt-get install -y cmake ninja-build git python3 build-essential
cmake -S . -B build-cuda86 -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86
ninja -C build-cuda86 strata
```

Use your card's compute capability in place of 86. The result is `build-906/strata` and `build-cuda86/strata`, the
two paths the configs and `split-launch.sh` use.

## 2. Get the model

Install Qwen3.8-Flash-Next IQ3_S with Strata's own setup (`./setup.sh`, then choose that model). It fills a data
folder with `gguf/IQ3_S/` (the two GGUF shards), `packs/iq3_s/` (tokenizer and expert pack) and `mtp/rt/` (the
draft layer). The measurements used the shards of Hugging Face `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` at
revision `ed59f92082b1e93c0e96d60a8b11aab089b52f09`.

## 3. Write the configs

```sh
python3 bench/split-gfx906/make_configs.py --data /path/to/strata-data
```

It writes `bench/split-gfx906/configs/*.json` and reports any file it cannot find. The expert arena is
`/dev/shm/strata-iq3s.arena` (`--arena` to change it); every engine maps this one copy, so `/dev/shm` needs about
50 GB free.

| Config | Setup | Extra environment for `serve/server.py` |
| --- | --- | --- |
| `two-mi50-split` | MI50 #0 prefill, MI50 #1 decode, as shipped | `STRATA_PIN_RESERVE_MIB=12288`, the gfx906 switches below |
| `two-mi50-one` | 2x MI50, one engine, all experts in VRAM | `HIP_VISIBLE_DEVICES=0,1`, the gfx906 switches |
| `mi50-one` | one MI50, one engine | `HIP_VISIBLE_DEVICES=1`, the gfx906 switches |
| `rtx3070-one` | RTX 3070, one engine | none |
| `rtx3070-mi50-split` | RTX 3070 prefill, one MI50 decode | `STRATA_SPLIT_HIP_DEVICES=1` |
| `rtx3070-2mi50-split` | RTX 3070 prefill, two MI50s decode | none (`STRATA_SPLIT_HIP_DEVICES` defaults to `0,1`) |
| `kv32k-*` | the same with `--kv-resident 32768`, for the 16, 12 and 10 GB rows | as above |
| `kv32k-8gb-*` | 8 GB: `--max-context 131072 --kv q4_0 --kv-resident 32768`, reserve 800 MiB | as above |

The gfx906 switches: `HSA_OVERRIDE_GFX_VERSION=9.0.6 STRATA_GR_SPLIT=1 STRATA_GFX906_ATTN_QUERY_SWIZZLE=1
STRATA_GFX906_ATTN_REDUCE12=1`. `split-launch.sh` sets them itself for the decode role of the RTX 3070 splits.
Add `STRATA_GFX906_TOPK_SPLIT=1` to any MI50 engine for #3987.

## 4. Run a setup

`serve/server.py` starts the engine (or both engines of a split) and serves the OpenAI API; it needs the Python
packages of `requirements.txt`. A split with the RTX 3070 needs both runtimes in one place. The measurements ran
the server in a ROCm 7 image on Ubuntu 24.04 with the NVIDIA card passed in and the CUDA 13 runtime libraries
(`libcudart.so.13`, `libcublas.so.13`, `libcublasLt.so.13`) mounted; with the community image it looks like this
(this exact command was not part of the measurements):

```sh
docker run -d --name split --network host --ipc host -v /dev/shm:/dev/shm \
  --gpus device=0 --device /dev/kfd --device /dev/dri --group-add video \
  -e STRATA_SPLIT_CUDA_LIBS=/cudalib -e STRATA_SPLIT_HIP_DEVICES=1 \
  -v /path/to/cuda13-libs:/cudalib:ro -v /path/to/strata-data:/path/to/strata-data -v "$PWD":"$PWD" -w "$PWD" \
  mixa3607/rocm-gfx906:7.14-complete \
  bash -lc "pip install -r requirements.txt && python3 serve/server.py --engine strata \
            --config bench/split-gfx906/configs/rtx3070-mi50-split.json --host 127.0.0.1 --port 8080"
```

The MI50-only configs need only the ROCm image, and `rtx3070-one` only the CUDA image. Wait until
`curl -s localhost:8080/v1/models` reports `"loaded"`.

## 5. Measure

```sh
PORT=8080 python3 bench/split-gfx906/depth_one.py 16384 32768 65536 131072 196608 258048 > result.tsv
docker logs split 2>&1 | grep -a 'split:'      # a split's prefill read and hand-over, one line per request
```

`prompts/ctx-N.txt` are the prompts of #4000 and of the issues: N tokens of Strata's docs and sources. Run one
request at a time, with nothing else on the cards. For a split, the prefill speed is the prompt tokens over the
read time on its `split:` line; decode comes from the client's TSV.

**Smaller consumer cards.** Start `vram_hold.py` on the NVIDIA card before the engine, with
`16384 - G * 1024 + 115` for a G GB card (16 GB board), and stop it after the engine exits:

```sh
docker run -d --name vhold --gpus device=0 -v "$PWD":"$PWD" nvidia/cuda:13.0.0-devel-ubuntu24.04 \
  python3 "$PWD/bench/split-gfx906/vram_hold.py" 8307      # 8 GB: the engine sees 7,665 MiB free
```

## What each setup needs, and what fails without it

- **Without #3986** the gfx906 build stops in `src/core/dflash2.cpp` (`cudaDeviceGetStreamPriorityRange`,
  `cudaStreamCreateWithPriority` undeclared).
- **Two MI50s as shipped** start only with `STRATA_PIN_RESERVE_MIB=12288`: both HIP engines map the shared arena,
  and the amdgpu driver charges it to each (#4514, problem 1). The RTX 3070 splits have one HIP engine and run without it.
- **An NVIDIA card with MI50s** needs `split-launch.sh` as the config's `exe`: stock `split_serve` starts both roles
  with one executable. `serve/split.py` itself is unchanged.
- **A split needs the same KV arguments on both engines** (`--kv`, `--kv-resident`, `--max-context`); otherwise the
  first request fails with `config fingerprint differs`. The configs above set them for both.
- **8 GB** starts only with the `kv32k-8gb-*` arguments; `--vram-reserve-mib 300`, which the engine suggests on a
  small card, makes the first request fail with `verify: instantiate: out of memory`.
