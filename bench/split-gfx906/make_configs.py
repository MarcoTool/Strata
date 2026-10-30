#!/usr/bin/env python3
"""Write the model configs of the split-serving measurements on gfx906.

Usage: python3 make_configs.py --data DIR [--out DIR] [--arena PATH]

DIR is the Strata data folder that setup filled for Qwen3.8-Flash-Next IQ3_S (it holds
gguf/IQ3_S, packs/iq3_s and mtp/rt).  The configs go to --out (default: ./configs next to
this file) and are started with serve/server.py --config.  Every path is checked; a missing
one is reported, not guessed.

One config per setup in the issues (see README.md for the table):
  two-mi50-split           MI50 #0 prefill, MI50 #1 decode (as shipped)
  two-mi50-one             2x MI50, one engine, all experts in VRAM
  mi50-one                 one MI50, one engine
  rtx3070-one              RTX 3070, one engine
  rtx3070-mi50-split       RTX 3070 prefill, one MI50 decode
  rtx3070-2mi50-split      RTX 3070 prefill, two MI50s decode
  kv32k-*                  the same with --kv-resident 32768 (consumer-card issue, 16/12/10 GB)
  kv32k-8gb-*              8 GB: --max-context 131072 --kv q4_0 --kv-resident 32768, reserve 800 MiB
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))


def base_args(data, arena, ctx=262144, kv='int8', resident=262144):
    """The engine arguments every engine of the measurements shares."""
    return ['--pack', f'{data}/packs/iq3_s',
            '--native', f'{data}/gguf/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf',
            '--ple-gguf', f'{data}/gguf/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf',
            '--expert-profile', f'{REPO}/data/expert-profile.bin',
            '--expert-cache', 'auto', '--prefill', 'auto',
            '--spec', '4', '--spec-min-p', '0.5', '--mtp', f'{data}/mtp/rt',
            '--max-context', str(ctx), '--kv', kv, '--kv-resident', str(resident),
            '--shared-expert-arena', arena]


def config(data, out, name, args, **extra):
    """One server config: the common keys, then the setup's own."""
    c = {'cwd': REPO, 'tokenizer': f'{data}/packs/iq3_s/tokenizer', 'model_name': 'qwen3.8-flash-next-iq3_s',
         'sampling': {'temperature': 1.0, 'top_p': 0.95, 'top_k': 20, 'min_p': 0.0},
         'log': os.path.join(out, name + '.log'), 'engine_ready_s': 3600, 'args': args}
    c.update(extra)
    return c


# prefill-card arguments of the measurements (the RTX 3070 engine, alone or in a split)
PREFILL = ['--vram-reserve-mib', '2000', '--prompt-cache', '48', '--prompt-cache-every', '8192']
CUDA = dict(exe=f'{REPO}/build-cuda86/strata', lib_dirs=['/usr/local/cuda/bin', '/usr/local/cuda/lib64'], gpu=0, gpus_asked=True)
HIP = dict(exe=f'{REPO}/build-906/strata', lib_dirs=['/opt/rocm/lib'], backend='hip')
LAUNCH = dict(exe=os.path.join(HERE, 'split-launch.sh'), lib_dirs=['/opt/rocm/lib'], backend='hip')


def split(prefill_reserve='2000', two_mi50=False):
    """split_serve block for the RTX 3070 + MI50 setups; the card numbers only have to differ,
    split-launch.sh decides which cards each build sees."""
    p = list(PREFILL)
    p[1] = prefill_reserve
    d = (['--expert-cache-device1', 'auto'] if two_mi50 else []) + ['--remote-expert-opt']
    return {'prefill_gpu': 1, 'decode_gpu': 0, 'prefill_args': p + ['--split-role-prefill'], 'decode_args': d}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--data', required=True, help='Strata data folder with gguf/IQ3_S, packs/iq3_s, mtp/rt')
    ap.add_argument('--out', default=os.path.join(HERE, 'configs'))
    ap.add_argument('--arena', default='/dev/shm/strata-iq3s.arena', help='shared expert arena file (tmpfs)')
    a = ap.parse_args()
    data, out = os.path.abspath(a.data), os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)

    full = base_args(data, a.arena)
    k32 = base_args(data, a.arena, resident=32768)
    k8 = base_args(data, a.arena, ctx=131072, kv='q4_0', resident=32768)
    cfgs = {
        # as shipped: SPLIT_SERVE.md's config on the two MI50s (needs STRATA_PIN_RESERVE_MIB=12288, see README)
        'two-mi50-split': (full + ['--remote-expert-opt'], dict(HIP, split_serve={'prefill_gpu': 0, 'decode_gpu': 1})),
        'two-mi50-one': (full + ['--expert-cache-device1', 'auto', '--remote-expert-opt'], HIP),
        'mi50-one': (full + ['--remote-expert-opt'], HIP),
        'rtx3070-one': (full + PREFILL, CUDA),
        'rtx3070-mi50-split': (full, dict(LAUNCH, split_serve=split())),
        'rtx3070-2mi50-split': (full, dict(LAUNCH, split_serve=split(two_mi50=True))),
        # consumer-card issue: KV streaming on every engine of the comparison (a split needs the same KV arguments on both)
        'kv32k-rtx3070-one': (k32 + PREFILL, CUDA),
        'kv32k-rtx3070-mi50-split': (k32, dict(LAUNCH, split_serve=split())),
        'kv32k-mi50-one': (k32 + ['--remote-expert-opt'], HIP),
        # 8 GB: a 128K context, a 4-bit KV cache and an 800 MiB reserve, on both engines
        'kv32k-8gb-rtx3070-one': (k8 + ['--vram-reserve-mib', '800'] + PREFILL[2:], CUDA),
        'kv32k-8gb-rtx3070-mi50-split': (k8, dict(LAUNCH, split_serve=split(prefill_reserve='800'))),
        'kv32k-8gb-mi50-one': (k8 + ['--remote-expert-opt'], HIP),
    }
    for name, (args, extra) in cfgs.items():
        with open(os.path.join(out, name + '.json'), 'w', encoding='utf-8', newline='\n') as f:
            json.dump(config(data, out, name, args, **extra), f, indent=1)
    print(f'wrote {len(cfgs)} configs to {out}')

    missing = [p for p in (f'{data}/packs/iq3_s/tokenizer', full[3], full[5], full[7], f'{data}/mtp/rt',
                           CUDA['exe'], HIP['exe']) if not os.path.exists(p)]
    for p in missing:
        print(f'missing: {p}', file=sys.stderr)
    return 1 if missing else 0


if __name__ == '__main__':
    sys.exit(main())
