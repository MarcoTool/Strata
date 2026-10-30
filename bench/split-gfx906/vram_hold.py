#!/usr/bin/env python3
"""Hold VRAM on a CUDA card so the engine sees a smaller card; sleeps until killed.

Usage: CUDA_VISIBLE_DEVICES=0 python3 vram_hold.py TARGET_USED_MIB

Allocates until the card's used memory (this process's context included) reaches
TARGET_USED_MIB.  The consumer-card measurements made a 16 GB RTX 3070 look like a
G GB card with TARGET_USED_MIB = 16384 - G * 1024 + 115, so the engine saw 11,761 (12 GB),
9,713 (10 GB) or 7,665 MiB (8 GB) free.  Compute and memory bandwidth stay the card's own.
Start it before the engine, and stop it after the engine has exited.
"""
import ctypes
import glob
import os
import sys
import time

libdir = os.environ.get('CUDA_LIB_DIR', '/usr/local/cuda/lib64')
lib = ctypes.CDLL(sorted(glob.glob(os.path.join(libdir, 'libcudart.so*')))[-1])
free, total = ctypes.c_size_t(), ctypes.c_size_t()
lib.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total))   # creates the context
need = (int(sys.argv[1]) << 20) - (total.value - free.value)
ptrs = []
while need > 0:
    step = min(need, 256 << 20)
    p = ctypes.c_void_p()
    if lib.cudaMalloc(ctypes.byref(p), ctypes.c_size_t(step)) != 0:
        print('cudaMalloc failed', flush=True)
        break
    lib.cudaMemset(p, 0, ctypes.c_size_t(step))   # touch it, so the pages are really taken
    ptrs.append(p)
    need -= step
lib.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total))
print(f'held: used {(total.value - free.value) >> 20} MiB, free {free.value >> 20} MiB of {total.value >> 20} MiB', flush=True)
while True:
    time.sleep(3600)
