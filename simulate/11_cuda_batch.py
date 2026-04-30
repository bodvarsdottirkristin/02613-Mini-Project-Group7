"""
Task 9: Jacobi via a batched CUDA kernel.

Improvement over 08_cuda_kernel.py: instead of processing one building at a time,
we use a 3D kernel grid where blockIdx.z indexes the building within a batch.
This lets the GPU execute Jacobi iterations for BATCH_SIZE buildings simultaneously,
keeping the device much more occupied.
"""
from __future__ import annotations

import sys
import time
from os.path import join

import numpy as np
import pandas as pd
from numba import cuda

from simulate_original import load_data, summary_stats

SIZE = 512
TX, TY = 16, 16
BATCH_SIZE = 64  # buildings processed per GPU batch; tune based on GPU VRAM


@cuda.jit
def jacobi_step_kernel_batch(u_in, u_out, interior_mask, n):
    """
    Single Jacobi iteration over a batch of buildings.
    u_in/u_out shape: (B, n+2, n+2)
    interior_mask shape: (B, n, n)
    blockIdx.z selects the building within the batch.
    """
    b = cuda.blockIdx.z
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    if i >= n or j >= n:
        return
    gi = i + 1
    gj = j + 1
    if interior_mask[b, i, j]:
        u_out[b, gi, gj] = 0.25 * (
            u_in[b, gi, gj - 1]
            + u_in[b, gi, gj + 1]
            + u_in[b, gi - 1, gj]
            + u_in[b, gi + 1, gj]
        )
    else:
        u_out[b, gi, gj] = u_in[b, gi, gj]


def jacobi_cuda_batch(u_batch, mask_batch, max_iter):
    """
    Run Jacobi for a batch of buildings entirely on the GPU.
    u_batch:    (B, SIZE+2, SIZE+2) float64
    mask_batch: (B, SIZE, SIZE)     bool
    Returns u_batch after max_iter iterations (shape unchanged).
    """
    B = u_batch.shape[0]
    u_batch = np.ascontiguousarray(u_batch, dtype=np.float64)
    mask_batch = np.ascontiguousarray(mask_batch, dtype=np.bool_)

    d_src = cuda.to_device(u_batch)
    d_dst = cuda.to_device(u_batch)
    d_mask = cuda.to_device(mask_batch)

    bx = (SIZE + TX - 1) // TX
    by = (SIZE + TY - 1) // TY
    blocks = (bx, by, B)
    threads = (TX, TY, 1)

    for _ in range(max_iter):
        jacobi_step_kernel_batch[blocks, threads](d_src, d_dst, d_mask, SIZE)
        d_src, d_dst = d_dst, d_src

    cuda.synchronize()
    return d_src.copy_to_host()


if __name__ == '__main__':
    LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'

    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        building_ids = f.read().splitlines()

    if len(sys.argv) < 2:
        n_buildings = len(building_ids)
    else:
        n_buildings = int(sys.argv[1])
    building_ids = building_ids[:n_buildings]

    print(f"Loading {n_buildings} buildings ...", file=sys.stderr)
    all_u0 = np.empty((n_buildings, SIZE + 2, SIZE + 2), dtype=np.float64)
    all_mask = np.empty((n_buildings, SIZE, SIZE), dtype=np.bool_)
    for i, bid in enumerate(building_ids):
        all_u0[i], all_mask[i] = load_data(LOAD_DIR, bid)

    max_iter = 20_000
    stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']

    rows = []
    t0 = time.perf_counter()

    for start in range(0, n_buildings, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n_buildings)
        batch_ids = building_ids[start:end]
        u_result = jacobi_cuda_batch(all_u0[start:end], all_mask[start:end], max_iter)
        for k, (bid, u, m) in enumerate(zip(batch_ids, u_result, all_mask[start:end])):
            stats = summary_stats(u, m)
            rows.append({'building_id': bid, **{k: stats[k] for k in stat_keys}})
        print(f"  processed {end}/{n_buildings}", file=sys.stderr)

    elapsed = time.perf_counter() - t0
    print(f'# wall_time_seconds, {elapsed:.6f}', file=sys.stderr)

    df = pd.DataFrame(rows, columns=['building_id'] + stat_keys)

    # Also print CSV to stdout for compatibility with the original scripts
    print('building_id, ' + ', '.join(stat_keys))
    for _, row in df.iterrows():
        print(f"{row['building_id']},", ", ".join(str(row[k]) for k in stat_keys))
