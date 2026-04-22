"""
Task 8: Jacobi via a custom CUDA kernel (Numba).

- One Jacobi sweep per kernel launch (threads sync between iterations at host level).
- Fixed iteration count (no atol / early exit).
- Helper: jacobi_cuda(u, interior_mask, max_iter) — same role as reference jacobi minus atol.

Requires: GPU, Numba with CUDA, numpy. Run on a CUDA node (e.g. course GPU queue).
"""
from __future__ import annotations

import sys
import time
from os.path import join

import numpy as np
from numba import cuda

from simulate_original import load_data, summary_stats

SIZE = 512
THREADS = (16, 16)


@cuda.jit
def jacobi_step_kernel(u_in, u_out, interior_mask, n):
    """Single Jacobi iteration: update only interior_mask True cells; walls copy through."""
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    if i >= n or j >= n:
        return
    gi = i + 1
    gj = j + 1
    if interior_mask[i, j]:
        u_out[gi, gj] = 0.25 * (
            u_in[gi, gj - 1]
            + u_in[gi, gj + 1]
            + u_in[gi - 1, gj]
            + u_in[gi + 1, gj]
        )
    else:
        u_out[gi, gj] = u_in[gi, gj]


def jacobi_cuda(u, interior_mask, max_iter):
    """
    Same inputs as reference ``jacobi`` except no ``atol``: run exactly ``max_iter`` steps.

    Double-buffer on device; host call loop launches one kernel per iteration so each
    iteration is separated by a global sync.
    """
    u = np.ascontiguousarray(np.copy(u), dtype=np.float64)
    interior_mask = np.ascontiguousarray(interior_mask, dtype=np.bool_)

    d_src = cuda.to_device(u)
    d_dst = cuda.to_device(u)
    d_mask = cuda.to_device(interior_mask)

    blocks = (
        (SIZE + THREADS[0] - 1) // THREADS[0],
        (SIZE + THREADS[1] - 1) // THREADS[1],
    )

    for _ in range(max_iter):
        jacobi_step_kernel[blocks, THREADS](d_src, d_dst, d_mask, SIZE)
        d_src, d_dst = d_dst, d_src

    cuda.synchronize()

    out = d_src.copy_to_host()
    return out


if __name__ == '__main__':
    LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'

    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        building_ids = f.read().splitlines()

    if len(sys.argv) < 2:
        n_buildings = 1
    else:
        n_buildings = int(sys.argv[1])
    building_ids = building_ids[:n_buildings]

    all_u0 = np.empty((n_buildings, SIZE + 2, SIZE + 2))
    all_mask = np.empty((n_buildings, SIZE, SIZE), dtype=np.bool_)
    for i, bid in enumerate(building_ids):
        u0, m = load_data(LOAD_DIR, bid)
        all_u0[i] = u0
        all_mask[i] = m

    max_iter = 20_000

    t0 = time.perf_counter()
    all_u = np.empty_like(all_u0)
    for i in range(n_buildings):
        all_u[i] = jacobi_cuda(all_u0[i], all_mask[i], max_iter)
    elapsed = time.perf_counter() - t0
    print(f'# wall_time_seconds, {elapsed:.6f}', file=sys.stderr)

    stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
    print('building_id, ' + ', '.join(stat_keys))
    for bid, u, m in zip(building_ids, all_u, all_mask):
        stats = summary_stats(u, m)
        print(f"{bid},", ", ".join(str(stats[k]) for k in stat_keys))
