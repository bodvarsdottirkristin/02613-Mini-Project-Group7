"""
Task 7: Jacobi solver with Numba JIT (CPU).
"""
from __future__ import annotations

import sys
import time
from os.path import join

import numpy as np
from numba import njit

from simulate_original import load_data, summary_stats


@njit(cache=True)
def jacobi_numba(u, interior_mask, max_iter, atol):
    """
    Same physics as simulate_original.jacobi: update only interior_mask cells each iteration;
    wall cells keep their previous values. Row-major i-j loops match C-contiguous layout.
    """
    u = np.copy(u)
    u_new = np.empty_like(u)
    n = interior_mask.shape[0]

    for _ in range(max_iter):
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                if interior_mask[i - 1, j - 1]:
                    u_new[i, j] = 0.25 * (
                        u[i, j - 1] + u[i, j + 1] + u[i - 1, j] + u[i + 1, j]
                    )
                else:
                    u_new[i, j] = u[i, j]

        delta = 0.0
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                if interior_mask[i - 1, j - 1]:
                    d = abs(u_new[i, j] - u[i, j])
                    if d > delta:
                        delta = d

        u[:] = u_new
        if delta < atol:
            break

    return u


def _warmup_jit() -> None:
    """Compile once so timing excludes JIT compilation."""
    u = np.zeros((4, 4), dtype=np.float64)
    m = np.zeros((2, 2), dtype=np.bool_)
    m[:] = True
    u[1, 1] = u[1, 2] = u[2, 1] = u[2, 2] = 1.0
    jacobi_numba(u, m, 2, 1e-6)


if __name__ == '__main__':
    LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'

    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        building_ids = f.read().splitlines()

    if len(sys.argv) < 2:
        N = 1
    else:
        N = int(sys.argv[1])
    building_ids = building_ids[:N]

    all_u0 = np.empty((N, 514, 514))
    all_interior_mask = np.empty((N, 512, 512), dtype=np.bool_)
    for i, bid in enumerate(building_ids):
        u0, interior_mask = load_data(LOAD_DIR, bid)
        all_u0[i] = u0
        all_interior_mask[i] = np.ascontiguousarray(interior_mask)

    MAX_ITER = 20_000
    ABS_TOL = 1e-4

    _warmup_jit()

    t0 = time.perf_counter()
    all_u = np.empty_like(all_u0)
    for i, (u0, interior_mask) in enumerate(zip(all_u0, all_interior_mask)):
        all_u[i] = jacobi_numba(u0, interior_mask, MAX_ITER, ABS_TOL)
    elapsed = time.perf_counter() - t0

    stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
    print('building_id, ' + ', '.join(stat_keys))
    for bid, u, interior_mask in zip(building_ids, all_u, all_interior_mask):
        stats = summary_stats(u, interior_mask)
        print(f"{bid},", ", ".join(str(stats[k]) for k in stat_keys))

    print(f'# wall_time_seconds, {elapsed:.6f}', file=sys.stderr)
