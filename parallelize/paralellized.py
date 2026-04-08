from os.path import join
import sys

import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
import time


def load_data(load_dir, bid):
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask


def jacobi(u, interior_mask, max_iter, atol=1e-6):
    u = np.copy(u)

    for i in range(max_iter):
        # Compute average of left, right, up and down neighbors, see eq. (1)
        u_new = 0.25 * (u[1:-1, :-2] + u[1:-1, 2:] + u[:-2, 1:-1] + u[2:, 1:-1])
        u_new_interior = u_new[interior_mask]
        delta = np.abs(u[1:-1, 1:-1][interior_mask] - u_new_interior).max()
        u[1:-1, 1:-1][interior_mask] = u_new_interior

        if delta < atol:
            break
    return u


def summary_stats(u, interior_mask):
    u_interior = u[1:-1, 1:-1][interior_mask]
    mean_temp = u_interior.mean()
    std_temp = u_interior.std()
    pct_above_18 = np.sum(u_interior > 18) / u_interior.size * 100
    pct_below_15 = np.sum(u_interior < 15) / u_interior.size * 100
    return {
        'mean_temp': mean_temp,
        'std_temp': std_temp,
        'pct_above_18': pct_above_18,
        'pct_below_15': pct_below_15,
    }


if __name__ == '__main__':
    # Load data
    LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'
    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        building_ids = f.read().splitlines()

    if len(sys.argv) < 2:
        N = 1
    else:
        N = int(sys.argv[1])
    building_ids = building_ids[:N]

    num_procs_list = [1, 2, 3, 4, 5, 6, 7]  # number of workers — change this for timing experiments

    times = []
    for num_proc in num_procs_list:
      print(f"Running with {num_proc} processes...")
      start_time = time.time()
      chunk_size = max(1,N // num_proc)
      pool = multiprocessing.Pool(num_proc)

      # Load floor plans
      all_u0 = np.empty((N, 514, 514))
      all_interior_mask = np.empty((N, 512, 512), dtype='bool')
      for i, bid in enumerate(building_ids):
          u0, interior_mask = load_data(LOAD_DIR, bid)
          all_u0[i] = u0
          all_interior_mask[i] = interior_mask

      # Run jacobi iterations for each floor plan
      MAX_ITER = 20_000
      ABS_TOL = 1e-4

      args = [(all_u0[i], all_interior_mask[i], MAX_ITER, ABS_TOL) for i in range(N)]
      results = pool.starmap(jacobi, args, chunk_size)
      all_u = np.array(results)
      pool.close()
      pool.join()

      # Print summary statistics in CSV format
      stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
      print('building_id, ' + ', '.join(stat_keys))  # CSV header
      for bid, u, interior_mask in zip(building_ids, all_u, all_interior_mask):
          stats = summary_stats(u, interior_mask)
          print(f"{bid},", ", ".join(str(stats[k]) for k in stat_keys))

      elapsed = time.time() - start_time
      times.append(elapsed)
      print(f"Time: {elapsed:.2f}s")

     # Speedup plot
    serial_time = times[0] if 1 in num_procs_list else times[num_procs_list.index(min(num_procs_list))]
    speedups = [serial_time / t for t in times]
    
    plt.figure(figsize=(8, 6))
    plt.plot(num_procs_list, speedups, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Number of Processes')
    plt.ylabel('Speedup')
    plt.title('Mandelbrot Parallel Speedup')
    plt.grid(True, alpha=0.3)
    plt.savefig('speedup.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Speedup plot saved as 'speedup.png'")