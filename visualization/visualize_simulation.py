import sys
from os.path import join

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, join(sys.path[0], '..', 'simulate'))
from ..simulate.simulate_original import load_data, jacobi, summary_stats

LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'

building_ids = ['10000', '10334', '10786', '11117']

MAX_ITER = 20_000
ABS_TOL = 1e-4

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

for col, bid in enumerate(building_ids):
    u0, interior_mask = load_data(LOAD_DIR, bid)
    u = jacobi(u0, interior_mask, MAX_ITER, ABS_TOL)

    # Top row: simulated temperature
    im = axes[0, col].imshow(u[1:-1, 1:-1], cmap='inferno', vmin=0, vmax=25)
    stats = summary_stats(u, interior_mask)
    axes[0, col].set_title(f'ID: {bid}\nmean={stats["mean_temp"]:.1f}°C')
    axes[0, col].axis('on')

    # Bottom row: interior mask
    axes[1, col].imshow(interior_mask, cmap='gray')
    axes[1, col].set_title(f'ID: {bid}')
    axes[1, col].axis('on')

cbar_ax = fig.add_axes([0.92, 0.55, 0.015, 0.35])
fig.colorbar(im, cax=cbar_ax, label='Temperature (°C)')

axes[0, 0].set_ylabel('Simulated temperature')
axes[1, 0].set_ylabel('Interior mask')

fig.suptitle('Simulation results: steady-state temperature (top) and interior masks (bottom)', fontsize=13)
plt.tight_layout(rect=[0, 0, 0.91, 1])
plt.savefig('simulation_results.png', dpi=100, bbox_inches='tight')
plt.show()
