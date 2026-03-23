#!/usr/bin/env python3
"""Generate density comparison figure for the tutorials.

Compares count, force, and lambda Li density in Li6PS5I.
"""

import os
os.environ['REVELSMD_FFT_WORKERS'] = '-1'

import matplotlib.pyplot as plt
from matplotlib import rcParams
from pathlib import Path
import cmcrameri.cm as cmc

REPO_ROOT = Path(__file__).parent.parent.parent.parent
EXAMPLES = REPO_ROOT / "examples"
OUTPUT_DIR = Path(__file__).parent.parent / "_static" / "images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

nearly_black = '#161616'
fontsize = 8
linewidth = 0.7

for k, v in {
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica Neue', 'Helvetica', 'DejaVu Sans'],
    'font.size': fontsize,
    'axes.formatter.limits': (-3, 3),
    'xtick.major.pad': 3,
    'ytick.major.pad': 3,
    'ytick.color': nearly_black,
    'xtick.color': nearly_black,
    'xtick.labelsize': fontsize,
    'ytick.labelsize': fontsize,
    'ytick.major.size': 3,
    'xtick.major.size': 3,
    'xtick.major.width': linewidth,
    'ytick.major.width': linewidth,
    'axes.labelcolor': nearly_black,
    'legend.frameon': False,
    'axes.labelpad': 2,
    'axes.labelsize': fontsize,
    'axes.linewidth': linewidth,
    'lines.linewidth': linewidth,
}.items():
    rcParams[k] = v


def main():
    from revelsMD.trajectories import VaspTrajectory
    from revelsMD.density import compute_density

    vasp_path = EXAMPLES / "Li6PS5I_run1_vasprun.xml"
    if not vasp_path.exists():
        print(f"Data not found: {vasp_path}")
        return

    print(f"Loading trajectory from {vasp_path}...")
    traj = VaspTrajectory(str(vasp_path), temperature=500.0)
    print(f"Trajectory: {traj.frames} frames")

    print("Computing density (200 bins, lambda)...")
    grid = compute_density(
        traj,
        'Li',
        density_type='number',
        nbins=200,
        compute_lambda=True,
        block_size=750,
    )

    # z = 0.375 slice captures Li cage structure
    z_frac = 0.375
    z_idx = int(z_frac * grid.nbinsz)

    count_slice = grid.rho_count[:, :, z_idx]
    force_slice = grid.rho_force[:, :, z_idx]
    lambda_slice = grid.rho_lambda[:, :, z_idx]

    vmin = min(count_slice.min(), force_slice.min(), lambda_slice.min())
    vmax = max(count_slice.max(), force_slice.max(), lambda_slice.max())

    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.1))

    for ax, data, title in zip(
        axes,
        [count_slice, force_slice, lambda_slice],
        ['Count', 'Force', 'Lambda'],
    ):
        ax.imshow(
            data.T,
            origin='lower',
            cmap=cmc.lipari,
            aspect='equal',
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    output_path = OUTPUT_DIR / "tutorial_density.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
