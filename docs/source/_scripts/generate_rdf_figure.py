#!/usr/bin/env python3
"""Generate RDF comparison figure for the tutorials.

Compares histogram-based vs force-based g(r) using LJ fluid data
with fine bins and limited frames to show the force-based advantage.
"""

import matplotlib.pyplot as plt
from matplotlib import rcParams
from pathlib import Path

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
    from revelsMD.trajectories import LammpsTrajectory
    from revelsMD.rdf import compute_rdf

    lj_path = EXAMPLES / "example_1_LJ"
    traj = LammpsTrajectory(
        str(lj_path / "dump.nh.lammps"),
        str(lj_path / "data.fin.nh.data"),
        temperature=1.35,
        units='lj',
        atom_style="id resid type q x y z ix iy iz",
    )

    # Fine bins + few frames: force-based advantage is striking
    rdf = compute_rdf(traj, '1', '1', integration='lambda', delr=0.005, stop=1)

    fig, ax = plt.subplots(figsize=(3.15, 2.1))
    ax.plot(rdf.r, rdf.g_count, '-', color='#F9844A', label='Histogram')
    ax.plot(rdf.r, rdf.g_force, '-', color='#277DA1', label='Force-based')
    ax.axhline(1, color='gray', linestyle='--', lw=linewidth * 0.8)
    ax.set_xlabel(r'$r$ (LJ units)')
    ax.set_ylabel(r'$g(r)$')
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 3.5)
    ax.legend()
    plt.tight_layout()

    output_path = OUTPUT_DIR / "tutorial_rdf.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
