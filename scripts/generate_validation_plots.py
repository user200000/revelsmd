#!/usr/bin/env python3
"""
Generate validation plots for visual inspection of integration test results.

This script produces plots that allow direct visual confirmation that the
calculated RDFs and densities match expected physical behaviour.

Output is written to tests/validation_plots/
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parents[1]))

OUTPUT_DIR = Path(__file__).parents[1] / "tests" / "validation_plots"
EXAMPLES_DIR = Path(__file__).parents[1] / "examples"


def ensure_output_dir():
    """Create output directory if it doesn't exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")


def plot_uniform_gas_rdf():
    """
    Plot RDF for uniform random gas.

    Expected: g(r) ~ 1 for all r (with statistical noise)
    """
    from revelsMD.trajectories import NumpyTrajectory
    from revelsMD.rdf import run_rdf, run_rdf_lambda

    print("\n=== Uniform Gas RDF ===")

    np.random.seed(42)
    n_atoms = 500
    n_frames = 10
    box = 10.0

    positions = np.random.uniform(0, box, (n_frames, n_atoms, 3))
    forces = np.random.randn(n_frames, n_atoms, 3) * 0.1
    species = ['1'] * n_atoms

    ts = NumpyTrajectory(positions, forces, box, box, box, species, units='lj', temperature=1.0)

    # Forward and backward integration
    rdf_forward = run_rdf(ts, '1', '1', delr=0.1, from_zero=True)
    rdf_backward = run_rdf(ts, '1', '1', delr=0.1, from_zero=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(rdf_forward[0], rdf_forward[1], 'b-', label='Forward (from_zero=True)')
    ax1.axhline(y=1.0, color='k', linestyle='--', label='Expected (g=1)')
    ax1.set_xlabel('r')
    ax1.set_ylabel('g(r)')
    ax1.set_title('Forward Integration')
    ax1.legend()
    ax1.set_ylim(-0.5, 2.5)
    ax1.grid(True, alpha=0.3)

    ax2.plot(rdf_backward[0], rdf_backward[1], 'r-', label='Backward (from_zero=False)')
    ax2.axhline(y=1.0, color='k', linestyle='--', label='Expected (g=1)')
    ax2.set_xlabel('r')
    ax2.set_ylabel('g(r)')
    ax2.set_title('Backward Integration')
    ax2.legend()
    ax2.set_ylim(-0.5, 2.5)
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Uniform Random Gas: g(r) should approach 1')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'uniform_gas_rdf.png', dpi=150)
    plt.close()
    print(f"  Saved: uniform_gas_rdf.png")
    print(f"  Backward mean g(r) for r>2: {np.mean(rdf_backward[1][rdf_backward[0] > 2]):.3f} (expected ~1.0)")


def plot_two_atom_rdf():
    """
    Plot RDF for two atoms at fixed separation.

    Expected: Sharp peak at r = 3.0 (the separation distance)
    """
    from revelsMD.trajectories import NumpyTrajectory
    from revelsMD.rdf import run_rdf, run_rdf_lambda

    print("\n=== Two Atom RDF ===")

    n_frames = 5
    box = 10.0
    separation = 3.0

    positions = np.zeros((n_frames, 2, 3))
    positions[:, 0, :] = [box/2, box/2, box/2]
    positions[:, 1, :] = [box/2 + separation, box/2, box/2]

    np.random.seed(43)
    forces = np.random.randn(n_frames, 2, 3) * 0.1
    species = ['1', '1']

    ts = NumpyTrajectory(positions, forces, box, box, box, species, units='lj', temperature=1.0)

    rdf = run_rdf(ts, '1', '1', delr=0.1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rdf[0], rdf[1], 'b-', linewidth=2)
    ax.axvline(x=separation, color='r', linestyle='--', label=f'Expected peak at r={separation}')
    ax.set_xlabel('r')
    ax.set_ylabel('g(r)')
    ax.set_title(f'Two Atoms at Separation d={separation}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    peak_idx = np.argmax(rdf[1])
    peak_r = rdf[0, peak_idx]
    ax.annotate(f'Peak at r={peak_r:.2f}', xy=(peak_r, rdf[1, peak_idx]),
                xytext=(peak_r + 0.5, rdf[1, peak_idx] * 0.8),
                arrowprops=dict(arrowstyle='->', color='green'))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'two_atom_rdf.png', dpi=150)
    plt.close()
    print(f"  Saved: two_atom_rdf.png")
    print(f"  Peak location: r={peak_r:.2f} (expected {separation})")


def plot_cubic_lattice_rdf():
    """
    Plot RDF for simple cubic lattice.

    Expected: Peaks at lattice spacings (2.5, 3.54, 4.33, ...)
    """
    from revelsMD.trajectories import NumpyTrajectory
    from revelsMD.rdf import run_rdf, run_rdf_lambda

    print("\n=== Cubic Lattice RDF ===")

    n_frames = 5
    box = 10.0
    spacing = 2.5
    n_per_dim = 4

    lattice_positions = []
    for i in range(n_per_dim):
        for j in range(n_per_dim):
            for k in range(n_per_dim):
                lattice_positions.append([i * spacing, j * spacing, k * spacing])

    lattice_positions = np.array(lattice_positions)
    n_atoms = len(lattice_positions)

    positions = np.zeros((n_frames, n_atoms, 3))
    for frame in range(n_frames):
        positions[frame] = lattice_positions

    np.random.seed(45)
    forces = np.random.randn(n_frames, n_atoms, 3) * 0.1
    species = ['1'] * n_atoms

    ts = NumpyTrajectory(positions, forces, box, box, box, species, units='lj', temperature=1.0)

    rdf = run_rdf(ts, '1', '1', delr=0.1, from_zero=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rdf[0], rdf[1], 'b-', linewidth=2)

    # Mark expected peak positions
    expected_peaks = [
        spacing,                    # nearest neighbour
        spacing * np.sqrt(2),       # face diagonal
        spacing * np.sqrt(3),       # body diagonal
        spacing * 2,                # second shell
    ]
    colors = ['r', 'orange', 'green', 'purple']
    labels = ['a', 'a*sqrt(2)', 'a*sqrt(3)', '2a']

    for peak, color, label in zip(expected_peaks, colors, labels):
        ax.axvline(x=peak, color=color, linestyle='--', alpha=0.7, label=f'{label}={peak:.2f}')

    ax.set_xlabel('r')
    ax.set_ylabel('g(r)')
    ax.set_title(f'Simple Cubic Lattice (spacing={spacing})')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 6)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'cubic_lattice_rdf.png', dpi=150)
    plt.close()
    print(f"  Saved: cubic_lattice_rdf.png")
    print(f"  Expected peaks at: {[f'{p:.2f}' for p in expected_peaks]}")


def plot_single_atom_density():
    """
    Plot 3D density for single atom at known position.

    Expected: Peak at atom location (5, 5, 5)
    """
    from revelsMD.trajectories import NumpyTrajectory
    from revelsMD.revels_3D import Revels3D

    print("\n=== Single Atom Density ===")

    n_frames = 5
    box = 10.0

    positions = np.zeros((n_frames, 1, 3))
    positions[:, 0, :] = [5.0, 5.0, 5.0]

    np.random.seed(44)
    forces = np.random.randn(n_frames, 1, 3) * 0.1
    species = ['1']

    ts = NumpyTrajectory(positions, forces, box, box, box, species, units='lj', temperature=1.0)

    nbins = 20
    gs = Revels3D.GridState(ts, 'number', nbins=nbins, temperature=1.0)
    gs.accumulate(ts, '1', kernel='triangular', rigid=False)
    gs.get_real_density()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # XY slice at z=10 (centre)
    ax = axes[0]
    im = ax.imshow(gs.rho[:, :, nbins//2].T, origin='lower', extent=[0, box, 0, box])
    ax.plot(5, 5, 'rx', markersize=15, markeredgewidth=3, label='Atom position')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('XY slice at z=5')
    ax.legend()
    plt.colorbar(im, ax=ax, label='density')

    # XZ slice at y=10
    ax = axes[1]
    im = ax.imshow(gs.rho[:, nbins//2, :].T, origin='lower', extent=[0, box, 0, box])
    ax.plot(5, 5, 'rx', markersize=15, markeredgewidth=3, label='Atom position')
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_title('XZ slice at y=5')
    ax.legend()
    plt.colorbar(im, ax=ax, label='density')

    # YZ slice at x=10
    ax = axes[2]
    im = ax.imshow(gs.rho[nbins//2, :, :].T, origin='lower', extent=[0, box, 0, box])
    ax.plot(5, 5, 'rx', markersize=15, markeredgewidth=3, label='Atom position')
    ax.set_xlabel('y')
    ax.set_ylabel('z')
    ax.set_title('YZ slice at x=5')
    ax.legend()
    plt.colorbar(im, ax=ax, label='density')

    fig.suptitle('Single Atom at (5, 5, 5): Density should peak at atom location')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'single_atom_density.png', dpi=150)
    plt.close()
    print(f"  Saved: single_atom_density.png")

    max_idx = np.unravel_index(np.argmax(gs.rho), gs.rho.shape)
    max_pos = [(idx + 0.5) * box / nbins for idx in max_idx]
    print(f"  Maximum density at bin {max_idx} = position ~{[f'{p:.1f}' for p in max_pos]} (expected 5,5,5)")


def plot_uniform_density():
    """
    Plot 3D density for uniform random gas.

    Expected: Relatively flat density field
    """
    from revelsMD.trajectories import NumpyTrajectory
    from revelsMD.revels_3D import Revels3D

    print("\n=== Uniform Gas Density ===")

    np.random.seed(42)
    n_atoms = 500
    n_frames = 10
    box = 10.0

    positions = np.random.uniform(0, box, (n_frames, n_atoms, 3))
    forces = np.random.randn(n_frames, n_atoms, 3) * 0.1
    species = ['1'] * n_atoms

    ts = NumpyTrajectory(positions, forces, box, box, box, species, units='lj', temperature=1.0)

    nbins = 20
    gs = Revels3D.GridState(ts, 'number', nbins=nbins, temperature=1.0)
    gs.accumulate(ts, '1', kernel='triangular', rigid=False)
    gs.get_real_density()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # XY slice
    ax = axes[0]
    im = ax.imshow(gs.rho[:, :, nbins//2].T, origin='lower', extent=[0, box, 0, box])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('XY slice at z=5')
    plt.colorbar(im, ax=ax, label='density')

    # Histogram of density values
    ax = axes[1]
    ax.hist(gs.rho.flatten(), bins=50, density=True, alpha=0.7)
    ax.axvline(x=np.mean(gs.rho), color='r', linestyle='--', label=f'Mean={np.mean(gs.rho):.3f}')
    ax.set_xlabel('Density')
    ax.set_ylabel('Frequency')
    ax.set_title('Density Distribution (should be narrow for uniform gas)')
    ax.legend()

    fig.suptitle('Uniform Random Gas: Density should be relatively flat')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'uniform_gas_density.png', dpi=150)
    plt.close()
    print(f"  Saved: uniform_gas_density.png")
    print(f"  Mean density: {np.mean(gs.rho):.4f}, Std: {np.std(gs.rho):.4f}")
    print(f"  CV (std/mean): {np.std(gs.rho)/np.mean(gs.rho):.3f} (lower is more uniform)")


def plot_lj_rdf():
    """
    Plot RDF from Example 1 LJ trajectory if available.

    Expected: Classic LJ fluid RDF with peak at ~1.1 sigma, g(r)=1 in bulk
    """
    from revelsMD.rdf import run_rdf, run_rdf_lambda

    print("\n=== LJ Fluid RDF (Example 1) ===")

    dump_file = EXAMPLES_DIR / "example_1_LJ" / "dump.nh.lammps"
    data_file = EXAMPLES_DIR / "example_1_LJ" / "data.fin.nh.data"

    if not dump_file.exists():
        print("  Skipped: Example 1 data not available")
        return

    from revelsMD.trajectories import LammpsTrajectory

    ts = LammpsTrajectory(
        str(dump_file),
        str(data_file),
        units='lj',
        atom_style="id resid type q x y z ix iy iz",
        temperature=1.35,
    )

    # Only use first 10 frames for speed
    rdf_forward = run_rdf(ts, '1', '1', delr=0.02, start=0, stop=10, from_zero=True)
    rdf_backward = run_rdf(ts, '1', '1', delr=0.02, start=0, stop=10, from_zero=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(rdf_forward[0], rdf_forward[1], 'b-', label='Forward')
    ax1.plot(rdf_backward[0], rdf_backward[1], 'r-', label='Backward')
    ax1.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='g=1')
    ax1.set_xlabel('r (LJ units)')
    ax1.set_ylabel('g(r)')
    ax1.set_title('LJ Fluid RDF (Example 1)')
    ax1.legend()
    ax1.set_xlim(0, 5)
    ax1.grid(True, alpha=0.3)

    # Zoom on first peak
    mask = rdf_backward[0] < 2.5
    ax2.plot(rdf_forward[0][mask], rdf_forward[1][mask], 'b-', label='Forward')
    ax2.plot(rdf_backward[0][mask], rdf_backward[1][mask], 'r-', label='Backward')
    ax2.axvline(x=1.12, color='g', linestyle='--', alpha=0.7, label='Expected peak ~1.12')
    ax2.set_xlabel('r (LJ units)')
    ax2.set_ylabel('g(r)')
    ax2.set_title('First Peak Region')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'lj_fluid_rdf.png', dpi=150)
    plt.close()
    print(f"  Saved: lj_fluid_rdf.png")

    # Find first peak
    peak_mask = (rdf_backward[0] > 0.8) & (rdf_backward[0] < 1.5)
    if np.any(peak_mask):
        peak_idx = np.argmax(rdf_backward[1][peak_mask])
        peak_r = rdf_backward[0][peak_mask][peak_idx]
        print(f"  First peak at r={peak_r:.3f} (expected ~1.12 for LJ fluid)")

    # Check bulk value
    bulk_mask = rdf_backward[0] > 3.0
    if np.any(bulk_mask):
        bulk_mean = np.mean(rdf_backward[1][bulk_mask])
        print(f"  Bulk g(r) mean: {bulk_mean:.3f} (expected ~1.0)")


def plot_vasp_rdf():
    """
    Plot RDF from VASP BaSnF4 trajectory if available.

    Expected: Ionic structure peaks
    """
    from revelsMD.rdf import run_rdf, run_rdf_lambda

    print("\n=== VASP BaSnF4 RDF ===")

    vasprun_file = EXAMPLES_DIR / "vasprun.xml"

    if not vasprun_file.exists():
        print("  Skipped: VASP data not available")
        return

    from revelsMD.trajectories import VaspTrajectory

    ts = VaspTrajectory(str(vasprun_file), temperature=600)

    # F-F RDF
    rdf_ff = run_rdf(ts, 'F', 'F', delr=0.1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rdf_ff[0], rdf_ff[1], 'b-', linewidth=2, label='F-F')
    ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='g=1')
    ax.set_xlabel('r (Angstrom)')
    ax.set_ylabel('g(r)')
    ax.set_title('BaSnF4 F-F RDF (VASP AIMD)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'vasp_ff_rdf.png', dpi=150)
    plt.close()
    print(f"  Saved: vasp_ff_rdf.png")


def plot_lambda_combination():
    """
    Plot lambda-combined RDF showing optimal variance reduction.
    """
    from revelsMD.trajectories import NumpyTrajectory
    from revelsMD.rdf import run_rdf, run_rdf_lambda

    print("\n=== Lambda-Combined RDF ===")

    np.random.seed(42)
    n_atoms = 500
    n_frames = 10
    box = 10.0

    positions = np.random.uniform(0, box, (n_frames, n_atoms, 3))
    forces = np.random.randn(n_frames, n_atoms, 3) * 0.1
    species = ['1'] * n_atoms

    ts = NumpyTrajectory(positions, forces, box, box, box, species, units='lj', temperature=1.0)

    rdf_lambda = run_rdf_lambda(ts, '1', '1', delr=0.2)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(rdf_lambda[:, 0], rdf_lambda[:, 1], 'b-', linewidth=2)
    ax1.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='g=1')
    ax1.set_ylabel('g_lambda(r)')
    ax1.set_title('Lambda-Combined RDF')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(rdf_lambda[:, 0], rdf_lambda[:, 2], 'r-', linewidth=2)
    ax2.axhline(y=0.5, color='k', linestyle='--', alpha=0.5, label='lambda=0.5')
    ax2.set_xlabel('r')
    ax2.set_ylabel('lambda(r)')
    ax2.set_title('Optimal Lambda Parameter (0=backward, 1=forward)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.1, 1.1)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'lambda_combined_rdf.png', dpi=150)
    plt.close()
    print(f"  Saved: lambda_combined_rdf.png")
    print(f"  Lambda range: [{np.min(rdf_lambda[:, 2]):.3f}, {np.max(rdf_lambda[:, 2]):.3f}]")


def main():
    print("Generating validation plots for visual inspection...")
    ensure_output_dir()

    # Synthetic data tests
    plot_uniform_gas_rdf()
    plot_two_atom_rdf()
    plot_cubic_lattice_rdf()
    plot_single_atom_density()
    plot_uniform_density()
    plot_lambda_combination()

    # Real data tests
    plot_lj_rdf()
    plot_vasp_rdf()

    print(f"\n=== Done ===")
    print(f"All plots saved to: {OUTPUT_DIR}")
    print("\nReview these plots to verify:")
    print("  1. Uniform gas RDF approaches g(r)=1")
    print("  2. Two-atom RDF has peak at r=3.0")
    print("  3. Cubic lattice shows peaks at expected spacings")
    print("  4. Single atom density peaks at atom location")
    print("  5. Uniform density is relatively flat")
    print("  6. LJ fluid shows expected first peak ~1.12 sigma")
    print("  7. Lambda parameter varies smoothly between 0 and 1")


if __name__ == "__main__":
    main()
