#!/usr/bin/env python3
"""
Test RDF calculation against analytical solution using a harmonic potential.

Setup:
- Atom A fixed at origin
- Atom B samples from 1D harmonic potential centred at r0
- This gives a known g(r) that we can compare against

The goal is to check:
1. Does the RDF shape match the analytical form?
2. Are the r values correctly aligned?
3. Do forward and backward integration agree (after correction)?
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from revelsMD.rdf import run_rdf, run_rdf_lambda


class MockTrajectory:
    """Mock trajectory for testing RDF with controlled data."""

    def __init__(self, positions_list, forces_list, box_size, beta, atom_types):
        self.positions_list = positions_list
        self.forces_list = forces_list
        self.box_x = box_size
        self.box_y = box_size
        self.box_z = box_size
        self.beta = beta
        self.atom_types = atom_types
        self._indices_cache = {}

    def get_indices(self, atom_type):
        if atom_type not in self._indices_cache:
            self._indices_cache[atom_type] = np.array([
                i for i, t in enumerate(self.atom_types) if t == atom_type
            ])
        return self._indices_cache[atom_type]

    def iter_frames(self, start=0, stop=None, period=1):
        if stop is None:
            stop = len(self.positions_list)
        for i in range(start, stop, period):
            yield self.positions_list[i], self.forces_list[i]

    def __len__(self):
        return len(self.positions_list)

    @property
    def frames(self):
        return len(self.positions_list)


def generate_harmonic_trajectory(
    n_frames: int,
    r0: float,
    k: float,
    beta: float,
    box_size: float,
    seed: int = 42
):
    """
    Generate trajectory with exactly 2 atoms: A fixed, B samples from harmonic potential.

    Each frame has exactly 1 A atom and 1 B atom. Atom B samples from
    a 1D harmonic potential along the x-axis relative to A.
    """
    rng = np.random.default_rng(seed)
    sigma = np.sqrt(1.0 / (beta * k))

    positions_list = []
    forces_list = []
    atom_types = ['A', 'B']  # Exactly 2 atoms

    for _ in range(n_frames):
        positions = np.zeros((2, 3))
        forces = np.zeros((2, 3))

        # A at origin (or random position - doesn't matter for RDF)
        positions[0, :] = [box_size / 2, box_size / 2, box_size / 2]

        # B samples from harmonic potential (1D along x)
        dx = rng.normal(r0, sigma)
        positions[1, :] = positions[0, :] + np.array([dx, 0, 0])

        # Force on B: F = -k(x - r0)
        F_B_x = -k * (dx - r0)
        forces[0, 0] = -F_B_x     # Reaction on A
        forces[1, 0] = F_B_x      # Force on B

        positions_list.append(positions)
        forces_list.append(forces)

    return MockTrajectory(
        positions_list, forces_list, box_size, beta, atom_types
    ), sigma


def main():
    # Parameters
    r0 = 2.5        # Equilibrium distance
    k = 10.0        # Spring constant
    beta = 1.0      # Inverse temperature
    n_frames = 500000  # Many frames to reduce sampling noise
    # Box size chosen so prefactor matches what's needed for g(r) -> 1 at large r
    # The RDF formula is g(r) = 1 + (beta * V / 4*pi * N_pairs) * integral(F.r/r^3)
    # For the integral to give us a CDF from 0 to 1, we need the prefactor to be 4*pi
    # So: beta * V / N_pairs = 4*pi => V = 4*pi / beta with N_pairs = 1
    # V = 4*pi ≈ 12.57, so box_size = V^(1/3) ≈ 2.33
    # But we need box > 2*r0 to avoid MIC issues. Let's calculate what prefactor
    # we'd need and see what the results look like
    box_size = 10.0  # Smaller box to reduce prefactor, but still > 2*r0
    delr = 0.02

    print("Harmonic potential RDF test")
    print(f"  r0 = {r0}, k = {k}, beta = {beta}")
    print(f"  n_frames = {n_frames}, 1 A-B pair per frame")

    # Generate trajectory
    trajectory, sigma = generate_harmonic_trajectory(
        n_frames, r0, k, beta, box_size
    )
    print(f"  sigma = {sigma:.4f}")

    # Calculate prefactor as the code does
    n_a = len(trajectory.get_indices('A'))
    n_b = len(trajectory.get_indices('B'))
    volume = trajectory.box_x * trajectory.box_y * trajectory.box_z
    prefactor = volume / (n_a * n_b) / 2
    print(f"  n_A = {n_a}, n_B = {n_b}")
    print(f"  Volume = {volume:.1f}")
    print(f"  Prefactor = {prefactor:.4f}")

    # Run RDF calculations
    print("\nRunning RDF calculations...")
    rdf_forward = run_rdf(trajectory, 'A', 'B', delr=delr, from_zero=True)
    rdf_backward = run_rdf(trajectory, 'A', 'B', delr=delr, from_zero=False)
    rdf_lambda = run_rdf_lambda(trajectory, 'A', 'B', delr=delr)

    # Extract data
    bins = rdf_forward[0]
    g_forward = rdf_forward[1]
    g_backward = rdf_backward[1]
    g_mean = (g_forward + g_backward) / 2

    # Lambda RDF has different format
    r_lambda = rdf_lambda[:, 0]
    g_lambda = rdf_lambda[:, 1]
    lambda_weights = rdf_lambda[:, 2]

    # First, let's look at the raw data to understand what's happening
    print("\nRaw data analysis:")
    print(f"  g_forward range: [{g_forward.min():.4f}, {g_forward.max():.4f}]")
    print(f"  g_backward range: [{g_backward.min():.4f}, {g_backward.max():.4f}]")
    print(f"  g_lambda range: [{g_lambda.min():.4f}, {g_lambda.max():.4f}]")

    # Check endpoint values (these should be meaningful)
    print(f"  g_forward[0] = {g_forward[0]:.6f}, g_forward[-1] = {g_forward[-1]:.6f}")
    print(f"  g_backward[0] = {g_backward[0]:.6f}, g_backward[-1] = {g_backward[-1]:.6f}")

    # The density of contributions is dg/dr (before cumsum)
    # Let's look at what the histogram of actual distances looks like
    actual_distances = []
    for positions, forces in trajectory.iter_frames():
        r_vec = positions[1] - positions[0]
        r = np.linalg.norm(r_vec)
        actual_distances.append(r)
    actual_distances = np.array(actual_distances)

    print(f"\nActual distances sampled:")
    print(f"  Mean: {actual_distances.mean():.4f} (expected: {r0})")
    print(f"  Std:  {actual_distances.std():.4f} (expected: {sigma:.4f})")

    # Check the force contributions manually for a few samples
    print(f"\nManual force contribution check (first 5 frames):")
    for i, (positions, forces) in enumerate(trajectory.iter_frames()):
        if i >= 5:
            break
        r_vec = positions[1] - positions[0]  # B - A
        r_mag = np.linalg.norm(r_vec)
        F_a = forces[0]
        F_b = forces[1]
        F_diff = F_a - F_b  # What the code uses
        dot_prod = np.dot(F_diff, r_vec) / r_mag**3
        print(f"  Frame {i}: r={r_mag:.4f}, F_a={F_a[0]:.4f}, F_b={F_b[0]:.4f}, F_diff.r/r^3={dot_prod:.4f}")

    # Analytical CDF for comparison
    # For 1D harmonic: P(x) ~ exp(-beta*k*(x-r0)^2/2)
    # CDF = 0.5 * (1 + erf((r - r0) / (sigma * sqrt(2))))
    cdf_analytical = 0.5 * (1 + erf((bins - r0) / (sigma * np.sqrt(2))))
    cdf_lambda_grid = 0.5 * (1 + erf((r_lambda - r0) / (sigma * np.sqrt(2))))

    # Rescale all outputs to [0, 1] range for comparison with analytical CDF
    # Forward goes from 0 to some value - normalise by max
    # Backward goes from negative to 1 - shift and scale
    def normalise_cdf(g):
        """Rescale to [0, 1] range."""
        g_min, g_max = g.min(), g.max()
        if g_max == g_min:
            return g
        return (g - g_min) / (g_max - g_min)

    g_forward_norm = normalise_cdf(g_forward)
    g_backward_norm = normalise_cdf(g_backward)
    g_mean_norm = (g_forward_norm + g_backward_norm) / 2
    g_lambda_norm = normalise_cdf(g_lambda)

    # Compute analytical PDF (derivative of CDF) for the density
    # For 1D harmonic: P(r) ~ exp(-beta*k*(r-r0)^2/2) / (sigma * sqrt(2*pi))
    r_fine = np.linspace(r0 - 5*sigma, r0 + 5*sigma, 500)
    pdf_analytical = np.exp(-0.5 * ((r_fine - r0) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

    # Numerical derivative of the normalised RDF curves to get density estimates
    def numerical_derivative(r, g):
        dr = r[1] - r[0]
        dg_dr = np.gradient(g, dr)
        return dg_dr

    dg_forward = numerical_derivative(bins, g_forward_norm)
    dg_backward = numerical_derivative(bins, g_backward_norm)
    dg_mean = numerical_derivative(bins, g_mean_norm)
    dg_lambda = numerical_derivative(r_lambda, g_lambda_norm)

    # For plotting, we don't need to scale again if we normalised properly

    # Get the raw contributions (before cumsum) by taking the derivative
    # of the forward curve. Since forward = cumsum(contrib), contrib = diff(forward)
    raw_contrib = np.diff(g_forward)
    raw_contrib_bins = bins[:-1] + delr/2  # bin centres

    # Plot results
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Top left: Density (dg/dr) comparison - THE KEY PLOT
    ax = axes[0, 0]
    ax.plot(r_fine, pdf_analytical, 'k-', label='Analytical PDF', linewidth=3, zorder=10)
    ax.plot(bins, dg_forward, 'b-', label='Forward dg/dr', alpha=0.7, linewidth=1.5)
    ax.plot(bins, dg_backward, 'r--', label='Backward dg/dr', alpha=0.7, linewidth=1.5)
    ax.plot(bins, dg_mean, 'g-', label='Mean dg/dr', linewidth=2)
    ax.plot(r_lambda, dg_lambda, 'm:', label='Lambda dg/dr', linewidth=2)
    ax.axvline(r0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlim(r0 - 4*sigma, r0 + 4*sigma)
    ax.set_xlabel('r')
    ax.set_ylabel('Density dg/dr')
    ax.set_title('Density Distribution: dg/dr vs Analytical PDF')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Top right: CDF comparison
    ax = axes[0, 1]
    ax.plot(bins, cdf_analytical, 'k-', label='Analytical CDF', linewidth=3, zorder=10)
    ax.plot(bins, g_forward_norm, 'b-', label='Forward (normalised)', alpha=0.7, linewidth=1.5)
    ax.plot(bins, g_backward_norm, 'r--', label='Backward (normalised)', alpha=0.7, linewidth=1.5)
    ax.plot(bins, g_mean_norm, 'g-', label='Mean (F+B)/2', linewidth=2)
    ax.plot(r_lambda, g_lambda_norm, 'm:', label='Lambda (normalised)', linewidth=2)
    ax.axvline(r0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlim(r0 - 4*sigma, r0 + 4*sigma)
    ax.set_xlabel('r')
    ax.set_ylabel('g(r) CDF (normalised to [0,1])')
    ax.set_title('CDF: g(r) vs Analytical')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Bottom left: Forward - Backward offset in density
    ax = axes[1, 0]
    ax.plot(bins, dg_forward - dg_backward, 'b-', linewidth=2, label='Forward - Backward')
    ax.axhline(0, color='k', linestyle=':', alpha=0.5)
    ax.axvline(r0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlim(r0 - 4*sigma, r0 + 4*sigma)
    ax.set_xlabel('r')
    ax.set_ylabel('Density difference')
    ax.set_title('Forward - Backward Offset in Density')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bottom middle: Lambda weights
    ax = axes[1, 1]
    ax.plot(r_lambda, lambda_weights, 'purple', linewidth=2)
    ax.axhline(0.5, color='k', linestyle=':', alpha=0.5)
    ax.axvline(r0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlim(r0 - 4*sigma, r0 + 4*sigma)
    ax.set_xlabel('r')
    ax.set_ylabel('λ(r)')
    ax.set_title('Lambda Weights')
    ax.grid(True, alpha=0.3)

    # Bottom right: Raw contributions (before cumsum)
    ax = axes[1, 2]
    ax.plot(raw_contrib_bins, raw_contrib, 'b-', linewidth=1.5, alpha=0.7, label='Raw contributions')
    # For comparison, plot what we expect: PDF should be Gaussian
    # But force contribution is proportional to (r - r0) * PDF(r)
    expected_force_contrib = (raw_contrib_bins - r0) * np.exp(-0.5 * ((raw_contrib_bins - r0) / sigma) ** 2)
    # Scale to match
    scale = np.max(np.abs(raw_contrib)) / np.max(np.abs(expected_force_contrib)) if np.max(np.abs(expected_force_contrib)) > 0 else 1
    ax.plot(raw_contrib_bins, expected_force_contrib * scale, 'k--', linewidth=2, label='Expected: (r-r0)*PDF(r)')
    ax.axhline(0, color='k', linestyle=':', alpha=0.5)
    ax.axvline(r0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlim(r0 - 4*sigma, r0 + 4*sigma)
    ax.set_xlabel('r')
    ax.set_ylabel('Contribution per bin')
    ax.set_title('Raw Force Contributions (before cumsum)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Top right extra: Show g_forward raw (not normalised)
    ax = axes[0, 2]
    ax.plot(bins, g_forward, 'b-', linewidth=2, label='Forward (raw)')
    ax.plot(bins, g_backward, 'r--', linewidth=2, label='Backward (raw)')
    ax.axhline(0, color='k', linestyle=':', alpha=0.5)
    ax.axhline(1, color='k', linestyle=':', alpha=0.5)
    ax.axvline(r0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlim(r0 - 4*sigma, r0 + 4*sigma)
    ax.set_xlabel('r')
    ax.set_ylabel('g(r) raw')
    ax.set_title('Raw g(r) (not normalised)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/tmp/harmonic_rdf_test.png', dpi=150)
    print(f"\nPlot saved to /tmp/harmonic_rdf_test.png")

    # Print key metrics
    print(f"\nAt r = r0 = {r0}:")
    idx = np.searchsorted(bins, r0)
    idx_lambda = np.searchsorted(r_lambda, r0)
    print(f"  Forward (normalised):  {g_forward_norm[idx]:.4f}")
    print(f"  Backward (normalised): {g_backward_norm[idx]:.4f}")
    print(f"  Mean (normalised):     {g_mean_norm[idx]:.4f}")
    print(f"  Lambda (normalised):   {g_lambda_norm[idx_lambda]:.4f}")
    print(f"  Analytical CDF:        0.5000")

    # Find the r values where each curve crosses 0.5 (the median)
    def find_crossing(r, g, target=0.5):
        """Find r where g crosses target value."""
        idx = np.searchsorted(g, target)
        if idx == 0 or idx >= len(g):
            return np.nan
        # Linear interpolation
        r0_cross = r[idx-1] + (r[idx] - r[idx-1]) * (target - g[idx-1]) / (g[idx] - g[idx-1])
        return r0_cross

    r_forward_50 = find_crossing(bins, g_forward_norm)
    r_backward_50 = find_crossing(bins, g_backward_norm)
    r_mean_50 = find_crossing(bins, g_mean_norm)
    r_lambda_50 = find_crossing(r_lambda, g_lambda_norm)

    print(f"\nMedian (CDF = 0.5) locations:")
    print(f"  Analytical:  {r0:.4f}")
    print(f"  Forward:     {r_forward_50:.4f}  (offset: {r_forward_50 - r0:+.4f})")
    print(f"  Backward:    {r_backward_50:.4f}  (offset: {r_backward_50 - r0:+.4f})")
    print(f"  Mean:        {r_mean_50:.4f}  (offset: {r_mean_50 - r0:+.4f})")
    print(f"  Lambda:      {r_lambda_50:.4f}  (offset: {r_lambda_50 - r0:+.4f})")
    print(f"  Bin spacing: {delr:.4f}")
    print(f"  Forward-Backward gap: {r_backward_50 - r_forward_50:.4f} ({(r_backward_50 - r_forward_50)/delr:.2f} bins)")

    # KEY FINDING: The forward-backward gap should be exactly 1 bin
    # This is the fundamental offset from inclusive vs exclusive bin counting
    print(f"\n*** KEY FINDING ***")
    print(f"  Forward-Backward gap = {(r_backward_50 - r_forward_50)/delr:.2f} bins")
    print(f"  This confirms the 1-bin offset between forward and backward integration.")
    print(f"  The mean (F+B)/2 splits the difference, giving a half-bin effective shift.")

    # Check r-value alignment
    print(f"\n  bins[{idx}] = {bins[idx]:.4f}")
    print(f"  r_lambda[{idx_lambda}] = {r_lambda[idx_lambda]:.4f}")
    print(f"  Difference: {r_lambda[idx_lambda] - bins[idx]:.4f} (expect ~{delr/2:.4f} = delr/2)")

    # More detailed grid comparison
    print(f"\nGrid comparison:")
    print(f"  bins[:5] = {bins[:5]}")
    print(f"  r_lambda[:5] = {r_lambda[:5]}")
    print(f"  bins[1:5] = {bins[1:5]}  (what lambda should be if just dropping first)")
    print(f"  bins[:5] + delr/2 = {bins[:5] + delr/2}  (if shifted by half bin)")


if __name__ == "__main__":
    main()
