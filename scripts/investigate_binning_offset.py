#!/usr/bin/env python3
"""
Investigate the forward/backward integration offset and whether
different binning strategies can resolve it.

This script:
1. Shows the one-bin offset between forward and backward cumsum
2. Tests whether 50/50 bin splitting removes the offset
3. Tests triangular deposition as an alternative
"""

import numpy as np
import matplotlib.pyplot as plt


def nearest_bin_assignment(distances: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """
    Current behaviour: assign each distance to exactly one bin.
    Uses np.digitize (right-edge assignment).
    """
    n_bins = len(bins)
    bin_indices = np.digitize(distances, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    contributions = np.ones_like(distances)  # unit contribution per particle
    storage = np.bincount(bin_indices, weights=contributions, minlength=n_bins)
    return storage.astype(np.float64)


def uniform_split_assignment(distances: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """
    50/50 split: each distance contributes equally to two adjacent bins.
    """
    n_bins = len(bins)
    delr = bins[1] - bins[0]  # assume uniform spacing
    storage = np.zeros(n_bins, dtype=np.float64)

    for d in distances:
        # Find which bin centre this is closest to
        bin_idx = int(d / delr)
        if bin_idx >= n_bins:
            continue

        # Split 50/50 between this bin and the next
        if bin_idx < n_bins:
            storage[bin_idx] += 0.5
        if bin_idx + 1 < n_bins:
            storage[bin_idx + 1] += 0.5

    return storage


def triangular_deposition(distances: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """
    Triangular deposition: weight split by distance from bin centres.
    A particle at bin centre -> 100% to that bin.
    A particle halfway between centres -> 50% to each.
    """
    n_bins = len(bins)
    delr = bins[1] - bins[0]  # assume uniform spacing
    storage = np.zeros(n_bins, dtype=np.float64)

    for d in distances:
        # Find the two nearest bin centres
        lower_idx = int(d / delr)
        upper_idx = lower_idx + 1

        if lower_idx >= n_bins:
            continue

        # Distance from lower bin centre (as fraction of delr)
        lower_centre = bins[lower_idx] if lower_idx < n_bins else bins[-1]
        frac = (d - lower_centre) / delr
        frac = np.clip(frac, 0, 1)

        # Triangular weights
        weight_lower = 1 - frac
        weight_upper = frac

        if lower_idx < n_bins:
            storage[lower_idx] += weight_lower
        if upper_idx < n_bins:
            storage[upper_idx] += weight_upper

    return storage


def forward_cumsum(contributions: np.ndarray) -> np.ndarray:
    """Forward integration: g(r) = cumsum from r=0"""
    return np.cumsum(contributions)


def backward_cumsum(contributions: np.ndarray) -> np.ndarray:
    """
    Backward integration: g(r) = 1 - cumsum from r=inf

    This matches the actual code:
        1 - np.cumsum(contributions[::-1])[::-1]
    """
    return 1 - np.cumsum(contributions[::-1])[::-1]


def backward_cumsum_shifted(contributions: np.ndarray) -> np.ndarray:
    """
    Backward integration with one-bin shift to align with forward.

    The idea: if forward at bin i includes contribution[i],
    then backward at bin i should represent "everything from i onwards"
    which is 1 - cumsum up to (but not including) i.
    """
    # cumsum[::-1] at position i gives sum from i to end (inclusive)
    # We want sum from i+1 to end, so shift by one
    rev_cumsum = np.cumsum(contributions[::-1])[::-1]
    # Shift: prepend 0, drop last element
    rev_cumsum_shifted = np.concatenate([[0], rev_cumsum[:-1]])
    return 1 - rev_cumsum_shifted


def plot_comparison(bins, forward, backward, title):
    """Plot forward and backward curves with their difference."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(bins, forward, 'b-', label='Forward (from zero)', linewidth=2)
    ax1.plot(bins, backward, 'r--', label='Backward (from inf)', linewidth=2)
    ax1.set_ylabel('g(r)')
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    diff = forward - backward
    ax2.plot(bins, diff, 'g-', linewidth=2)
    ax2.axhline(0, color='k', linestyle=':', alpha=0.5)
    ax2.set_xlabel('r')
    ax2.set_ylabel('Forward - Backward')
    ax2.set_title(f'Difference (max abs: {np.max(np.abs(diff)):.6f})')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def main():
    # Create test data: random distances that should give a known RDF
    np.random.seed(42)
    n_particles = 10000

    # Simulate distances with a distribution that has structure
    # (e.g., excluded volume at small r, peaks at certain distances)
    distances = np.concatenate([
        np.random.normal(2.5, 0.3, n_particles // 2),  # first shell
        np.random.normal(4.5, 0.5, n_particles // 2),  # second shell
    ])
    distances = distances[distances > 0]  # remove negative values
    distances = distances[distances < 6]  # cutoff

    # Set up bins
    delr = 0.1
    rmax = 6.0
    bins = np.arange(0, rmax, delr)

    print(f"Number of distances: {len(distances)}")
    print(f"Number of bins: {len(bins)}")
    print(f"Bin spacing: {delr}")
    print()

    # Test each binning strategy
    strategies = [
        ("Nearest bin (current)", nearest_bin_assignment),
        ("50/50 uniform split", uniform_split_assignment),
        ("Triangular deposition", triangular_deposition),
    ]

    results = {}

    for name, strategy in strategies:
        contributions = strategy(distances, bins)

        # Normalise so total = 1 (like a proper RDF)
        contributions = contributions / np.sum(contributions)

        forward = forward_cumsum(contributions)
        backward = backward_cumsum(contributions)

        diff = forward - backward
        max_diff = np.max(np.abs(diff))

        print(f"{name}:")
        print(f"  Max |forward - backward|: {max_diff:.6f}")
        print(f"  Forward endpoint: {forward[-1]:.6f}")
        print(f"  Backward startpoint: {backward[0]:.6f}")
        print()

        results[name] = {
            'contributions': contributions,
            'forward': forward,
            'backward': backward,
        }

        fig = plot_comparison(bins, forward, backward, name)
        fig.savefig(f"/tmp/binning_{name.replace(' ', '_').replace('/', '_')}.png", dpi=150)
        plt.close(fig)

    # Let's understand the math more carefully
    print("Understanding the offset mathematically:")
    contributions = nearest_bin_assignment(distances, bins)
    contributions = contributions / np.sum(contributions)
    n = len(contributions)

    print(f"\n  contributions sum = {np.sum(contributions):.6f}")

    # Forward: F[i] = sum(c[0:i+1])
    # Backward (current): B[i] = 1 - sum(c[i:n]) reversed back
    #                          = 1 - sum(c[i:n])
    #
    # For these to match: F[i] = B[i]
    #   sum(c[0:i+1]) = 1 - sum(c[i:n])
    #   sum(c[0:i+1]) = 1 - sum(c[i:n])
    #   sum(c[0:i+1]) + sum(c[i:n]) = 1
    #   sum(c[0:i]) + c[i] + sum(c[i:n]) = 1
    #   sum(c[0:i]) + c[i] + c[i] + sum(c[i+1:n]) = 1
    #   total + c[i] = 1
    #
    # So they differ by c[i] at each point!

    forward = forward_cumsum(contributions)
    backward = backward_cumsum(contributions)

    # The difference should be exactly contributions[i]
    diff = forward - backward
    print(f"\n  forward - backward at each bin should equal contributions:")
    print(f"  Max |diff - contributions| = {np.max(np.abs(diff - contributions)):.10f}")

    # So the "correct" aligned curves would be:
    # Forward: includes c[i] at bin i
    # Backward: excludes c[i] at bin i
    # Average: (F + B) / 2 = F - c[i]/2 = B + c[i]/2
    mean_curve = (forward + backward) / 2
    print(f"\n  Mean curve = forward - contributions/2:")
    print(f"  Max |mean - (forward - contributions/2)| = {np.max(np.abs(mean_curve - (forward - contributions/2))):.10f}")

    # Plot to visualise
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(bins, forward, 'b-', label='Forward', linewidth=2)
    ax1.plot(bins, backward, 'r--', label='Backward', linewidth=2)
    ax1.plot(bins, mean_curve, 'g-', label='Mean (F+B)/2', linewidth=2, alpha=0.7)
    ax1.set_ylabel('g(r)')
    ax1.set_title('Forward, Backward, and Mean')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(bins, diff, 'b-', label='forward - backward', linewidth=2)
    ax2.plot(bins, contributions, 'r--', label='contributions', linewidth=2)
    ax2.set_xlabel('r')
    ax2.set_ylabel('Difference')
    ax2.set_title('Offset equals the contribution at each bin')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig('/tmp/binning_offset_explained.png', dpi=150)
    plt.close(fig)

    print(f"\n  Plot saved to /tmp/binning_offset_explained.png")

    print()
    print(f"Plots saved to /tmp/binning_*.png")


if __name__ == "__main__":
    main()
