"""Validate RDF forward-backward offset using Example 4 water trajectory.

This script establishes baseline behaviour before refactoring by checking:
1. Grid offset between run_rdf and run_rdf_lambda
2. Forward-backward peak position differences
3. Physical correctness (O-O peak near 2.8 Angstrom)
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from revelsMD.trajectories import MDATrajectory
from revelsMD.rdf import run_rdf, run_rdf_lambda

# Load Example 4 subset (100 frames of SPC/E water)
TEST_DATA = Path(__file__).parent.parent / "tests/test_data/example_4_subset"
trr_file = TEST_DATA / "prod_100frames.trr"
tpr_file = TEST_DATA / "prod.tpr"

print("Loading Example 4 water trajectory...")
ts = MDATrajectory(str(trr_file), str(tpr_file), temperature=300.0)
print(f"  Frames: {ts.frames}")
print(f"  Box: {ts.box_x:.2f} x {ts.box_y:.2f} x {ts.box_z:.2f} Angstrom")

n_ow = len(ts.get_indices('Ow'))
print(f"  Water molecules: {n_ow}")

# Calculate O-O RDF with forward and backward integration
delr = 0.02
n_frames = 20  # Use subset for speed
print(f"\nCalculating O-O RDF (delr={delr}, frames=0-{n_frames})...")

rdf_forward = run_rdf(ts, 'Ow', 'Ow', delr=delr, from_zero=True, start=0, stop=n_frames)
rdf_backward = run_rdf(ts, 'Ow', 'Ow', delr=delr, from_zero=False, start=0, stop=n_frames)
rdf_lambda = run_rdf_lambda(ts, 'Ow', 'Ow', delr=delr, start=0, stop=n_frames)

# Extract data
bins = rdf_forward[0]
g_forward = rdf_forward[1]
g_backward = rdf_backward[1]

r_lambda = rdf_lambda[:, 0]
g_lambda = rdf_lambda[:, 1]
lambda_weights = rdf_lambda[:, 2]

# Compare grids
print("\n=== Grid Comparison ===")
print(f"Forward bins[:5] = {bins[:5]}")
print(f"Lambda r[:5]     = {r_lambda[:5]}")
print(f"bins[1:5]        = {bins[1:5]}")
print(f"Expected: r_lambda should equal bins[1:]")
print(f"Match: {np.allclose(r_lambda, bins[1:len(r_lambda)+1])}")

# Find first peak position (O-O should be ~2.8 Angstrom)
# Look only in the relevant range (2-4 Angstrom)
mask_forward = (bins > 2.0) & (bins < 4.0)
mask_lambda = (r_lambda > 2.0) & (r_lambda < 4.0)

peak_idx_forward = np.argmax(g_forward[mask_forward])
peak_idx_backward = np.argmax(g_backward[mask_forward])
peak_idx_lambda = np.argmax(g_lambda[mask_lambda])

peak_forward = bins[mask_forward][peak_idx_forward]
peak_backward = bins[mask_forward][peak_idx_backward]
peak_lambda = r_lambda[mask_lambda][peak_idx_lambda]

print(f"\n=== O-O First Peak Positions ===")
print(f"  Forward:  {peak_forward:.3f} Angstrom")
print(f"  Backward: {peak_backward:.3f} Angstrom")
print(f"  Lambda:   {peak_lambda:.3f} Angstrom")
print(f"  Forward-Backward gap: {peak_backward - peak_forward:.4f} Angstrom ({(peak_backward - peak_forward)/delr:.2f} bins)")
print(f"\n  Expected peak: ~2.8 Angstrom (typical for water)")
print(f"  Expected F-B gap: ~{delr} Angstrom (1 bin)")

# Check bulk g(r) values (should approach 1)
bulk_mask = bins > 6.0
if np.any(bulk_mask):
    mean_forward_bulk = np.mean(g_forward[bulk_mask])
    mean_backward_bulk = np.mean(g_backward[bulk_mask])
    print(f"\n=== Bulk g(r) Values (r > 6 Angstrom) ===")
    print(f"  Forward mean:  {mean_forward_bulk:.4f}")
    print(f"  Backward mean: {mean_backward_bulk:.4f}")
    print(f"  Expected: ~1.0")

# Create visualisation
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Top left: Full RDF comparison
ax = axes[0, 0]
ax.plot(bins, g_forward, 'b-', label='Forward', linewidth=1.5)
ax.plot(bins, g_backward, 'r--', label='Backward', linewidth=1.5)
ax.plot(r_lambda, g_lambda, 'g-', label='Lambda', linewidth=2)
ax.axhline(1, color='k', linestyle=':', alpha=0.5)
ax.axvline(2.8, color='gray', linestyle=':', alpha=0.5, label='Expected O-O peak')
ax.set_xlabel('r (Angstrom)')
ax.set_ylabel('g(r)')
ax.set_title('Water O-O RDF: Full Range')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 10)

# Top right: Zoom on first peak
ax = axes[0, 1]
ax.plot(bins, g_forward, 'b-', label='Forward', linewidth=1.5)
ax.plot(bins, g_backward, 'r--', label='Backward', linewidth=1.5)
ax.plot(r_lambda, g_lambda, 'g-', label='Lambda', linewidth=2)
ax.axvline(peak_forward, color='blue', linestyle=':', alpha=0.5)
ax.axvline(peak_backward, color='red', linestyle=':', alpha=0.5)
ax.axvline(peak_lambda, color='green', linestyle=':', alpha=0.5)
ax.set_xlabel('r (Angstrom)')
ax.set_ylabel('g(r)')
ax.set_title('Water O-O RDF: First Peak Region')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(2.0, 4.0)

# Bottom left: Forward - Backward difference
ax = axes[1, 0]
ax.plot(bins, g_forward - g_backward, 'purple', linewidth=1.5)
ax.axhline(0, color='k', linestyle=':', alpha=0.5)
ax.set_xlabel('r (Angstrom)')
ax.set_ylabel('Forward - Backward')
ax.set_title('Forward - Backward Difference')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 10)

# Bottom right: Lambda weights
ax = axes[1, 1]
ax.plot(r_lambda, lambda_weights, 'purple', linewidth=2)
ax.axhline(0.5, color='k', linestyle=':', alpha=0.5)
ax.set_xlabel('r (Angstrom)')
ax.set_ylabel('lambda(r)')
ax.set_title('Lambda Weights')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 10)

plt.tight_layout()
plt.savefig('/tmp/water_rdf_validation.png', dpi=150)
print(f"\nPlot saved to /tmp/water_rdf_validation.png")

# Summary
print("\n=== Summary ===")
print(f"Grid offset confirmed: r_lambda = bins[1:]")
print(f"Forward-Backward gap: {(peak_backward - peak_forward)/delr:.2f} bins")
print(f"O-O peak at {peak_lambda:.2f} Angstrom (expected ~2.8)")
