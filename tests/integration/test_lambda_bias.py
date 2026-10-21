"""
The lambda-combined density is not biased by estimating lambda from its own data.

For a fixed lambda(x), rho_lambda = rho_count + lambda * (rho_force - rho_count)
is unbiased, since rho_count and rho_force are. Estimating lambda from the
same blocks whose means it weights adds Cov(lambda, Delta), which is large
where voxels are sparsely sampled: lambda comes out large where the histogram
over-counted by chance and zero where it saw nothing, so the combination
falls below both estimators. RevelsMD estimates lambda in each time half of
the frames and applies it to the other half, which removes that term.

The test system is independent particles in a harmonic well, sampled exactly
from the Boltzmann distribution with exact forces, on a grid fine enough that
most voxels in the well see about one sample or fewer. rho_count and
rho_force each integrate to N exactly; a lambda estimated from the same
blocks gives about half of N here.
"""

import numpy as np
import pytest
from scipy.integrate import trapezoid

from revelsMD.density.density_grid import DensityGrid
from revelsMD.rdf import RDF
from revelsMD.trajectories.numpy import NumpyTrajectory

pytestmark = [pytest.mark.integration, pytest.mark.analytical]

BOX = 10.0
SIGMA = 0.6  # at most BOX/15, so the density at the cell boundary is below 1e-12
N_PARTICLES = 20
N_FRAMES = 20
NBINS = 64
SEEDS = 20


def _harmonic_well(rng):
    """Positions sampled exactly from a harmonic well at beta = 1, with the exact forces."""
    centre = BOX / 2
    positions = rng.normal(centre, SIGMA, size=(N_FRAMES, N_PARTICLES, 3))
    forces = -(positions - centre) / SIGMA**2
    return NumpyTrajectory(positions, forces, BOX, BOX, BOX, ['A'] * N_PARTICLES,
                           temperature=1.0, units='lj')


def test_rho_lambda_integrates_to_the_number_of_particles_on_average():
    """Averaged over independent runs, rho_lambda holds N particles, as rho_count and rho_force do."""
    voxel_volume = (BOX / NBINS) ** 3
    totals = []
    for seed in range(SEEDS):
        trajectory = _harmonic_well(np.random.default_rng(seed))
        grid = DensityGrid(trajectory, 'number', nbins=NBINS)
        grid.accumulate(trajectory, atom_names='A', compute_lambda=True)  # one frame per block
        totals.append(grid.rho_lambda.sum() * voxel_volume)
    totals = np.array(totals)
    standard_error = totals.std(ddof=1) / np.sqrt(SEEDS)

    # The fixed-lambda expectation differs from N only by a discretisation
    # term (rho_count and rho_force differ per voxel at second order in the
    # voxel size), about 0.01 particles here; same-sample lambda gives ~10.5.
    assert abs(totals.mean() - N_PARTICLES) < 0.03 * N_PARTICLES, (
        f"mean integral of rho_lambda {totals.mean():.2f} +/- {standard_error:.2f} "
        f"over {SEEDS} runs, expected {N_PARTICLES}"
    )


# The RDF lambda combines the forward and backward estimators in the same way,
# with one sample per frame. Its same-sample bias falls as 1/frames, so it
# shows at a few frames: about a fifth of the pair count at 10 frames for
# this harmonic pair.
PAIR_K, PAIR_R0, PAIR_BETA, PAIR_BOX, PAIR_DELR = 62.5, 2.5, 0.4, 10.0, 0.05
PAIR_FRAMES = 10
PAIR_SEEDS = 300


def _pair_weight(r):
    """The radial measure r^2 exp(-beta u(r)) of the harmonic pair, unnormalised."""
    return r**2 * np.exp(-PAIR_BETA * 0.5 * PAIR_K * (r - PAIR_R0) ** 2)


def _harmonic_pair(rng):
    """One A-B pair with u = k (r - r0)^2 / 2, sampled exactly at beta, with the exact forces."""
    envelope = _pair_weight(np.linspace(1.0, 4.0, 4001)).max() * 1.05
    separations = []
    while len(separations) < PAIR_FRAMES:
        r = rng.uniform(1.0, 4.0)
        if rng.uniform(0.0, envelope) < _pair_weight(r):
            separations.append(r)
    r = np.array(separations)
    v = rng.normal(size=(PAIR_FRAMES, 3))
    direction = v / np.linalg.norm(v, axis=1, keepdims=True)
    centre = np.full(3, PAIR_BOX / 2)
    positions = np.empty((PAIR_FRAMES, 2, 3))
    positions[:, 0] = centre
    positions[:, 1] = centre + r[:, None] * direction
    forces = np.empty((PAIR_FRAMES, 2, 3))
    forces[:, 1] = (-PAIR_K * (r - PAIR_R0))[:, None] * direction
    forces[:, 0] = -forces[:, 1]
    return NumpyTrajectory(positions, forces, PAIR_BOX, PAIR_BOX, PAIR_BOX, ['A', 'B'],
                           temperature=1 / PAIR_BETA, units='lj')


def _rdf_estimates(trajectory):
    """The lambda, forward and backward g(r), and lambda, of one run."""
    rdf = RDF(trajectory, 'A', 'B', delr=PAIR_DELR)
    rdf.accumulate(trajectory)
    rdf.get_rdf('lambda')
    g_lambda, lam = rdf.g.copy(), rdf.lam.copy()
    rdf.get_rdf('forward')
    g_forward = rdf.g.copy()
    rdf.get_rdf('backward')
    return rdf.r, g_lambda, g_forward, rdf.g.copy(), lam


def test_rdf_lambda_is_not_biased_by_estimating_lambda_from_the_same_frames():
    """g_lambda agrees on average with the same combination using lambda from an independent run.

    A lambda from another run is independent of this run's estimators, so
    (1 - lambda) g_backward + lambda g_forward has exactly the expectation of
    the combination at a fixed lambda. The pair count inside rmax is
    compared, the integral of 4 pi r^2 g(r) / V.
    """
    differences = []
    for seed in range(PAIR_SEEDS):
        r, g_lambda, g_forward, g_backward, _ = _rdf_estimates(_harmonic_pair(np.random.default_rng(seed)))
        lam = _rdf_estimates(_harmonic_pair(np.random.default_rng(10**6 + seed)))[4]
        reference = (1 - lam) * g_backward + lam * g_forward
        differences.append(trapezoid(4 * np.pi * r**2 * (g_lambda - reference), r) / PAIR_BOX**3)
    differences = np.array(differences)
    standard_error = differences.std(ddof=1) / np.sqrt(PAIR_SEEDS)

    assert abs(differences.mean()) < 4 * standard_error, (
        f"g_lambda differs from the independent-lambda combination by "
        f"{differences.mean():+.4f} +/- {standard_error:.4f} pairs on average"
    )
