"""
Invariant test for the RDF lambda combination.

Encodes what lambda is FOR (Coles et al. (2021), J. Chem. Phys. 154,
191101, Eqs. 3-4): the combination E_lambda = E_0 + lambda*Delta with
Delta = E_1 - E_0 and lambda* = -Cov(E_0, Delta)/Var(Delta) is
variance-minimising, so wherever one estimator is exact (zero variance
across frames), the in-sample optimal combination must reproduce that
estimator to float precision. Here E_0 = g_inf (the backward estimator)
and E_1 = g_0 (the forward estimator). Note the naming collision with
rdf.py's internals: base_zero_rdf there is the FORWARD estimator (E_1,
integrated from g(0) = 0), not E_0.

This is the test a pinned-output regression cannot provide: it fails on
any inversion of the combination (e.g. commit 40d443f) regardless of
whether reference baselines have been regenerated.
"""

import numpy as np

from revelsMD.rdf import RDF
from revelsMD.trajectories import NumpyTrajectory

# Frame-to-frame noise is added to bins NOISE_LO..NOISE_HI-1 of the raw
# per-bin sums S. Forward g(r) = cumsum(S) is then exact at aligned bins
# j < NOISE_LO; backward g(r) = 1 - revcumsum(S) is exact at aligned bins
# j >= NOISE_HI - 1. The noise alternates sign over an odd number of
# frames, so its sample mean is nonzero: the noise alone guarantees an
# inverted combination is off by at least ~(0.4 + 0.25)/9 ~ 0.07 where
# the correct one is exact, even if the profile were ever normalised so
# the forward/backward estimator gap vanished. In practice base_profile
# sums to 1.2, so under the historical mutation (40d443f) the observed
# deviation is dominated by that gap and is ~0.27.
N_FRAMES = 9
NOISE_LO, NOISE_HI = 5, 7


def _make_rdf():
    positions = np.zeros((1, 2, 3))
    positions[0, 1, 0] = 1.0
    forces = np.zeros((1, 2, 3))
    trajectory = NumpyTrajectory(
        positions, forces,
        box_x=10.0, box_y=10.0, box_z=10.0,
        species_list=['A', 'A'],
        temperature=1.0, units='lj',
    )
    return RDF(trajectory, 'A', 'A', delr=0.1, rmax=1.0)


def _inject_synthetic_frames(rdf, scaled_frames):
    """Install per-frame per-bin sums, pre-dividing out the physical scale
    so that _compute_lambda's prefactor*beta/(4*pi) multiplication
    reproduces `scaled_frames` exactly."""
    scale = rdf._prefactor * rdf._beta / (4 * np.pi)
    raw_frames = [frame / scale for frame in scaled_frames]
    rdf._frame_data = raw_frames
    rdf._accumulated = np.sum(raw_frames, axis=0)
    # Counts feed only _compute_g_count, which this test never asserts;
    # ones simply avoid divide-by-zero noise.
    rdf._counts = np.ones_like(rdf._accumulated)
    rdf._frame_count = len(raw_frames)
    rdf.progress = 'accumulated'


def test_lambda_reproduces_exact_estimator_in_each_region():
    """g_lambda equals whichever estimator has zero variance, with lambda 1 or 0."""
    rdf = _make_rdf()
    n_bins = len(rdf._bins)
    assert n_bins == 12  # 11 real bins + overflow, from delr=0.1, rmax=1.0

    base_profile = np.linspace(0.05, 0.15, n_bins)
    frames = []
    for f in range(N_FRAMES):
        sign = (-1) ** f
        frame = base_profile.copy()
        frame[NOISE_LO] += 0.40 * sign
        frame[NOISE_HI - 1] += 0.25 * sign
        frames.append(frame)
    _inject_synthetic_frames(rdf, frames)

    rdf.get_rdf(integration='lambda')
    g = rdf.g
    lam = rdf.lam

    # Expected exact curves from the noise-free profile, using the same
    # alignment as _compute_lambda: forward = cumsum[:-1] (E_1), backward
    # = 1 - revcumsum[1:] (E_0), both on aligned bins j = 0..n_bins-2.
    fwd_exact = np.cumsum(base_profile)[:-1]
    bwd_exact = 1 - np.cumsum(base_profile[::-1])[::-1][1:]

    # Output layout: g[0] = 0 padding; g[1 + j] = g_lambda at aligned bin j
    # for j = 0..n_bins-3 (the last aligned bin is dropped). Same for lam.

    # Region A (aligned j = 0..NOISE_LO-1): forward exact => lambda = 1
    # and g_lambda = forward, both exactly in-sample.
    region_a = slice(0, NOISE_LO)
    np.testing.assert_allclose(
        lam[1:][region_a], 1.0, atol=1e-8,
        err_msg="lambda should be 1 where the forward estimator is exact",
    )
    np.testing.assert_allclose(
        g[1:][region_a], fwd_exact[region_a], atol=1e-8,
        err_msg=(
            "g_lambda should reproduce the exact forward estimator where "
            "it has zero variance; a large deviation means the combination "
            "weights the wrong estimator (the 40d443f inversion)"
        ),
    )

    # Region B (aligned j = NOISE_HI-1..n_bins-3): backward exact =>
    # lambda = 0 and g_lambda = backward.
    region_b = slice(NOISE_HI - 1, n_bins - 2)
    np.testing.assert_allclose(
        lam[1:][region_b], 0.0, atol=1e-8,
        err_msg="lambda should be 0 where the backward estimator is exact",
    )
    np.testing.assert_allclose(
        g[1:][region_b], bwd_exact[region_b], atol=1e-8,
        err_msg=(
            "g_lambda should reproduce the exact backward estimator where "
            "it has zero variance"
        ),
    )
