"""
Invariant test for the RDF lambda combination.

Encodes what lambda is FOR (Coles et al. (2021), J. Chem. Phys. 154,
191101, Eqs. 3-4): the combination E_lambda = E_0 + lambda*Delta with
Delta = E_1 - E_0 and lambda* = -Cov(E_0, Delta)/Var(Delta) is
variance-minimising, so wherever one estimator is exact (zero variance
across frames), the in-sample optimal combination must reproduce that
estimator to float precision. Here E_0 = g_inf (the backward estimator)
and E_1 = g_0 (the forward estimator) -- note the collision in the
paper's notation: E_0 is NOT g_0. rdf.py's internals are named
g_forward/g_backward (rather than zero/inf) to avoid inheriting it.

This is the test a pinned-output regression cannot provide: it fails on
any inversion of the combination (e.g. commit 40d443f) regardless of
whether reference baselines have been regenerated.
"""

import numpy as np
import pytest

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
NOISE_AMPLITUDES = {5: 0.40, 6: 0.25}
assert set(NOISE_AMPLITUDES) == set(range(NOISE_LO, NOISE_HI)), (
    "NOISE_AMPLITUDES must cover exactly the bins NOISE_LO..NOISE_HI-1"
)


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
    reproduces `scaled_frames` to floating-point round-trip precision."""
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
        for b, amp in NOISE_AMPLITUDES.items():
            frame[b] += amp * sign
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


@pytest.mark.parametrize("size", [1e-7, 1e-6, 0.1])
def test_delta_constant_to_rounding_takes_the_negligible_variance_fallback(size):
    """Var(Delta) at rounding level is treated as zero: lambda 0, g_lambda backward.

    Delta in every bin is the frame's total minus 1, so frames that move
    weight between bins without changing their total give a Delta that is
    constant to rounding while both estimators vary. Dividing by that
    variance would give a huge lambda and a garbage or zeroed g_lambda. Bins with
    a small g(r) are the hardest case: Delta's rounding is set by the
    total, of order 1, not by g(r) in the bin.
    """
    rdf = _make_rdf()
    n_bins = len(rdf._bins)
    base_profile = np.full(n_bins, size / 2)
    base_profile[-1] = 1 + size - base_profile[:-1].sum()
    frames = []
    for f in range(N_FRAMES):
        moved = size * (0.3 * (-1) ** f + 0.01 * f)
        frame = base_profile.copy()
        frame[2] += moved
        frame[5] -= moved
        frames.append(frame)
    _inject_synthetic_frames(rdf, frames)

    rdf.get_rdf(integration='lambda')

    backward = np.mean([1 - np.cumsum(frame[::-1])[::-1][1:] for frame in frames], axis=0)
    np.testing.assert_array_equal(rdf.lam[1:], 0.0)
    np.testing.assert_allclose(rdf.g[1:], backward[:-1], rtol=0, atol=1e-12)


def test_lambda_matches_a_calculation_from_per_frame_estimators():
    """lambda and g_lambda equal Eqs. 3-4 evaluated on per-frame forward and backward g(r).

    On the reported grid r_k = k*delr, forward g(r_k) sums the bins below
    r_k and backward g(r_k) is 1 minus the sum of the bins from r_k out.
    At r = 0 the forward estimator is exactly 0, so g = 0 and lambda = 1.
    """
    rdf = _make_rdf()
    n_bins = len(rdf._bins)
    rng = np.random.default_rng(11)
    frames = [rng.normal(0.1, 0.05, n_bins) for _ in range(N_FRAMES)]
    _inject_synthetic_frames(rdf, frames)
    rdf.get_rdf(integration='lambda')

    s = np.array(frames)
    k = np.arange(1, n_bins - 1)
    forward = np.array([[f[:i].sum() for i in k] for f in s])
    backward = np.array([[1 - f[i:].sum() for i in k] for f in s])
    delta = forward - backward
    var = np.mean((delta - delta.mean(axis=0)) ** 2, axis=0)
    cov = np.mean((backward - backward.mean(axis=0)) * (delta - delta.mean(axis=0)), axis=0)
    lam = -cov / var
    g = (1 - lam) * backward.mean(axis=0) + lam * forward.mean(axis=0)

    np.testing.assert_allclose(rdf.r, np.arange(n_bins - 1) * 0.1, atol=1e-12)
    assert rdf.g[0] == 0.0 and rdf.lam[0] == 1.0
    np.testing.assert_allclose(rdf.lam[1:], lam, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(rdf.g[1:], g, rtol=1e-9, atol=1e-12)


def test_single_frame_degenerate_contract_returns_backward():
    """With one frame Var(Delta) = 0: lambda == 0 for r > 0 and g_lambda equals the backward estimator bit-for-bit."""
    rng = np.random.default_rng(3)
    positions = np.zeros((1, 2, 3))
    positions[0, 1, 0] = 0.55
    forces = rng.normal(scale=1.0, size=(1, 2, 3))
    trajectory = NumpyTrajectory(
        positions, forces,
        box_x=10.0, box_y=10.0, box_z=10.0,
        species_list=['A', 'A'],
        temperature=1.0, units='lj',
    )
    rdf = RDF(trajectory, 'A', 'A', delr=0.1, rmax=1.0)
    rdf.accumulate(trajectory)

    # get_rdf recomputes from the unmutated accumulators, so the same
    # object can safely be re-run with a different integration; capture
    # the lambda results first because the standard path clears lam.
    rdf.get_rdf(integration='lambda')
    g_lambda = rdf.g.copy()
    lam = rdf.lam.copy()
    rdf.get_rdf(integration='backward')
    g_backward = rdf.g

    np.testing.assert_array_equal(
        lam[1:], np.zeros_like(lam[1:]),
        err_msg="degenerate input (single frame) must give lambda = 0 for r > 0",
    )
    # At r = 0 the forward estimator is exact: g = 0 and lambda = 1, while
    # the backward estimator reports 1 minus the full integral there. All
    # other points must match the backward estimator exactly (no tolerance).
    assert g_lambda[0] == 0.0 and lam[0] == 1.0
    np.testing.assert_array_equal(
        g_lambda[1:], g_backward[1:],
        err_msg=(
            "g_lambda must equal the backward estimator bit-for-bit under "
            "the degenerate Var(Delta) = 0 contract"
        ),
    )
