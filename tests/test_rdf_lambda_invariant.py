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

# The synthetic runs here are short, with fewer than 10 frames in each half.
pytestmark = pytest.mark.filterwarnings('ignore:.*frames in one half of the frames')

# Frame-to-frame noise is added to bins NOISE_LO..NOISE_HI-1 of the raw
# per-bin sums S. On the reported grid r_k, forward g(r_k) sums the bins
# j < k, so it is exact for k <= NOISE_LO; backward g(r_k) is 1 minus the
# sum of the bins j >= k, so it is exact for k >= NOISE_HI. The noise alternates sign over an odd number of
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
    """Add per-frame per-bin sums through RDF._add_frame, which accumulate()
    calls for every frame, pre-dividing out the physical scale so that the RDF's
    prefactor*beta/(4*pi) multiplication reproduces `scaled_frames` to
    floating-point round-trip precision. As accumulate() does, the first half
    of the frames (the larger when their number is odd) goes to lambda fold 0
    and the rest to fold 1. Counts feed only g_count, which these tests never
    assert; ones avoid divide-by-zero noise."""
    scale = rdf._prefactor * rdf._beta / (4 * np.pi)
    half = -(-len(scaled_frames) // 2)
    for k, frame in enumerate(scaled_frames):
        rdf._add_frame(frame / scale, np.ones_like(frame), fold=int(k >= half))


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

    # Exact curves from the noise-free profile on the reported grid r_k,
    # k = 0..n_bins-2: forward (E_1) sums the bins below r_k, backward
    # (E_0) is 1 minus the sum of the bins from r_k out.
    grid = range(n_bins - 1)
    fwd_exact = np.array([base_profile[:k].sum() for k in grid])
    bwd_exact = np.array([1 - base_profile[k:].sum() for k in grid])

    # Region A (k = 0..NOISE_LO): forward exact => lambda = 1 and
    # g_lambda = forward, both exactly in-sample.
    region_a = slice(0, NOISE_LO + 1)
    np.testing.assert_allclose(
        lam[region_a], 1.0, atol=1e-8,
        err_msg="lambda should be 1 where the forward estimator is exact",
    )
    np.testing.assert_allclose(
        g[region_a], fwd_exact[region_a], atol=1e-8,
        err_msg=(
            "g_lambda should reproduce the exact forward estimator where "
            "it has zero variance; a large deviation means the combination "
            "weights the wrong estimator (the 40d443f inversion)"
        ),
    )

    # Region B (k = NOISE_HI..n_bins-2): backward exact => lambda = 0 and
    # g_lambda = backward.
    region_b = slice(NOISE_HI, n_bins - 1)
    np.testing.assert_allclose(
        lam[region_b], 0.0, atol=1e-8,
        err_msg="lambda should be 0 where the backward estimator is exact",
    )
    np.testing.assert_allclose(
        g[region_b], bwd_exact[region_b], atol=1e-8,
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

    backward = np.mean([[1 - frame[k:].sum() for k in range(1, n_bins - 1)] for frame in frames],
                       axis=0)
    np.testing.assert_array_equal(rdf.lam[1:], 0.0)
    np.testing.assert_allclose(rdf.g[1:], backward, rtol=0, atol=1e-12)


def test_lambda_matches_a_cross_fitted_calculation_from_per_frame_estimators():
    """lambda and g_lambda equal the cross-fitted Eqs. 3-4 evaluated on per-frame forward and backward g(r).

    On the reported grid r_k = k*delr, forward g(r_k) sums the bins below
    r_k and backward g(r_k) is 1 minus the sum of the bins from r_k out.
    lambda is estimated in each time half of the frames and weights the
    other half's estimators. At r = 0 the forward estimator is exactly 0,
    so g = 0 and lambda = 1.
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
    halves = [slice(0, 5), slice(5, N_FRAMES)]  # 9 frames: 5 and 4
    fold_lam = []
    for h in halves:
        delta = forward[h] - backward[h]
        var = np.mean((delta - delta.mean(axis=0)) ** 2, axis=0)
        cov = np.mean((backward[h] - backward[h].mean(axis=0)) * (delta - delta.mean(axis=0)), axis=0)
        fold_lam.append(-cov / var)
    weights = [5 / N_FRAMES, 4 / N_FRAMES]
    # Each half's estimators are weighted by the other half's lambda.
    g = sum(w * ((1 - other_lam) * backward[h].mean(axis=0) + other_lam * forward[h].mean(axis=0))
            for w, h, other_lam in zip(weights, halves, fold_lam[::-1]))
    lam = sum(w * other_lam for w, other_lam in zip(weights, fold_lam[::-1]))

    np.testing.assert_allclose(rdf.r, np.arange(n_bins - 1) * 0.1, atol=1e-12)
    assert rdf.g[0] == 0.0 and rdf.lam[0] == 1.0
    np.testing.assert_allclose(rdf.lam[1:], lam, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(rdf.g[1:], g, rtol=1e-9, atol=1e-12)


def _pair_trajectory(n_frames):
    """Two 'A' atoms 0.55 apart, with random forces, repeated for n_frames identical frames."""
    rng = np.random.default_rng(3)
    positions = np.zeros((1, 2, 3))
    positions[0, 1, 0] = 0.55
    forces = rng.normal(scale=1.0, size=(1, 2, 3))
    return NumpyTrajectory(
        np.repeat(positions, n_frames, axis=0), np.repeat(forces, n_frames, axis=0),
        box_x=10.0, box_y=10.0, box_z=10.0,
        species_list=['A', 'A'],
        temperature=1.0, units='lj',
    )


@pytest.mark.parametrize('calls', [[(0, 1)], [(0, 3)], [(k, k + 1) for k in range(4)]],
                         ids=['one frame', 'three frames', 'one-frame calls'])
def test_lambda_raises_with_fewer_than_two_frames_in_a_half(calls):
    """lambda is undefined without two frames in each half, so it raises; forward and backward still work.

    One frame gives a valid g(r) from its pairs, but no spread across frames
    from which to estimate lambda. Three frames give halves of 2 and 1, and
    one-frame accumulate() calls put every frame in the first half.
    """
    trajectory = _pair_trajectory(4)
    rdf = RDF(trajectory, 'A', 'A', delr=0.1, rmax=1.0)
    for start, stop in calls:
        rdf.accumulate(trajectory, start=start, stop=stop)

    with pytest.raises(ValueError, match="at least 2 frames in each half"):
        rdf.get_rdf(integration='lambda')
    for integration in ('forward', 'backward'):
        rdf.get_rdf(integration=integration)
        assert np.all(np.isfinite(rdf.g))


def test_identical_frames_take_the_backward_estimator():
    """With every frame identical, Var(Delta) = 0: for r > 0, lambda == 0 and g_lambda equals the backward estimator bit-for-bit."""
    trajectory = _pair_trajectory(4)
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
        err_msg="degenerate input (identical frames) must give lambda = 0 for r > 0",
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


def _water_like_trajectory(n_frames):
    """Random positions and forces for 3 'A' and 4 'B' atoms, in a 10 x 10 x 10 box."""
    rng = np.random.default_rng(29)
    positions = rng.uniform(0.0, 10.0, (n_frames, 7, 3))
    forces = rng.normal(0.0, 1.0, (n_frames, 7, 3))
    return NumpyTrajectory(positions, forces, box_x=10.0, box_y=10.0, box_z=10.0,
                           species_list=['A'] * 3 + ['B'] * 4, temperature=1.0, units='lj')


@pytest.mark.parametrize('calls', [[(0, 9)], [(0, 5), (5, 9)]], ids=['one call', 'two calls'])
def test_accumulate_splits_each_call_into_time_halves_for_lambda(calls):
    """Through accumulate(), lambda is estimated in each time half of each call and applied to the other.

    The first half of a call's frames (the larger when the number is odd)
    is one fold, the rest the other: 9 frames give frames 0-4 and 5-8;
    calls of 5 and 4 frames give 0-2 and 5-6, and 3-4 and 7-8. The
    expected g(r) and lambda come from each frame's forward and backward
    g(r), computed one frame at a time.
    """
    trajectory = _water_like_trajectory(9)
    rdf = RDF(trajectory, 'A', 'B', delr=0.5)
    for start, stop in calls:
        rdf.accumulate(trajectory, start=start, stop=stop)
    rdf.get_rdf(integration='lambda')

    forward, backward = [], []
    for k in range(9):
        single = RDF(trajectory, 'A', 'B', delr=0.5)
        single.accumulate(trajectory, start=k, stop=k + 1)
        single.get_rdf(integration='forward')
        forward.append(single.g[1:].copy())
        single.get_rdf(integration='backward')
        backward.append(single.g[1:].copy())
    forward, backward = np.array(forward), np.array(backward)

    folds = [[], []]
    for start, stop in calls:
        frames = list(range(start, stop))
        half = -(-len(frames) // 2)
        folds[0] += frames[:half]
        folds[1] += frames[half:]
    fold_lam = []
    for frames in folds:
        delta = forward[frames] - backward[frames]
        var = np.mean((delta - delta.mean(axis=0)) ** 2, axis=0)
        cov = np.mean((backward[frames] - backward[frames].mean(axis=0))
                      * (delta - delta.mean(axis=0)), axis=0)
        fold_lam.append(-cov / var)
    weights = [len(frames) / 9 for frames in folds]
    g = sum(w * ((1 - other_lam) * backward[frames].mean(axis=0) + other_lam * forward[frames].mean(axis=0))
            for w, frames, other_lam in zip(weights, folds, fold_lam[::-1]))
    lam = sum(w * other_lam for w, other_lam in zip(weights, fold_lam[::-1]))

    assert rdf.g[0] == 0.0 and rdf.lam[0] == 1.0
    np.testing.assert_allclose(rdf.lam[1:], lam, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(rdf.g[1:], g, rtol=1e-9, atol=1e-12)
