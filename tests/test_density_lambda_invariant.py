"""
Invariant tests for the DensityGrid lambda combination.

Encodes what lambda is FOR (Coles et al. (2021), J. Chem. Phys. 154,
191101, Eqs. 3-4): the combination E_lambda = E_0 + lambda*Delta with
Delta = E_1 - E_0 and lambda* = -Cov(E_0, Delta)/Var(Delta) is
variance-minimising, so wherever one estimator is exact (zero variance
across blocks), the optimal combination must reproduce that estimator to
float precision. For densities E_0 = rho_count and E_1 = rho_force.
lambda is estimated in each time half of the frames and applied to the
other half; where the halves agree, as they do where one estimator is
exact, rho_lambda = (1 - lambda) * rho_count + lambda * rho_force with the
exposed lambda_weights.

Pinned-output tests can show that a result changed, but not which
estimator should win; these tests encode that. Because rho_force comes from
a global FFT solve, the RDF invariant test's region-wise-exactness trick is
unavailable; instead these tests drive the two constructions where one
estimator is exact everywhere, and compare with lambda and rho_lambda
computed from scratch from per-frame densities.
"""

import numpy as np
import pytest

from revelsMD.density.density_grid import DensityGrid
from revelsMD.trajectories.numpy import NumpyTrajectory


BOX = 10.0
N_ATOMS = 4
N_FRAMES = 8

# Each test runs in a cubic box with cubic bins, and in a triclinic cell with
# a different number of bins along each axis.
CELLS = {
    'cubic': np.diag([BOX, BOX, BOX]),
    'triclinic': np.array([[10.0, 0.0, 0.0], [3.0, 9.0, 0.0], [1.0, 2.0, 8.0]]),
}
geometries = pytest.mark.parametrize('geometry', ['cubic', 'triclinic'])


def _nbins(geometry, n):
    return n if geometry == 'cubic' else (n, n - 1, n + 1)


def _make_trajectory(fractional, forces, geometry):
    return NumpyTrajectory(
        fractional @ CELLS[geometry], forces,
        cell_matrix=CELLS[geometry],
        species_list=['A'] * N_ATOMS,
        temperature=1.0, units='lj',
    )


@geometries
def test_lambda_reproduces_exact_count_estimator(geometry):
    """With identical positions every frame, rho_count is exact: lambda -> 0 and rho_lambda == rho_count."""
    rng = np.random.default_rng(42)
    base_fractional = rng.random((N_ATOMS, 3))
    fractional = np.broadcast_to(
        base_fractional, (N_FRAMES, N_ATOMS, 3)
    ).copy()
    # Frame-to-frame varying forces: rho_count has zero variance across
    # blocks while rho_force (and hence delta) varies, so the in-sample
    # optimal weight on rho_force is exactly zero.
    forces = rng.normal(scale=5.0, size=(N_FRAMES, N_ATOMS, 3))

    trajectory = _make_trajectory(fractional, forces, geometry)
    grid = DensityGrid(trajectory, 'number', nbins=_nbins(geometry, 8))
    # Default block_size=1: each frame is its own block (8 blocks).
    grid.accumulate(trajectory, atom_names='A', compute_lambda=True)

    weights = grid.lambda_weights
    # The guard for a negligible Var(delta) reports weight 1; with random
    # forces every voxel's Var(delta) is far from negligible, so no voxel
    # may carry the guard value.
    degenerate = weights == 1.0
    assert not degenerate.any(), (
        "unexpected zero-variance voxels with frame-varying forces"
    )
    np.testing.assert_allclose(
        weights, 0.0, atol=1e-8,
        err_msg="lambda weight on rho_force should be 0 where rho_count is exact",
    )
    np.testing.assert_allclose(
        grid.rho_lambda, grid.rho_count,
        atol=1e-10 * np.max(np.abs(grid.rho_count)),
        err_msg=(
            "rho_lambda should reproduce the exact count estimator; a large "
            "deviation means the combination weights the wrong estimator"
        ),
    )


@geometries
def test_lambda_degenerate_blocks_return_force_estimator(geometry):
    """With every frame identical, Var(delta) = 0 everywhere: lambda == 1 and rho_lambda == rho_force."""
    rng = np.random.default_rng(7)
    base_fractional = rng.random((N_ATOMS, 3))
    base_forces = rng.normal(scale=1.0, size=(N_ATOMS, 3))
    fractional = np.broadcast_to(
        base_fractional, (N_FRAMES, N_ATOMS, 3)
    ).copy()
    forces = np.broadcast_to(
        base_forces, (N_FRAMES, N_ATOMS, 3)
    ).copy()

    trajectory = _make_trajectory(fractional, forces, geometry)
    grid = DensityGrid(trajectory, 'number', nbins=_nbins(geometry, 8))
    grid.accumulate(trajectory, atom_names='A', compute_lambda=True)

    # Documented degenerate contract (lambda_weights docstring): voxels
    # whose Var(delta) is negligible report lambda = 1, i.e. the force
    # estimator.
    np.testing.assert_array_equal(
        grid.lambda_weights, np.ones_like(grid.lambda_weights),
        err_msg="degenerate voxels must report lambda = 1 (force estimator)",
    )
    np.testing.assert_array_equal(
        grid.rho_lambda, grid.rho_force,
        err_msg=(
            "rho_lambda must equal rho_force exactly under the degenerate "
            "Var(delta) = 0 contract"
        ),
    )
    # Sanity: the contract is only meaningfully pinned if the two
    # estimators actually differ.
    assert not np.array_equal(grid.rho_force, grid.rho_count)


def _fold_blocks(frames, blocking, size):
    """Blocks of one half of an accumulate() call's frames, as lists of frame indices."""
    if blocking == 'contiguous':
        return [frames[i:i + size] for i in range(0, len(frames), size)]
    return [frames[k::size] for k in range(size)]


def _cross_fitted_lambda(count, force, calls, blocking, size):
    """Cross-fitted lambda and rho_lambda from per-frame densities, from scratch.

    Each call's frames (a list of frame indices) are split by time into a
    first half (the larger when the number is odd) and a second half; fold 0
    collects the first halves, fold 1 the second. Blocks are formed within
    each half. lambda is estimated in each fold from its blocks, weighted by
    frames, and applied to the other fold's mean densities.
    """
    folds = [[], []]
    for frames in calls:
        half = -(-len(frames) // 2)
        folds[0] += _fold_blocks(frames[:half], blocking, size)
        folds[1] += _fold_blocks(frames[half:], blocking, size)
    stats = []
    for blocks in folds:
        weight = np.array([len(b) for b in blocks], dtype=float)
        block_count = np.array([count[b].mean(axis=0) for b in blocks])
        block_delta = np.array([(force[b] - count[b]).mean(axis=0) for b in blocks])
        w = weight[:, None, None, None] / weight.sum()
        mean_count = (w * block_count).sum(axis=0)
        mean_delta = (w * block_delta).sum(axis=0)
        var_delta = (w * (block_delta - mean_delta) ** 2).sum(axis=0)
        cov = (w * (block_count - mean_count) * (block_delta - mean_delta)).sum(axis=0)
        stats.append((weight.sum(), mean_count, mean_delta, -cov / var_delta))
    total = stats[0][0] + stats[1][0]
    rho_lambda = sum(n / total * (mean_count + other_lam * mean_delta)
                     for (n, mean_count, mean_delta, _), (_, _, _, other_lam)
                     in zip(stats, stats[::-1]))
    # The weight on the force density overall: each half's data carries the
    # other half's lambda.
    lam = sum(n / total * other_lam for (n, _, _, _), (_, _, _, other_lam) in zip(stats, stats[::-1]))
    return lam, rho_lambda


CROSS_FIT_CASES = {
    # name: (accumulate() calls as (start, stop), blocking, block_size or sections)
    'one call, one frame per block': ([(0, 24)], 'contiguous', 1),
    'odd frames, blocks of three': ([(0, 23)], 'contiguous', 3),
    'interleaved sections': ([(0, 24)], 'interleaved', 4),
    'two calls': ([(0, 10), (10, 24)], 'contiguous', 2),
}


@geometries
@pytest.mark.parametrize('case', CROSS_FIT_CASES)
def test_lambda_matches_a_cross_fitted_calculation_from_per_frame_densities(geometry, case):
    """lambda and rho_lambda equal the cross-fitted Eqs. 3-4 evaluated directly on per-frame densities.

    lambda is estimated in each time half of the frames and applied to the
    other half, so it is never estimated from the blocks it weights.
    """
    calls, blocking, size = CROSS_FIT_CASES[case]
    n_frames = calls[-1][1]
    rng = np.random.default_rng(67)
    fractional = rng.random((n_frames, N_ATOMS, 3))
    forces = rng.normal(scale=5.0, size=(n_frames, N_ATOMS, 3))
    trajectory = _make_trajectory(fractional, forces, geometry)
    nbins = _nbins(geometry, 4)
    grid = DensityGrid(trajectory, 'number', nbins=nbins)
    option = {'block_size': size} if blocking == 'contiguous' else {'sections': size}
    for start, stop in calls:
        grid.accumulate(trajectory, atom_names='A', start=start, stop=stop,
                        compute_lambda=True, blocking=blocking, **option)

    count, force = [], []
    for k in range(n_frames):
        frame = DensityGrid(trajectory, 'number', nbins=nbins)
        frame.accumulate(trajectory, atom_names='A', start=k, stop=k + 1)
        count.append(frame.rho_count)
        force.append(frame.rho_force)
    lam, rho_lambda = _cross_fitted_lambda(
        np.array(count), np.array(force),
        [list(range(start, stop)) for start, stop in calls], blocking, size,
    )

    np.testing.assert_allclose(grid.lambda_weights, lam, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(grid.rho_lambda, rho_lambda, rtol=1e-9, atol=1e-12)
