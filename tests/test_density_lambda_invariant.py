"""
Invariant tests for the DensityGrid lambda combination.

Encodes what lambda is FOR (Coles et al. (2021), J. Chem. Phys. 154,
191101, Eqs. 3-4): the combination E_lambda = E_0 + lambda*Delta with
Delta = E_1 - E_0 and lambda* = -Cov(E_0, Delta)/Var(Delta) is
variance-minimising, so wherever one estimator is exact (zero variance
across blocks), the in-sample optimal combination must reproduce that
estimator to float precision. For densities E_0 = rho_count and
E_1 = rho_force; the exposed lambda_weights is the Eq. 3 weight on
rho_force, so rho_lambda = (1 - lambda) * rho_count + lambda * rho_force.

This guard exists because pinned-output tests cannot catch an inverted
combination: swapping the estimator arguments in _finalise_lambda's
combine_estimators call leaves every other test passing. Because
rho_force comes from a global FFT solve, the RDF invariant test's
region-wise-exactness trick is unavailable; instead these tests drive
the two constructions where one estimator is exact everywhere.
"""

import numpy as np

from revelsMD.density.density_grid import DensityGrid
from revelsMD.trajectories.numpy import NumpyTrajectory


BOX = 10.0
N_ATOMS = 4
N_FRAMES = 6


def _make_trajectory(positions, forces):
    return NumpyTrajectory(
        positions, forces,
        box_x=BOX, box_y=BOX, box_z=BOX,
        species_list=['A'] * N_ATOMS,
        temperature=1.0, units='lj',
    )


def test_lambda_reproduces_exact_count_estimator():
    """With identical positions every frame, rho_count is exact: lambda -> 0 and rho_lambda == rho_count."""
    rng = np.random.default_rng(42)
    base_positions = rng.random((N_ATOMS, 3)) * BOX
    positions = np.broadcast_to(
        base_positions, (N_FRAMES, N_ATOMS, 3)
    ).copy()
    # Frame-to-frame varying forces: rho_count has zero variance across
    # blocks while rho_force (and hence delta) varies, so the in-sample
    # optimal weight on rho_force is exactly zero.
    forces = rng.normal(scale=5.0, size=(N_FRAMES, N_ATOMS, 3))

    trajectory = _make_trajectory(positions, forces)
    grid = DensityGrid(trajectory, 'number', nbins=8)
    # Default block_size=1: each frame is its own block (6 blocks).
    grid.accumulate(trajectory, atom_names='A', compute_lambda=True)

    weights = grid.lambda_weights
    # The degenerate Var(delta) = 0 guard reports weight 1; with random
    # forces every voxel must have Var(delta) > 0, so no voxel may carry
    # the guard value.
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


def test_lambda_degenerate_blocks_return_force_estimator():
    """With every frame identical, Var(delta) = 0 everywhere: lambda == 1 and rho_lambda == rho_force."""
    rng = np.random.default_rng(7)
    base_positions = rng.random((N_ATOMS, 3)) * BOX
    base_forces = rng.normal(scale=1.0, size=(N_ATOMS, 3))
    positions = np.broadcast_to(
        base_positions, (N_FRAMES, N_ATOMS, 3)
    ).copy()
    forces = np.broadcast_to(
        base_forces, (N_FRAMES, N_ATOMS, 3)
    ).copy()

    trajectory = _make_trajectory(positions, forces)
    grid = DensityGrid(trajectory, 'number', nbins=8)
    grid.accumulate(trajectory, atom_names='A', compute_lambda=True)

    # Documented degenerate contract (lambda_weights docstring): voxels
    # with Var(delta) = 0 report lambda = 1, i.e. the force estimator.
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
