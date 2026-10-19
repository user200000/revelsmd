"""
Synthetic trajectories behind the stored synthetic baselines.

scripts/generate_reference_data.py and the regression tests both build their
inputs here, so the two cannot drift apart. Each system draws from its own
``RandomState``, which gives the same numbers as the legacy global
``np.random.seed`` stream the baselines were generated with, but is not
affected by draws elsewhere. ``RandomState``'s stream is frozen across NumPy
versions (NEP 19); a ``default_rng`` stream is not, and could change the
inputs under the stored baselines.
"""

import numpy as np

from revelsMD.trajectories import NumpyTrajectory

UNIFORM_GAS_SEED = 42
UNIFORM_GAS_ATOMS = 500
UNIFORM_GAS_FRAMES = 50
UNIFORM_GAS_BOX = 10.0

TRICLINIC_SEED = 77
TRICLINIC_CELL = np.array([
    [10.0, 0.0, 0.0],
    [3.0, 9.0, 0.0],
    [1.0, 2.0, 8.0],
])
TRICLINIC_ATOMS = 300
TRICLINIC_FRAMES = 20


def uniform_gas_trajectory() -> NumpyTrajectory:
    """Uniformly random positions and random forces in a cubic box."""
    rng = np.random.RandomState(UNIFORM_GAS_SEED)
    shape = (UNIFORM_GAS_FRAMES, UNIFORM_GAS_ATOMS, 3)
    positions = rng.uniform(0, UNIFORM_GAS_BOX, shape)
    forces = rng.randn(*shape) * 0.1
    box = UNIFORM_GAS_BOX
    return NumpyTrajectory(
        positions, forces, box, box, box, ['1'] * UNIFORM_GAS_ATOMS,
        temperature=1.0, units='lj',
    )


def triclinic_trajectory() -> NumpyTrajectory:
    """Uniformly random positions and random forces in a triclinic cell."""
    rng = np.random.RandomState(TRICLINIC_SEED)
    shape = (TRICLINIC_FRAMES, TRICLINIC_ATOMS, 3)
    fractional = rng.rand(*shape)
    positions = np.einsum('fai,ij->faj', fractional, TRICLINIC_CELL)
    forces = rng.randn(*shape) * 0.1
    return NumpyTrajectory(
        positions, forces, cell_matrix=TRICLINIC_CELL,
        species_list=['1'] * TRICLINIC_ATOMS,
        temperature=1.0, units='lj',
    )
