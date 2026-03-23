# Architecture overview

This page is aimed at contributors. It describes how the RevelsMD package is
structured and the key design decisions behind each component.

## Package layout

```
revelsMD/
    __init__.py
    backends.py          # backend selection (numba / numpy)
    cell.py              # cell geometry utilities
    frame_sources.py     # Frame dataclass, contiguous_blocks, interleaved_blocks
    statistics.py        # WelfordAccumulator3D, compute_lambda_weights, combine_estimators
    density/
        density_grid.py  # DensityGrid accumulation and FFT conversion
        selection.py     # Selection: atom picking, weight computation
        grid_helpers.py  # triangular_allocation / box_allocation (numpy backend)
        grid_helpers_numba.py  # numba backend equivalents
        constants.py     # density type validation
        writers/         # output helpers
    rdf/
        rdf.py           # RDF class
        rdf_helpers.py   # pairwise / bin helpers (numpy backend)
        rdf_helpers_numba.py  # numba backend equivalents
    trajectories/
        _base.py         # Trajectory ABC, normalize_bounds
        lammps.py        # LAMMPS dump reader
        mda.py           # MDAnalysis adapter
        numpy.py         # in-memory numpy trajectory
        vasp.py          # VASP OUTCAR reader
```

## Two-level API

Every calculation is available at two levels. Convenience functions
(`compute_rdf`, `compute_density`) handle the common case in one call and are
thin wrappers around the class-based API. The classes (`RDF`, `DensityGrid`)
expose full control over frame ranges, bin parameters, and incremental
accumulation from multiple trajectories.

```python
# Convenience
grid = compute_density(traj, atom_names='O', nbins=50)

# Class-based (equivalent result, more control)
grid = DensityGrid(traj, density_type='number', nbins=50)
grid.accumulate(traj, atom_names='O', start=100)
print(grid.rho_force.mean())
```

## Trajectory interface

All trajectory backends implement `Trajectory` (`revelsMD/trajectories/_base.py`),
an abstract base class that defines:

- `frames` — total frame count
- `cell_matrix` — 3x3 array with rows as lattice vectors (works for any cell geometry)
- `units`, `temperature`, `beta` — unit system and inverse thermal energy
- `get_indices(atype)` — atom indices for a named species (abstract)
- `get_charges(atype)`, `get_masses(atype)` — raise `DataUnavailableError` by default; subclasses override as needed
- `iter_frames(start, stop, stride)` — sequential frame iteration, yields `Frame` instances
- `get_frame(index)` — random access by index (abstract)

The base class normalises start/stop/stride bounds (negative indices, `None` stop)
before delegating to `_iter_frames_impl()`, which subclasses implement with
non-negative bounds.

### Frame

`Frame` (`revelsMD/frame_sources.py`) is a frozen dataclass:

```python
@dataclass(frozen=True, slots=True, eq=False)
class Frame:
    positions: np.ndarray   # shape (n_atoms, 3)
    forces: np.ndarray      # shape (n_atoms, 3)
```

`__post_init__` validates that both arrays are 2D with a second dimension of 3,
and that the atom counts match. The `frozen=True` constraint prevents accidental
mutation after creation. `eq=False` leaves identity-based equality in place.

## The deposit/accumulate pattern

Both `DensityGrid` and `RDF` follow the same two-level pattern:

- `deposit()` — low-level single-frame method; user controls the iteration loop.
- `accumulate()` — convenience wrapper that iterates frames and calls `deposit()`.

This separation makes it straightforward to add custom iteration logic
(e.g. subsampling, multi-trajectory accumulation) without duplicating any
normalisation or bookkeeping code.

### DensityGrid.deposit takes raw arrays

`DensityGrid.deposit(positions, forces, weights, kernel)` operates below the
`Selection` abstraction. It accepts raw numpy arrays (or lists of arrays for
multi-species) directly. The `accumulate()` method builds a `Selection` from the
provided `atom_names` and calls `Selection.extract(frame)` to produce the inputs
for each `deposit()` call.

This design keeps `deposit()` general enough to be called with pre-processed data,
without requiring the caller to go through a trajectory object.

### RDF.deposit takes a Frame

`RDF.deposit(frame)` accepts a `Frame` directly and performs all atom selection
internally using the indices stored during `__init__`. This is appropriate because
the RDF always computes a pairwise property between two named species — the
selection is fixed at construction and there is no equivalent of the flexible
`Selection` class.

This is a genuine structural difference: `DensityGrid` supports runtime
reconfiguration of what is deposited (rigid molecules, charge weights, different
species), while `RDF` is always a two-species pairwise calculation.

## Selection and Selection.extract

`Selection` (`revelsMD/density/selection.py`) bridges a `Frame` to the inputs
required by `DensityGrid.deposit()`. It is constructed once per `accumulate()`
call with the trajectory and `atom_names` configuration, then called once per
frame via `extract(frame)`.

`extract(frame)` returns a `(positions, forces, weights)` tuple:

- **positions**: selected atom positions, or COM for rigid molecules, or a list of
  per-species arrays for non-rigid multi-species selections.
- **forces**: selected forces, summed across the molecule for rigid cases.
- **weights**: `1.0` for number density; per-atom charges for charge density;
  dipole projection along `polarisation_axis` for polarisation density.

The minimum-image convention is applied inside `Selection` when computing COMs and
dipole projections for molecules that may span periodic boundaries.

## Backend system

The backend controls which implementation of numerically intensive inner loops is
used. It is selected once at import time from the `REVELSMD_BACKEND` environment
variable (default: `'numba'`).

`get_backend()` (`revelsMD/backends.py`) returns the resolved backend name.
`get_backend_functions()` in `grid_helpers.py` and `rdf_helpers.py` returns the
appropriate pair of functions for that backend.

The numba backend (`grid_helpers_numba.py`, `rdf_helpers_numba.py`) provides
JIT-compiled implementations for production speed. The numpy backend uses
`np.add.at()` for correct accumulation when multiple particles share a voxel —
standard fancy indexing with `+=` silently keeps only the last write for duplicate
indices.

FFT parallelism is configured separately via `REVELSMD_FFT_WORKERS`.

## Statistical machinery

`revelsMD/statistics.py` provides three components:

**`WelfordAccumulator3D`** — accumulates per-voxel variance and covariance across
blocks using a weighted online algorithm. The caller provides
`update(delta, rho_force, weight)` for each block, where `delta = rho_force - rho_count`.
`finalise()` returns population variance and covariance arrays. At least two blocks
are required.

**`compute_lambda_weights(variance, covariance)`** — computes the optimal per-voxel
combination weight $\lambda = \text{Cov}(\delta, \rho_\text{force}) / \text{Var}(\delta)$.
Zero-variance voxels and non-finite values are mapped to zero (pure counting density).

**`combine_estimators(rho_count, rho_force, weights)`** — evaluates the linear
combination $(1-\lambda)\,\rho_\text{count} + \lambda\,\rho_\text{force}$ and
sanitises non-finite values.

These are used internally by `DensityGrid` when `compute_lambda=True` is passed to
`accumulate()`. The computed $\lambda$ weights and combined density are exposed as
`grid.lambda_weights` and `grid.rho_lambda`.

## Cell geometry

Simulation cells are represented throughout as a 3x3 `cell_matrix` with rows as
lattice vectors. This handles both orthorhombic and non-orthorhombic cells
uniformly. For orthorhombic cells, convenience properties `box_x`, `box_y`, `box_z`
expose the diagonal elements; they raise `AttributeError` on non-orthorhombic cells
to prevent misuse.

`DensityGrid` works in fractional coordinates internally: positions are transformed
to $[0, 1)$ before grid assignment. This makes bin edges and voxel sizes
dimensionless and generalises correctly to triclinic cells.

## Lazy density computation

`DensityGrid` does not compute density arrays during `accumulate()`. The
`rho_force` and `rho_count` properties trigger the FFT conversion and normalisation
on first access. The conversion uses:

$$
\delta\tilde{\rho}(\mathbf{k}) = \frac{i\beta}{k^2} \,\mathbf{k} \cdot \tilde{\mathbf{F}}(\mathbf{k})
$$

This provides an exact solution to the Poisson equation under periodic boundary
conditions with $O(N \log N)$ scaling. `rho_lambda` is similarly lazy, finalising
the Welford accumulator only when first accessed.

Accumulating additional frames invalidates all cached densities so that stale
results are never returned.
