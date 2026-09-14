# Architecture overview

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
        vasp.py          # VASP vasprun.xml reader
```

## Two-level API

Every calculation is available at two levels. Convenience functions
(`compute_rdf`, `compute_density`) handle the common case in one call as thin
wrappers around the class-based API. The classes (`RDF`, `DensityGrid`) expose
full control over frame ranges, bin parameters, and multi-trajectory
accumulation.

```python
# Convenience
grid = compute_density(traj, atom_names='O', nbins=50)

# Class-based (equivalent result, more control)
grid = DensityGrid(traj, density_type='number', nbins=50)
grid.accumulate(traj, atom_names='O', start=100)
print(grid.rho_force.mean())
```

## Trajectory interface

All backends implement `Trajectory` (`revelsMD/trajectories/_base.py`), which
defines:

- `frames` — total frame count
- `cell_matrix` — 3x3 array with rows as lattice vectors (works for any cell geometry)
- `units`, `temperature`, `beta` — unit system and inverse thermal energy
- `get_indices(atype)` — atom indices for a named species (abstract)
- `get_charges(atype)`, `get_masses(atype)` — raise `DataUnavailableError` by default; subclasses override as needed
- `iter_frames(start, stop, stride)` — sequential frame iteration, yields `Frame` instances
- `get_frame(index)` — random access by index (abstract)

The base class normalises start/stop/stride (negative indices, `None` stop) before
delegating to `_iter_frames_impl()`, which subclasses implement with non-negative
bounds.

### Frame

`Frame` (`revelsMD/frame_sources.py`) is a frozen dataclass:

```python
@dataclass(frozen=True, slots=True, eq=False)
class Frame:
    positions: np.ndarray   # shape (n_atoms, 3)
    forces: np.ndarray      # shape (n_atoms, 3)
```

`__post_init__` validates that both arrays are 2D with second dimension 3 and
that atom counts match. `frozen=True` prevents mutation after creation.
`eq=False` leaves identity-based equality in place.

## The deposit/accumulate pattern

Both `DensityGrid` and `RDF` follow the same two-level pattern:

- `deposit()` — low-level single-frame method; user controls the iteration loop.
- `accumulate()` — convenience wrapper that iterates frames and calls `deposit()`.

This separation lets you add custom iteration logic (e.g. subsampling,
multi-trajectory accumulation) without duplicating normalisation or bookkeeping.

### DensityGrid.deposit takes raw arrays

`DensityGrid.deposit(positions, forces, weights, kernel)` operates below
`Selection`, accepting raw numpy arrays (or lists of arrays for multi-species).
Each call deposits exactly one frame: for a multi-species selection, every
array in the lists belongs to that same frame, and `count` advances by one
per call rather than per array. `accumulate()` builds a `Selection` from
`atom_names` and calls `Selection.extract(frame)` to produce inputs for each
`deposit()` call.

This keeps `deposit()` general enough to call with pre-processed data, without
requiring a trajectory.

### RDF.deposit takes a Frame

`RDF.deposit(frame)` accepts a `Frame` and performs atom selection internally
using indices stored at `__init__`. This is appropriate because the RDF always
computes a pairwise property between two fixed species — there is no equivalent
of the flexible `Selection` class.

This is a structural difference: `DensityGrid` supports runtime reconfiguration
(rigid molecules, charge weights, different species), while `RDF` is always a
two-species pairwise calculation.

## Selection and Selection.extract

`Selection` (`revelsMD/density/selection.py`) bridges `Frame` to
`DensityGrid.deposit()` inputs. It is constructed once per `accumulate()` call
with the trajectory and `atom_names`, then called per frame via
`extract(frame)`.

`extract(frame)` returns a `(positions, forces, weights)` tuple:

- **positions**: selected atom positions, or COM for rigid molecules, or a list of
  per-species arrays for non-rigid multi-species selections.
- **forces**: selected forces, summed across the molecule for rigid cases.
- **weights**: `1.0` for number density; per-atom charges for charge density;
  dipole projection along `polarisation_axis` for polarisation density.

`Selection` applies the minimum-image convention when computing COMs and dipole
projections for molecules spanning periodic boundaries.

## Backend system

The backend selects the implementation of inner loops, chosen at import time from
`REVELSMD_BACKEND` (default: `'numba'`).

`get_backend()` (`revelsMD/backends.py`) returns the resolved name.
`get_backend_functions()` in `grid_helpers.py` and `rdf_helpers.py` returns the
matching function pair.

The numba backend (`grid_helpers_numba.py`, `rdf_helpers_numba.py`) provides
JIT-compiled implementations. The numpy backend uses `np.add.at()` for correct
accumulation when multiple particles share a voxel — `+=` with fancy indexing
silently drops duplicate writes.

FFT parallelism is configured separately via `REVELSMD_FFT_WORKERS`.

## Statistical machinery

`revelsMD/statistics.py` provides three components:

**`WelfordAccumulator3D`** — accumulates per-voxel variance and covariance across
blocks using a weighted online algorithm. The caller calls
`update(delta, rho_force, weight)` per block, where `delta = rho_force - rho_count`.
`finalise()` returns population variance and covariance arrays. At least two blocks
are required.

**`compute_lambda_weights(variance, covariance)`** — computes the optimal per-voxel
combination weight $\lambda = \text{Cov}(\delta, \rho_\text{force}) / \text{Var}(\delta)$.
Zero-variance voxels and non-finite values are mapped to zero (pure counting density).

**`combine_estimators(rho_count, rho_force, weights)`** — evaluates the linear
combination $(1-\lambda)\,\rho_\text{count} + \lambda\,\rho_\text{force}$ and
sanitises non-finite values.

`DensityGrid` uses these when `compute_lambda=True`, exposing the results as
`grid.lambda_weights` and `grid.rho_lambda`.

## Cell geometry

Simulation cells are a 3x3 `cell_matrix` with rows as lattice vectors, handling
orthorhombic and non-orthorhombic cells uniformly. For orthorhombic cells,
`box_x`, `box_y`, `box_z` expose the diagonal elements; they raise
`AttributeError` on non-orthorhombic cells.

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
