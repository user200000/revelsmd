# Architecture overview

## Package layout

```
revelsMD/
    __init__.py
    backends.py          # backend selection (numba / numpy)
    cell.py              # cell geometry utilities
    frame_sources.py     # Frame dataclass, contiguous_blocks, interleaved_blocks
    statistics.py        # WelfordAccumulator3D, control_variate_combination
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
        _base.py         # Trajectory ABC, normalize_bounds, frame_range, check_same_system
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
- `get_frame(index)` — random access by index; it checks the index and calls `_get_frame()`, which subclasses implement

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

## Accumulation

The constructor of `DensityGrid` or `RDF` takes the system from a trajectory:
its cell and beta (temperature and units), and for the RDF the number of atoms
of each of its two species. It keeps those values, not the trajectory. Frames go in only
through `accumulate()`, which takes a trajectory of that system, raises if its
cell, temperature or units differ (or, for the RDF, its species counts), and
iterates its frames. Calling `accumulate()` again adds frames from the same
trajectory or another of the same system, such as an independent run; see
{doc}`/how-to/numpy-interface` for a set of structures.

Whole frames are deposited at a time, so a frame cannot be split across calls.
`DensityGrid` builds its `Selection` from each call's `atom_names`, so a grid
can still mix selections across calls: accumulating `'O'` and then `'H'` from
the same frames counts each frame twice, giving half the total that
`['O', 'H']` gives in one call.

### DensityGrid deposition

Without `compute_lambda`, `accumulate()` calls the private
`DensityGrid._deposit(positions, forces, weights, kernel)` once per frame, with
what `Selection.extract(frame)` returns: arrays for one species or rigid
molecules, lists of arrays for several species. `count` advances by one per
frame, not per array. With `compute_lambda=True`, `_accumulate_blocks` deposits
each block's frames into block buffers instead, and adds them to the grid once
the block is complete.

### RDF deposition

`RDF._deposit(frame)` takes a `Frame` and selects the atoms of the two species
using the indices found in the trajectory being accumulated. The RDF always
computes a pairwise property between two species, so it needs no `Selection`.

## Selection and Selection.extract

`Selection` (`revelsMD/density/selection.py`) turns a `Frame` into the arrays
`DensityGrid` deposits. It is constructed once per `accumulate()` call
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

`revelsMD/statistics.py` provides two components:

**`WelfordAccumulator3D`** — accumulates per-voxel variance and covariance across
blocks using a weighted online algorithm. The caller calls
`update(delta, rho_force, weight)` per block, where `delta = rho_force - rho_count`.
`finalise()` returns population variance and covariance arrays. At least two blocks
are required.

**`control_variate_combination(mean_e0, mean_e1, var_delta, cov_e0_delta, scale=..., fallback=...)`**
— combines two estimators with the variance-minimising weight
$\lambda = -\text{Cov}(E_0, \delta)/\text{Var}(\delta)$, where $\delta = E_1 - E_0$
(Coles et al. 2021), returning $(1-\lambda)\,E_0 + \lambda\,E_1$ and $\lambda$. Where the variance is
negligible for `scale`, or the ratio is non-finite, $\lambda$ is `fallback`.

`DensityGrid` uses both when `compute_lambda=True`, with $E_0 = \rho_\text{count}$,
$E_1 = \rho_\text{force}$ and a fallback of 1 (force density), exposing the results
as `grid.lambda_weights` and `grid.rho_lambda`. `RDF` uses
`control_variate_combination` with $E_0$ the backward $g(r)$, $E_1$ the forward
$g(r)$ and a fallback of 0 (backward $g(r)$).

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
