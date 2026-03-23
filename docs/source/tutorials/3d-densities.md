# Computing 3D densities

This tutorial computes a 3D number density for a binary Lennard-Jones fluid
using the example data in `examples/example_2_LJ_3D/`.

## Load the trajectory

```python
from revelsMD.trajectories import LammpsTrajectory

traj = LammpsTrajectory(
    'examples/example_2_LJ_3D/dump.nh.lammps',
    'examples/example_2_LJ_3D/data.fin.nh.data',
    temperature=1.35,
    units='lj',
)
```

## Compute the density

The quickest route is `compute_density`, which accumulates all frames and
returns a `DensityGrid` in one call:

```python
from revelsMD.density import compute_density

grid = compute_density(traj, '1', density_type='number', nbins=30)
```

Species are identified by LAMMPS atom type number, passed as a string.
`nbins=30` sets a uniform 30x30x30 voxel grid. `density_type='number'` is the
default and can be omitted; the alternatives are `'charge'` and
`'polarisation'`.

## Access the results

```python
rho_force = grid.rho_force  # force-based density
rho_count = grid.rho_count  # counting (histogram) density
```

Both are three-dimensional NumPy arrays of shape `(nbinsx, nbinsy, nbinsz)`.
They are computed lazily on first access.

## Visualise a 2D slice

```python
import matplotlib.pyplot as plt

mid_z = grid.nbinsz // 2

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].imshow(grid.rho_force[:, :, mid_z].T, origin='lower')
axes[0].set_title('Force-based')

axes[1].imshow(grid.rho_count[:, :, mid_z].T, origin='lower')
axes[1].set_title('Count-based')

plt.tight_layout()
plt.show()
```

The `.T` transpose is needed because `imshow` treats the first axis as rows.
`origin='lower'` places the origin at the bottom-left.

## Hybrid density

The force estimator can produce spurious negative values in poorly sampled
voxels. `rho_hybrid` switches to the counting estimate below a density
threshold:

```python
rho = grid.rho_hybrid(threshold=0.5)
```

Voxels with `rho_count >= threshold` use the force-based value; voxels below
the threshold use the counting value. The threshold is in the same units as
`rho_count`.

## Using DensityGrid directly

`compute_density` is a convenience wrapper. For separate control over grid
construction and accumulation, use `DensityGrid`:

```python
from revelsMD.density import DensityGrid

grid = DensityGrid(traj, density_type='number', nbins=(40, 40, 20))
grid.accumulate(traj, atom_names='1', start=10)
```

Per-axis bin counts are set with a `(nbinsx, nbinsy, nbinsz)` tuple.
`accumulate()` accepts `start`, `stop`, and `period` with the same semantics
as Python slice indices.

## Grid metadata

```python
grid.nbinsx, grid.nbinsy, grid.nbinsz  # voxels per axis
grid.lx, grid.ly, grid.lz             # fractional voxel sizes (1/nbins)
grid.count                             # number of accumulated frames
```

## Key parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `atom_names` | required | LAMMPS atom type(s) to include |
| `density_type` | `'number'` | `'number'`, `'charge'`, or `'polarisation'` |
| `nbins` | `100` | Voxels per axis; int or `(nx, ny, nz)` tuple |
| `kernel` | `'triangular'` | Deposition kernel: `'triangular'` or `'box'` |
| `start` | `0` | First frame index |
| `stop` | `None` | Stop frame index; `None` = all frames |
| `period` | `1` | Frame stride |

## Lambda and hybrid estimators

For variance-minimised densities, see [lambda and hybrid estimators](../how-to/lambda-weighting.md).
The `rho_lambda` property is available after passing `compute_lambda=True` to
`compute_density` or `accumulate()`.
