# 3D densities

Compute a 3D lithium density for Li6PS5I using VASP
trajectory data. The example file `Li6PS5I_run1_vasprun.xml` is available
in the `examples/` directory of the
[repository](https://github.com/user200000/revelsmd).

## Load the trajectory

```python
from revelsMD.trajectories import VaspTrajectory

traj = VaspTrajectory(
    'examples/Li6PS5I_run1_vasprun.xml',
    temperature=500.0,
)
```

## Compute the density

The quickest route is `compute_density`, which accumulates all frames and
returns a `DensityGrid` in one call:

```python
from revelsMD.density import compute_density

grid = compute_density(
    traj,
    'Li',
    density_type='number',
    nbins=200,
    compute_lambda=True,
)
```

Species are identified by name. `nbins=200` sets a uniform 200x200x200
voxel grid. `compute_lambda=True` enables the variance-minimised estimator.

## Access the results

```python
rho_count  = grid.rho_count   # counting (histogram) density
rho_force  = grid.rho_force   # force-based density
rho_lambda = grid.rho_lambda  # variance-minimised density
```

All three are 3D NumPy arrays of shape `(nbinsx, nbinsy, nbinsz)`,
computed lazily on first access.

## Visualise a 2D slice

```python
import matplotlib.pyplot as plt

z_frac = 0.375
z_idx = int(z_frac * grid.nbinsz)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

for ax, rho, title in zip(
    axes,
    [grid.rho_count, grid.rho_force, grid.rho_lambda],
    ['Count', 'Force', 'Lambda'],
):
    ax.imshow(rho[:, :, z_idx].T, origin='lower')
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()
```

```{image} /_static/images/tutorial_density.png
:alt: Comparison of count, force, and lambda density estimates for Li in Li6PS5I
:width: 100%
```

The count density is noisy; the force-based density resolves the Li cage
structure with far less variance; the lambda estimate combines both for
the optimal result.

## Hybrid density

The force estimator can produce spurious negative values in poorly sampled
voxels. `rho_hybrid` switches to the counting estimate below a density
threshold:

```python
rho = grid.rho_hybrid(threshold=0.5)
```

Voxels with `rho_count >= threshold` use the force-based value; voxels below
the threshold use the counting value.

## Using DensityGrid directly

`compute_density` is a convenience wrapper. For separate control over grid
construction and accumulation, use `DensityGrid`:

```python
from revelsMD.density import DensityGrid

grid = DensityGrid(traj, density_type='number', nbins=(200, 200, 200))
grid.accumulate(
    traj,
    atom_names='Li',
    compute_lambda=True,
)
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

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `atom_names` | required | Species to include |
| `density_type` | `'number'` | `'number'`, `'charge'`, or `'polarisation'` |
| `nbins` | `100` | Voxels per axis; int or `(nx, ny, nz)` tuple |
| `kernel` | `'triangular'` | Deposition kernel: `'triangular'` or `'box'` |
| `start` | `0` | First frame index |
| `stop` | `None` | Stop frame index; `None` = all frames |
| `period` | `1` | Frame stride |

## Lambda and hybrid estimators

Lambda and hybrid estimators are covered in the [lambda weighting how-to](../how-to/lambda-weighting.md).
