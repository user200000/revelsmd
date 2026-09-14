# Use lambda and hybrid density estimators

## Lambda estimator

The lambda estimator combines counting and force densities with per-voxel
weights that minimise variance. Pass `compute_lambda=True` to `accumulate()`.

```python
from revelsMD.density import DensityGrid

grid = DensityGrid(traj, density_type='number', nbins=100)
grid.accumulate(
    traj,
    atom_names='Li',
    compute_lambda=True,
    blocking='contiguous',
    block_size=50,          # frames per block for variance estimation
)

rho = grid.rho_lambda       # variance-minimised density field
lam = grid.lambda_weights   # per-voxel weights (0 = count, 1 = force)
```

Individual estimators remain accessible:

```python
rho_force = grid.rho_force   # force-based density
rho_count = grid.rho_count   # counting-based density
```

At least two blocks are needed before accessing `rho_lambda`. Statistics
accumulate across multiple `accumulate()` calls, so you can split a long
trajectory across several calls. Calling
`accumulate(..., compute_lambda=False)` clears existing statistics.

Voxels where every block gave identical force and counting densities have
no variance to weigh and report a weight of 1 (force density). This is a
guard for degenerate input, not a treatment for poorly sampled regions; use
the hybrid estimator below for that.

## Hybrid estimator

The hybrid estimator switches between force and counting density per voxel
based on a counting-density threshold. It does not require
`compute_lambda=True`.

```python
grid = DensityGrid(traj, density_type='number', nbins=100)
grid.accumulate(traj, atom_names='Li')

rho = grid.rho_hybrid(threshold=0.01)
```

Voxels where `rho_count >= threshold` use `rho_force`; those below use
`rho_count`. This removes negative artefacts from the force estimator in
poorly sampled regions while preserving its resolution elsewhere.

Choose the threshold by inspecting `grid.rho_count` -- a value near its
noise floor is typical.
