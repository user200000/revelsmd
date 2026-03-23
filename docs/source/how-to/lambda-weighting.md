# Use lambda and hybrid density estimators

## Lambda estimator

The lambda estimator combines the counting-based and force-based densities
with per-voxel weights that minimise total variance. Enable it by passing
`compute_lambda=True` to `accumulate()`.

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

The individual estimators remain accessible:

```python
rho_force = grid.rho_force   # force-based density
rho_count = grid.rho_count   # counting-based density
```

At least two blocks must be accumulated before accessing `rho_lambda`.
Variance statistics accumulate across multiple `accumulate()` calls, so
you can split a long trajectory across several calls. Calling
`accumulate(..., compute_lambda=False)` clears any existing statistics.

## Hybrid estimator

The hybrid estimator switches between the force-based and counting-based
density on a per-voxel basis according to a threshold on the local
counting density. It does not require `compute_lambda=True`.

```python
grid = DensityGrid(traj, density_type='number', nbins=100)
grid.accumulate(traj, atom_names='Li')

rho = grid.rho_hybrid(threshold=0.01)
```

Voxels where `rho_count >= threshold` use `rho_force`; voxels below the
threshold use `rho_count`. This removes spurious negative artefacts from
the force estimator in poorly sampled regions while preserving its higher
resolution elsewhere.

Choose the threshold by inspecting `grid.rho_count` — a value near the
noise floor of the counting density is typical.
