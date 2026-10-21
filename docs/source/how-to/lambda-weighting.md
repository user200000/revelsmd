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

Lambda is estimated in each half in time of the frames of every
`accumulate()` call and applied to the other half's densities, so it is never
estimated from the blocks it weights; `lambda_weights` is the weight
`rho_lambda` puts on the force density overall. Each half needs at least two
blocks before accessing `rho_lambda`, and fewer than 10 in either half gives a
warning. Statistics accumulate across multiple `accumulate()` calls, such as one
per trajectory of the same system. Each call's frames are split into their own
two halves, so the result depends on how frames are divided between calls, and
each call should be long compared with the correlation time of the trajectory:
with short calls, the two halves interleave in time and are no longer
independent. A call of one frame adds only to the first half. Calling
`accumulate(..., compute_lambda=False)` clears existing statistics.

Voxels where every block of a half gave the same difference between the force
and counting densities, to within rounding, have no variance to weigh and take
a weight of 1 (force density) in that half. This is a
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
