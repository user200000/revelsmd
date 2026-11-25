# Use the lambda density estimator

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
`rho_lambda` puts on the force density overall. Each half needs at least four
blocks before accessing `rho_lambda`; with few blocks, the noise in lambda can
make `rho_lambda` less precise than `rho_force`. Statistics accumulate across multiple `accumulate()` calls, such as one
per trajectory of the same system. Each call's frames are split into their own
two halves, so the result depends on how frames are divided between calls, and
each call should be long compared with the correlation time of the trajectory:
with short calls, the two halves interleave in time and are no longer
independent. A call of one frame adds only to the first half. Calling
`accumulate(..., compute_lambda=False)` clears existing statistics, and
`accumulate(..., compute_lambda=True)` on a grid holding frames accumulated
without them raises `ValueError`.

Voxels where every block of a half gave the same difference between the force
and counting densities, to within rounding, have no variance to weigh and take
a weight of 1 (force density) in that half. This is a guard for degenerate
input such as identical frames.

In voxels where a half's blocks have no counts, that half's lambda is 0 unless
the guard above applies. In voxels that no sample reached in either half,
`rho_lambda` is therefore zero, the counting density. The force density there
is in general not zero, and it can be negative.
