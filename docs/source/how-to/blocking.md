# Configure blocking strategies

Blocking divides trajectory frames into groups (blocks) used to estimate
variance for the lambda estimator. The `blocking` parameter of
`accumulate()` accepts two strategies.

## Contiguous blocking (default)

Each block is a consecutive slice of frames. This works with all
trajectory backends, including those that only support sequential access.

```python
from revelsMD.density import DensityGrid

grid = DensityGrid(traj, density_type='number', nbins=100)
grid.accumulate(
    traj,
    atom_names='O',
    compute_lambda=True,
    blocking='contiguous',
    block_size=100,   # 100 frames per block
)
```

If `block_size` is not set it defaults to one frame per block. The final
block may be smaller than `block_size` if the total frame count is not
evenly divisible.

## Interleaved blocking

Section *k* receives every *k*-th frame: section 0 gets frames
0, N, 2N, …; section 1 gets frames 1, N+1, 2N+1, …; and so on.
This pattern spreads each block evenly across the trajectory, which can
improve variance estimates when slow relaxations are present.

Interleaved blocking requires random frame access, so the trajectory
backend must implement `get_frame()`. `NumpyTrajectory` and file-backed
formats that support random access (e.g. `LammpsTrajectory`) satisfy this
requirement.

```python
grid = DensityGrid(traj, density_type='number', nbins=100)
grid.accumulate(
    traj,
    atom_names='O',
    compute_lambda=True,
    blocking='interleaved',
    sections=10,   # 10 interleaved sections
)
```

If `sections` is not set it defaults to one section per frame.

## Choosing block size

More blocks give a better variance estimate but fewer frames per block.
As a rule of thumb, use 5–20 blocks and ensure each block contains enough
frames to be statistically meaningful. The total number of frames divided
by the block count gives the frames per block.
