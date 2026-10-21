# Configure blocking strategies

Blocking divides frames into groups (blocks) for the variance estimates behind
a density's lambda weights. `DensityGrid.accumulate()` splits each call's frames
into two halves in time, estimates lambda in each half and applies it to the
other, and forms blocks within each half. The `blocking` parameter accepts two
strategies.

## Contiguous blocking (default)

Each block is a consecutive slice of a half's frames. This works with all
backends, including those that only support sequential access.

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

If `block_size` is not set, it defaults to one frame per block. The final
block of a half may be shorter if the half is not evenly divisible.

## Interleaved blocking

With *N* sections, each half of the frames has *N* sections, and section *k*
receives every *N*-th frame of the half from its frame *k*: section 0 gets the
half's frames 0, N, 2N, …; section 1 gets frames 1, N+1, 2N+1, …; and so on.
Neighbouring frames fall in different sections, so the sections are correlated
with each other; contiguous blocks, each long compared with the correlation
time, are closer to independent samples.

Each section is read in one pass over the trajectory, skipping the frames of
other sections.

```python
grid = DensityGrid(traj, density_type='number', nbins=100)
grid.accumulate(
    traj,
    atom_names='O',
    compute_lambda=True,
    blocking='interleaved',
    sections=10,   # 10 interleaved sections in each half
)
```

If `sections` is not set it defaults to one section per frame. `sections`
larger than the number of frames in a half raises `ValueError`.

## Choosing block size

More blocks give a better variance estimate but fewer frames per block. Each
half needs at least 2 blocks, across all `accumulate()` calls. With few blocks
per half, lambda is noisy and `rho_lambda` can be less precise than
`rho_force`; how many blocks are enough depends on the system (see
[Variance Reduction](../explanation/variance-reduction.md)).
