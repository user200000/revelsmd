# Block averaging and error estimation

The force-sampling estimator produces two density fields per run: a counting-based
estimate $\rho_\text{count}$ and a force-based estimate $\rho_\text{force}$. Their
optimal linear combination is

$$
\rho_\lambda = (1 - \lambda)\,\rho_\text{count} + \lambda\,\rho_\text{force}
$$

where the per-voxel weight $\lambda$ minimises the variance of the combined
estimator. Computing $\lambda$ requires variance and covariance statistics across
independent samples of $(\rho_\text{force} - \rho_\text{count})$ and
$\rho_\text{force}$. Block averaging provides those independent samples from a
single trajectory by dividing the frames into groups.

## Contiguous vs interleaved blocking

RevelsMD supports two strategies, selected via the `blocking` parameter of
`DensityGrid.accumulate()`.

### Contiguous blocks

Frames are divided into sequential slices. With `blocking='contiguous'` and
`block_size=500`, the first 500 frames form block 0, the next 500 form block 1,
and so on. The final block may contain fewer frames if the total is not evenly
divisible.

Contiguous blocking works with any trajectory backend because it only requires
sequential frame access via `iter_frames()`. It is the default strategy.

```python
grid.accumulate(traj, atom_names='O', compute_lambda=True,
                blocking='contiguous', block_size=500)
```

### Interleaved blocks

Section $k$ receives every $n$-th frame starting at offset $k$. With
`blocking='interleaved'` and `sections=4`, section 0 gets frames
$[0, 4, 8, \ldots]$, section 1 gets $[1, 5, 9, \ldots]$, and so on.

Interleaved blocking requires random frame access via `get_frame()`, so it is
only available for trajectory backends that support it. It can give better
statistical independence between sections when the trajectory is not well
equilibrated throughout: each section samples the entire time range rather
than a single contiguous window.

```python
grid.accumulate(traj, atom_names='O', compute_lambda=True,
                blocking='interleaved', sections=4)
```

### Choosing a strategy

Use contiguous blocking unless you have a specific reason to prefer interleaved.
It is simpler, works with all backends, and is adequate when the trajectory is
reasonably well converged. Interleaved blocking is useful when you want each
block to represent the full ensemble rather than a temporal slice, but it requires
a backend that supports `get_frame()`.

At least two blocks must be accumulated (across all `accumulate()` calls on the
same `DensityGrid`) before `rho_lambda` can be computed. Attempting to access it
with fewer than two blocks raises `ValueError`.

## How blocks feed the Welford accumulator

Each block is processed independently. After all frames in a block are deposited
into temporary accumulators, the block-level densities are normalised to give
per-block estimates of $\rho_\text{force}$ and $\rho_\text{count}$. The difference
$\delta = \rho_\text{force} - \rho_\text{count}$ is then passed to the
`WelfordAccumulator3D` along with $\rho_\text{force}$ and the block frame count
as its weight.

Accumulation is additive across multiple `accumulate()` calls: calling
`accumulate(..., compute_lambda=True)` a second time (e.g. with a different
trajectory) adds more blocks to the same accumulator. This allows $\lambda$ to be
estimated from multiple independent trajectories.

Calling `accumulate(..., compute_lambda=False)` clears any existing lambda
statistics and raises a `UserWarning`.

## WelfordAccumulator3D internals

`WelfordAccumulator3D` implements a weighted variant of Welford's online algorithm
to compute running mean, variance, and covariance without storing all samples.

For each new block $(k)$ with weight $w_k$, the accumulator updates:

- Running weighted mean of $\delta$
- Running weighted mean of $\rho_\text{force}$
- Weighted sum of squared deviations $M_2(\delta)$, from which variance is derived
- Weighted cross-deviation $C(\delta, \rho_\text{force})$, from which covariance is derived

After all blocks, `finalise()` returns:

$$
\text{Var}(\delta) = \frac{M_2(\delta)}{\sum_k w_k}, \qquad
\text{Cov}(\delta, \rho_\text{force}) = \frac{C(\delta, \rho_\text{force})}{\sum_k w_k}
$$

These are population (not sample) estimates — the denominator is the total weight,
not the total weight minus one. `finalise()` raises `ValueError` if fewer than two
blocks have been accumulated.

### Lambda weight computation

`compute_lambda_weights(variance, covariance)` computes

$$
\lambda = \frac{\text{Cov}(\delta, \rho_\text{force})}{\text{Var}(\delta)}
$$

with safe handling for voxels where variance is zero (lambda is set to 0 there,
corresponding to pure counting density). Non-finite values are also replaced with
zero.

### Estimator combination

`combine_estimators(rho_count, rho_force, lambda_weights)` evaluates
$(1 - \lambda)\,\rho_\text{count} + \lambda\,\rho_\text{force}$ and sanitises
any NaN or Inf values in the output to 0.

## Note on user-facing access

Error bars derived from block variance are not currently exposed as public outputs.
The variance and covariance statistics are used internally to compute per-voxel
$\lambda$ weights. The combined density is available as `grid.rho_lambda` once at
least two blocks have been accumulated.
