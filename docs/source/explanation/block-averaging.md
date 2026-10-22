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

$\lambda$ is estimated separately in the first and second halves in time of the
frames of each `accumulate()` call, and each half's $\lambda$ weights the other
half's densities, so it is never estimated from the blocks it weights (see
[Variance Reduction](variance-reduction.md)). Blocks are formed within each half.

## Contiguous vs interleaved blocking

RevelsMD supports two strategies, selected via the `blocking` parameter of
`DensityGrid.accumulate()`.

### Contiguous blocks

Each half of the frames is divided into sequential slices. With
`blocking='contiguous'` and `block_size=500`, the first 500 frames of a half form
its first block, the next 500 its second, and so on. The final block of a half
may contain fewer frames if the half is not evenly divisible.

Contiguous blocking works with any backend (it only requires sequential access
via `iter_frames()`) and is the default.

```python
grid.accumulate(traj, atom_names='O', compute_lambda=True,
                blocking='contiguous', block_size=500)
```

### Interleaved blocks

Within each half of the frames, section $k$ receives every $n$-th frame
starting at offset $k$. With `blocking='interleaved'` and `sections=4`, section 0
of the first half gets its frames $[0, 4, 8, \ldots]$, section 1 gets
$[1, 5, 9, \ldots]$, and so on; the second half has its own 4 sections.

Interleaved blocking reads each section in one pass over the trajectory,
skipping the frames of other sections. Neighbouring frames fall in different
sections, so the sections are correlated with each other, and the spread across
them understates the variation that the variances and covariances for $\lambda$
are meant to measure.

```python
grid.accumulate(traj, atom_names='O', compute_lambda=True,
                blocking='interleaved', sections=4)
```

### Choosing a strategy

Use contiguous blocking. Blocks that are each long compared with the
correlation time of the trajectory are close to independent samples, and
contiguous blocking reads frames in order.

Each half needs at least two blocks, across all `accumulate()` calls on the same
`DensityGrid`, before `rho_lambda` can be computed; fewer raise `ValueError`.
With few blocks in a half, that half's $\lambda$ is noisy, and $\rho_\lambda$
can be less precise than $\rho_\text{force}$ (see
[Variance Reduction](variance-reduction.md)).

## How blocks feed the Welford accumulator

Each block is processed independently. After all frames in a block are deposited,
the block-level densities are normalised to per-block $\rho_\text{force}$ and
$\rho_\text{count}$. The difference
$\delta = \rho_\text{force} - \rho_\text{count}$ is then passed to the
`WelfordAccumulator` along with $\rho_\text{count}$ and the block frame count
as its weight. Each half of the frames has its own accumulator.

Accumulation is additive across `accumulate()` calls: a second call (e.g. with a
different trajectory) adds the blocks of its first half to the first accumulator
and those of its second half to the second, allowing $\lambda$ estimation from
multiple trajectories.

Calling `accumulate(..., compute_lambda=False)` clears any existing lambda
statistics, and issues a `UserWarning` if there were any. Calling
`accumulate(..., compute_lambda=True)` on a grid holding frames accumulated
without lambda statistics raises `ValueError`: each half's $\lambda$ weights the
other half's densities, which must together be the grid's densities.

## WelfordAccumulator internals

`WelfordAccumulator` implements a weighted Welford algorithm, computing running
mean, variance, and covariance without storing all samples. It works on arrays
of any shape and takes $\delta$ and a second quantity; `DensityGrid` passes
$\rho_\text{count}$ as that quantity, so the terms below are in density terms.

For each new block $(k)$ with weight $w_k$, the accumulator updates:

- Running weighted mean of $\delta$
- Running weighted mean of $\rho_\text{count}$
- Weighted sum of squared deviations $M_2(\delta)$, from which variance is derived
- Weighted cross-deviation $C(\delta, \rho_\text{count})$, from which covariance is derived

After all blocks, `finalise()` returns:

$$
\text{Var}(\delta) = \frac{M_2(\delta)}{\sum_k w_k}, \qquad
\text{Cov}(\delta, \rho_\text{count}) = \frac{C(\delta, \rho_\text{count})}{\sum_k w_k}
$$

These are population (not sample) estimates — the denominator is total weight,
not total weight minus one. `finalise()` raises `ValueError` if fewer than two
blocks have been accumulated. The accumulator's weighted means of $\delta$ and
$\rho_\text{count}$ are its half's densities, since the block densities are
linear in the accumulated arrays.

### Lambda weights and the combined density

With $E_0 = \rho_\text{count}$ and $E_1 = \rho_\text{force}$, each half $h$ of
the frames gives a weight on the force density,

$$
\lambda_h = -\frac{\text{Cov}_h(\rho_\text{count}, \delta)}{\text{Var}_h(\delta)},
$$

and `cross_fitted_combination` applies each half's $\lambda$ to the other half:

$$
\rho_\lambda = w_A\,(\bar\rho_{\text{count},A} + \lambda_B\,\bar\delta_A)
              + w_B\,(\bar\rho_{\text{count},B} + \lambda_A\,\bar\delta_B),
$$

where $w_A$ and $w_B$ are the halves' fractions of the frames. `lambda_weights`
is $w_A\lambda_B + w_B\lambda_A$, the weight $\rho_\lambda$ puts on
$\rho_\text{force}$ overall; where $\lambda_A = \lambda_B$,
$\rho_\lambda = (1 - \lambda)\,\rho_\text{count} + \lambda\,\rho_\text{force}$
exactly. A variance is negligible where the standard deviation of $\delta$
across a half's blocks is at most $10^{-10}$ of the largest estimate anywhere in
the grid: that is rounding, not sampling, and dividing by it would give an
arbitrarily large $\lambda$. In voxels with negligible variance, where every
block of a half gave the same $\delta$ to within rounding, that half's
$\lambda$ is 1 (force density).

## Note on user-facing access

Block variance is not exposed as error bars. Variance and covariance are used
internally to compute per-voxel $\lambda$ weights. The combined density is
available as `grid.rho_lambda` once each half of the frames has at least two
blocks.
