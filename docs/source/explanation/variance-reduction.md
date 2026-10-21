# Variance Reduction and Lambda Weighting

RevelsMD provides two independent estimators for each quantity: counting-based and force-based (see [Force Sampling](force-sampling.md)). Both are unbiased, but their variance differs by region. The lambda method combines them with position-dependent weights that minimise variance, producing a single estimate with lower variance than either, given enough samples to estimate the weights.

## The problem

For RDFs, the two estimators are the forward-integrated $g_\text{fwd}(r)$ and backward-integrated $g_\text{bwd}(r)$. Forward integration is more accurate near $r = 0$; backward integration at large $r$. For 3D densities, the two estimators are the counting-based $\rho_\text{count}(\mathbf{r})$ and the force-based $\rho_\text{force}(\mathbf{r})$. Counting performs better in high-density regions; force-based in low-density regions.

In both cases, the optimal strategy is a position-dependent weighted average favouring whichever estimator is locally more reliable.

## The optimal linear combination

Given two unbiased estimators $A$ and $B$, write their difference as $\delta = B - A$. Any linear combination that stays unbiased has the form

$$\hat{\theta}_\lambda = A + \lambda\,\delta = (1 - \lambda) A + \lambda B$$

so $\lambda$ is the weight on $B$. The variance of this combined estimate is:

$$\operatorname{Var}(\hat{\theta}_\lambda) = \operatorname{Var}(A) + 2\lambda\operatorname{Cov}(A, \delta) + \lambda^2 \operatorname{Var}(\delta)$$

Differentiating with respect to $\lambda$ and setting to zero gives the optimal weight:

$$\lambda^* = -\frac{\operatorname{Cov}(A, \delta)}{\operatorname{Var}(\delta)}$$

This is Eq. 3 of Coles et al. (2021), with $A$ and $B$ playing the roles of their $E_0$ and $E_1$.

## Position-dependent weights

The optimal $\lambda$ varies with position. RevelsMD computes a $\lambda(r)$ profile for RDFs and a three-dimensional $\lambda(\mathbf{r})$ field for densities. The combined estimates are then:

For RDFs, $A = g_\text{bwd}$ and $B = g_\text{fwd}$, so $\lambda(r)$ is the weight on the forward estimator:

$$g_\lambda(r) = (1 - \lambda(r))\, g_\text{bwd}(r) + \lambda(r)\, g_\text{fwd}(r)$$

$\lambda(r)$ approaches 1 at small $r$, where forward integration is accurate, and 0 at large $r$. At $r = 0$ the forward estimator is exact ($g(0) = 0$), so $\lambda(0) = 1$ and $g_\lambda(0) = 0$.

For 3D densities, $A = \rho_\text{count}$ and $B = \rho_\text{force}$, so $\lambda(\mathbf{r})$ is the weight on the force estimator:

$$\rho_\lambda(\mathbf{r}) = (1 - \lambda(\mathbf{r}))\, \rho_\text{count}(\mathbf{r}) + \lambda(\mathbf{r})\, \rho_\text{force}(\mathbf{r})$$

RevelsMD estimates $\lambda$ in two halves of the frames and applies each half's $\lambda$ to the other half (see *Estimating λ from held-out frames* below). `rdf.lam` and `grid.lambda_weights` report the weight the combination puts on $B$ overall; where the two halves' $\lambda$ agree, the combined estimate is exactly the expression above with that weight.

## Estimating variance and covariance

Computing $\lambda^*$ requires $\operatorname{Var}(\delta)$ and $\operatorname{Cov}(A, \delta)$ estimated from the trajectory. Given $N$ independent samples ($A_i$, $B_i$) of the two estimators, the statistics are computed from their spread:

$$\operatorname{Var}(\delta) \approx \frac{1}{N} \sum_{i=1}^N (\delta_i - \bar{\delta})^2$$

$$\operatorname{Cov}(A, \delta) \approx \frac{1}{N} \sum_{i=1}^N (A_i - \bar{A})(\delta_i - \bar{\delta})$$

The two calculations draw their samples differently:

- **RDFs** take each frame's $g_\text{fwd}(r)$ and $g_\text{bwd}(r)$ as one sample, so no blocking parameters apply. The statistics are gathered for every frame; passing `integration='lambda'` to `compute_rdf` or `get_rdf` selects the combination.
- **3D densities** divide each half of the frames into blocks, and each block yields one sample. By default, blocks are contiguous (consecutive frames), controlled by `block_size`. Interleaved blocking (`sections` parameter) can reduce bias from slow drift. Strategy selection is covered in [Block Averaging](block-averaging.md).

### The Welford accumulator

RevelsMD does not store the samples. An online algorithm (`WelfordAccumulator` in `revelsMD/statistics.py`) updates the running mean, variance and covariance with each one, so memory does not grow with the length of the trajectory. Each half of the frames has its own accumulator. Density blocks are weighted by frame count, so unequal sizes are handled correctly; RDF frames have equal weight.

### Block parameters

`block_size` sets frames per block (contiguous); `sections` sets the number of interleaved sections in each half of the frames. More blocks yield better variance estimates but fewer frames per block. Each half needs at least two samples, blocks for a density and frames for an RDF, and fewer raise `ValueError`; forward and backward g(r) need only one frame. For a density, fewer than 10 blocks in either half give a warning, since `rho_lambda` can then be less precise than `rho_force`; in tests, the RDF's lambda g(r) was at least as precise as the better of the forward and backward estimators even at 2 frames per half, so it does not warn.

## Estimating λ from held-out frames

An estimate of $\lambda$ from the same samples whose means it weights is correlated with them, which biases the combination:

$$\operatorname{E}[\hat\lambda\,\bar\delta] = \lambda\operatorname{E}[\bar\delta] + \operatorname{Cov}(\hat\lambda, \bar\delta)$$

The second term is small when there are many independent samples, and large where samples are sparse. For a density, a voxel the histogram over-counted by chance gets a large $\hat\lambda$, and a voxel it never reached gets $\hat\lambda = 0$, so the combination falls below both estimators. On a 7500-frame ab initio trajectory of Li$_6$PS$_5$I on a $200^3$ grid, a $\lambda$ estimated from the same blocks gave a density holding 72 to 82% of the Li atoms, depending on the block size.

RevelsMD therefore splits the frames of each `accumulate()` call into a first and a second half in time. $\lambda_A$ is estimated from the samples in the first half and $\lambda_B$ from those in the second, and each weights the other half's estimators:

$$\hat\theta = w_A\,(\bar A_A + \lambda_B\,\bar\delta_A) + w_B\,(\bar A_B + \lambda_A\,\bar\delta_B)$$

where $w_A$ and $w_B$ are the halves' fractions of the frames, and $\bar A_A$, $\bar\delta_A$ are the first half's means. Each $\lambda$ is independent of the samples it weights, so the combination is unbiased. The frames are split into halves in time, not into alternating samples, because neighbouring frames are correlated. For the same reason, each `accumulate()` call should be long compared with the correlation time: the halves of short calls interleave in time and are no longer independent.

The weight the combination puts on $B$ overall, reported as `rdf.lam` and `grid.lambda_weights`, is $w_A\lambda_B + w_B\lambda_A$.

Each $\lambda$ is estimated from half the samples, and its noise adds to the variance of the combination. For a density with fewer than about 10 blocks in each half, the combination can be less precise than the force estimator alone; with 50 or more, the loss is a few percent.

## Edge cases

Where $\operatorname{Var}(\delta)$ is zero, because $\delta$ is the same in every sample of a half, that half's $\lambda$ is undefined. A variance whose standard deviation is at most $10^{-10}$ of the largest estimate is rounding rather than sampling, and counts as zero. That half then takes a fixed weight: $\lambda = 0$ for RDFs at $r > 0$ (the backward estimator) and $\lambda = 1$ for 3D densities (the force estimator). These are guards against degenerate input such as a single sample; they are not a policy for poorly sampled regions. For that, use `rho_hybrid`, which switches to the counting density below a threshold you choose.

## Cost

For a density, `compute_lambda=True` adds block buffers and two sets of per-block statistics, one for each half of the frames, to accumulation. For an RDF the statistics cost one pass over the bins per frame and are always gathered.

## References

- Coles, S. W., Mangaud, E., Frenkel, D., & Rotenberg, B. (2021). Reduced variance analysis of molecular dynamics simulations by linear combination of estimators. *The Journal of Chemical Physics*, 154(19), 191101. [doi:10.1063/5.0053737](https://doi.org/10.1063/5.0053737)

- Assaraf, R., & Caffarel, M. (1999). Zero-variance principle for Monte Carlo algorithms. *Physical Review Letters*, 83(23), 4682. [doi:10.1103/PhysRevLett.83.4682](https://doi.org/10.1103/PhysRevLett.83.4682)
