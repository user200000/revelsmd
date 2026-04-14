# Variance Reduction and Lambda Weighting

RevelsMD provides two independent estimators for each structural quantity it computes: a counting-based estimate and a force-based estimate (see [Force Sampling](force-sampling.md)). Both are unbiased, but their variance characteristics differ by region. The lambda method combines them with position-dependent weights chosen to minimise total variance, producing a single estimate that outperforms either alone.

## The problem

For RDFs, the two estimators are the forward-integrated $g_\text{fwd}(r)$ and backward-integrated $g_\text{bwd}(r)$. Forward integration is more accurate near $r = 0$; backward integration at large $r$. For 3D densities, the two estimators are the counting-based $\rho_\text{count}(\mathbf{r})$ and the force-based $\rho_\text{force}(\mathbf{r})$. Counting performs better in high-density regions; force-based in low-density regions.

In both cases, the optimal strategy is to take a weighted average that varies with position — using whichever estimator is more reliable locally.

## The optimal linear combination

Given two unbiased estimators $A$ and $B$ for the same quantity, the minimum-variance linear combination is:

$$\hat{\theta}_\lambda = (1 - \lambda) A + \lambda B$$

The variance of this combined estimate is:

$$\operatorname{Var}(\hat{\theta}_\lambda) = (1-\lambda)^2 \operatorname{Var}(A) + \lambda^2 \operatorname{Var}(B) + 2\lambda(1-\lambda)\operatorname{Cov}(A, B)$$

Differentiating with respect to $\lambda$ and setting to zero gives the optimal weight:

$$\lambda^* = \frac{\operatorname{Cov}(\delta, B)}{\operatorname{Var}(\delta)}$$

where $\delta = B - A$.

## Position-dependent weights

The optimal $\lambda$ is a function of position, not a scalar. RevelsMD computes a full $\lambda(r)$ profile for RDFs and a three-dimensional $\lambda(\mathbf{r})$ field for density calculations. The combined estimates are then:

For RDFs:

$$g_\lambda(r) = (1 - \lambda(r))\, g_\text{fwd}(r) + \lambda(r)\, g_\text{bwd}(r)$$

For 3D densities:

$$\rho_\lambda(\mathbf{r}) = (1 - \lambda(\mathbf{r}))\, \rho_\text{count}(\mathbf{r}) + \lambda(\mathbf{r})\, \rho_\text{force}(\mathbf{r})$$

## Estimating variance and covariance

Computing $\lambda^*$ requires estimates of $\operatorname{Var}(\delta)$ and $\operatorname{Cov}(\delta, B)$ from the trajectory data itself. RevelsMD obtains these by dividing trajectory frames into $N$ blocks. Both contiguous blocking (the default) and interleaved blocking are supported. Each block yields an independent pair of estimates $A_i$ and $B_i$, and the statistics are computed from the spread across blocks:

$$\operatorname{Var}(\delta) \approx \frac{1}{N} \sum_{i=1}^N (\delta_i - \bar{\delta})^2$$

$$\operatorname{Cov}(\delta, B) \approx \frac{1}{N} \sum_{i=1}^N (\delta_i - \bar{\delta})(B_i - \bar{B})$$

By default blocks are contiguous (consecutive frames), controlled by the `block_size` parameter. Interleaved blocking, controlled by the `sections` parameter, is also available and can reduce bias from slow drift. Block strategy selection is covered in [Block Averaging](block-averaging.md).

### The Welford accumulator

For 3D density fields, where storing all block densities would be prohibitive, RevelsMD uses an online algorithm (`WelfordAccumulator3D` in `revelsMD/statistics.py`) that updates the running mean, variance, and covariance in a single pass as each block is processed. Each block can be weighted by the number of frames it contains, so blocks of unequal size are handled correctly.

### Block parameters

For contiguous blocking, the `block_size` parameter (passed to `compute_density` or `accumulate`) sets the number of frames per block. For interleaved blocking, the `sections` parameter controls the number of interleaved blocks. In either case, more blocks yield more accurate variance estimates but fewer frames per block; fewer blocks give cruder variance estimates but more frames per block. At least two blocks are required.

For RDFs, lambda estimation is enabled by passing `integration='lambda'` to `compute_rdf` or `get_rdf`.

## Edge cases

Where $\operatorname{Var}(\delta)$ is zero — because both estimators agree exactly across all blocks — the optimal $\lambda$ is undefined. RevelsMD defaults to $\lambda = 0$ in these regions (use estimator $A$). Non-finite $\lambda$ values arising from other numerical issues are also replaced with zero. These defaults are conservative: they fall back on the counting-based estimator rather than introducing spurious values.

## When to use lambda estimation

Lambda estimation is most beneficial when:

- Trajectories are short and variance is the dominant source of error.
- The system is inhomogeneous and different regions have very different sampling quality.
- Low-density or high-barrier regions are of particular interest.

For long, well-converged trajectories where the counting estimator already meets accuracy requirements, the extra computation is unnecessary.

## References

- Coles, S. W., Mangaud, E., Frenkel, D., & Rotenberg, B. (2021). Reduced variance analysis of molecular dynamics simulations by linear combination of estimators. *The Journal of Chemical Physics*, 154(19), 191101. [doi:10.1063/5.0053737](https://doi.org/10.1063/5.0053737)

- Assaraf, R., & Caffarel, M. (1999). Zero-variance principle for Monte Carlo algorithms. *Physical Review Letters*, 83(23), 4682. [doi:10.1103/PhysRevLett.83.4682](https://doi.org/10.1103/PhysRevLett.83.4682)
