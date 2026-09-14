# Variance Reduction and Lambda Weighting

RevelsMD provides two independent estimators for each quantity: counting-based and force-based (see [Force Sampling](force-sampling.md)). Both are unbiased, but their variance differs by region. The lambda method combines them with position-dependent weights that minimise variance, producing a single estimate that outperforms either alone.

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

$\lambda(r)$ approaches 1 at small $r$, where forward integration is accurate, and 0 at large $r$.

For 3D densities, $A = \rho_\text{count}$ and $B = \rho_\text{force}$, so $\lambda(\mathbf{r})$ is the weight on the force estimator:

$$\rho_\lambda(\mathbf{r}) = (1 - \lambda(\mathbf{r}))\, \rho_\text{count}(\mathbf{r}) + \lambda(\mathbf{r})\, \rho_\text{force}(\mathbf{r})$$

These are the weights exposed as `rdf.lam` and `grid.lambda_weights`.

## Estimating variance and covariance

Computing $\lambda^*$ requires $\operatorname{Var}(\delta)$ and $\operatorname{Cov}(A, \delta)$ estimated from the trajectory. Given $N$ independent samples ($A_i$, $B_i$) of the two estimators, the statistics are computed from their spread:

$$\operatorname{Var}(\delta) \approx \frac{1}{N} \sum_{i=1}^N (\delta_i - \bar{\delta})^2$$

$$\operatorname{Cov}(A, \delta) \approx \frac{1}{N} \sum_{i=1}^N (A_i - \bar{A})(\delta_i - \bar{\delta})$$

The two calculations draw their samples differently:

- **RDFs** keep the per-frame $g_\text{fwd}(r)$ and $g_\text{bwd}(r)$ profiles, so each frame is one sample and no blocking parameters apply. Lambda estimation is enabled by passing `integration='lambda'` to `compute_rdf` or `get_rdf`.
- **3D densities** divide frames into $N$ blocks, and each block yields one sample. By default, blocks are contiguous (consecutive frames), controlled by `block_size`. Interleaved blocking (`sections` parameter) can reduce bias from slow drift. Strategy selection is covered in [Block Averaging](block-averaging.md).

### The Welford accumulator

For 3D density fields, where storing all block densities would be prohibitive, RevelsMD uses an online algorithm (`WelfordAccumulator3D` in `revelsMD/statistics.py`) that updates running mean, variance, and covariance in a single pass. Blocks are weighted by frame count, so unequal sizes are handled correctly.

### Block parameters

`block_size` sets frames per block (contiguous); `sections` sets the number of interleaved blocks. More blocks yield better variance estimates but fewer frames per block. At least two blocks are required.

## Edge cases

Where $\operatorname{Var}(\delta)$ is zero — because both estimators agree exactly across every sample — $\lambda$ is undefined. RevelsMD then reports a fixed weight: $\lambda = 0$ for RDFs (the backward estimator) and $\lambda = 1$ for 3D densities (the force estimator). Non-finite values from numerical issues receive the same fixed weight. These are guards against degenerate input such as a single sample; they are not a policy for poorly sampled regions. For that, use `rho_hybrid`, which switches to the counting density below a threshold you choose.

## When to use lambda estimation

Lambda estimation is most beneficial when:

- Trajectories are short and variance is the dominant source of error.
- The system is inhomogeneous and different regions have very different sampling quality.
- Low-density or high-barrier regions are of particular interest.

For well-converged trajectories where counting already meets accuracy requirements, the extra computation is unnecessary.

## References

- Coles, S. W., Mangaud, E., Frenkel, D., & Rotenberg, B. (2021). Reduced variance analysis of molecular dynamics simulations by linear combination of estimators. *The Journal of Chemical Physics*, 154(19), 191101. [doi:10.1063/5.0053737](https://doi.org/10.1063/5.0053737)

- Assaraf, R., & Caffarel, M. (1999). Zero-variance principle for Monte Carlo algorithms. *Physical Review Letters*, 83(23), 4682. [doi:10.1103/PhysRevLett.83.4682](https://doi.org/10.1103/PhysRevLett.83.4682)
