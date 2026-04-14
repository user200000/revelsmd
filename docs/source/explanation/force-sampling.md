# Force Sampling

Force sampling is an alternative to histogram counting for computing radial distribution functions (RDFs) and 3D density fields from molecular dynamics trajectories. Instead of counting how often particles visit a region, it uses the forces acting on particles to infer the underlying probability density. The result is the same quantity — $g(r)$ or $\rho(\mathbf{r})$ — but with substantially lower variance, particularly in low-density or high-energy regions where histogram counts are sparse.

RevelsMD implements force sampling for both RDFs and 3D density fields, with an optional third stage that combines force-based and counting-based estimates to minimise total variance (see [Variance Reduction and Lambda Weighting](variance-reduction.md)).

## Why histograms struggle

Histogram estimators for $g(r)$ and $\rho(\mathbf{r})$ converge as $1/\sqrt{N_\text{bin}}$, where $N_\text{bin}$ is the number of particle visits to a bin. In low-density regions — near energy barriers, in solvent-excluded volumes, or at the tails of distribution functions — bins accumulate counts slowly regardless of total trajectory length. The estimator is unbiased but high-variance, and the variance is worst precisely where the quantity of interest is often most physically significant.

## The force-based alternative

The potential of mean force $W(\mathbf{r})$ is related to the equilibrium density by:

$$W(\mathbf{r}) = -k_B T \ln \rho(\mathbf{r})$$

Taking the gradient:

$$\langle \mathbf{F}(\mathbf{r}) \rangle = -\nabla W(\mathbf{r}) = k_B T \nabla \ln \rho(\mathbf{r})$$

The mean force field thus encodes the same information as the density, but estimated from forces rather than counts. Forces are available at every particle at every frame, so the effective sample size is much larger — in principle, every particle contributes information about every bin it influences.

## RDF from forces

Borgis et al. (2013) showed that the force density $F(r)$ — the mean radial force between particle pairs, accumulated over all pair distances — can be integrated directly to yield $g(r)$ without passing through the potential of mean force. The key relation (Borgis et al. Eq. 9) is:

$$\rho_b\, h_{ab}(r) = -\beta \int_r^\infty F(r')\, dr'$$

where $h_{ab} = g_{ab} - 1$ and $\rho_b$ is the bulk number density. This gives $g(r)$ by linear integration of the force density, not by exponentiating a PMF.

In practice, this can be integrated in either direction, yielding two complementary estimators (Coles et al. 2021, Eqs. 6--7):

**Forward integration** (using $g(0) = 0$): accumulates force contributions from all pairs separated by a distance smaller than $r$.

**Backward integration** (using $g(\infty) = 1$): accumulates from all pairs separated by a distance larger than $r$, anchored at $g = 1$ at large separations.

Both are exact in principle but accumulate numerical noise in different regions: forward integration is more accurate near $r = 0$, backward integration at large $r$. The [lambda method](variance-reduction.md) finds the optimal position-dependent combination of the two.

## 3D density from forces

For three-dimensional density fields, the analogous relationship is:

$$\nabla \rho(\mathbf{r}) = -\beta \rho(\mathbf{r}) \langle \mathbf{F}(\mathbf{r}) \rangle$$

This is solved efficiently in Fourier space. Treating the density as a small perturbation around the mean $\bar{\rho}$, the reciprocal-space density perturbation is:

$$\delta\tilde{\rho}(\mathbf{k}) = \frac{i \beta}{k^2} \mathbf{k} \cdot \tilde{\mathbf{F}}(\mathbf{k})$$

where $\tilde{\mathbf{F}}(\mathbf{k})$ is the Fourier transform of the accumulated force field. The real-space density is then recovered by inverse FFT:

$$\rho(\mathbf{r}) = \bar{\rho} + \mathcal{F}^{-1}[\delta\tilde{\rho}(\mathbf{k})]$$

The mean density $\bar{\rho}$ is obtained from conventional counting and anchors the absolute scale of the result.

## Advantages

- **Lower variance in sparse regions.** The force field is estimated from all particles at all distances, not just those within a given bin. Low-density regions benefit most.
- **Complementary error structure.** Force-based and counting-based estimators make different errors in different regions, which makes combining them worthwhile.
- **No additional simulation cost.** Forces are already computed during MD; force sampling requires no changes to the simulation itself, only to the analysis.

## Implementation in revelsMD

RevelsMD implements force sampling as a two-stage process:

1. **Accumulation.** Forces are binned (for RDFs) or deposited onto a voxel grid (for density fields) and summed across trajectory frames.
2. **Conversion.** The accumulated force field is converted to $g(r)$ via numerical integration, or to $\rho(\mathbf{r})$ via FFT.

An optional third stage combines the force-based and counting-based estimates using position-dependent lambda weights to minimise total variance; see [Variance Reduction and Lambda Weighting](variance-reduction.md).

## Brief history

Force sampling methods for RDFs and 3D densities were introduced by Borgis, Assaraf, Rotenberg, and Vuilleumier (2013). Coles, Borgis, Vuilleumier, and Rotenberg (2019) extended the approach to rigid molecules, charge densities, and polarisation densities. The lambda combination approach — which RevelsMD also implements — was introduced by Coles, Mangaud, Frenkel, and Rotenberg (2021). A review by Rotenberg (2020) covers the wider context of force sampling methods.

## References

- Borgis, D., Assaraf, R., Rotenberg, B., & Vuilleumier, R. (2013). Computation of pair distribution functions and three-dimensional densities with a reduced variance principle. *Molecular Physics*, 111(22-23), 3486-3492. [doi:10.1080/00268976.2013.838316](https://doi.org/10.1080/00268976.2013.838316)

- Coles, S. W., Borgis, D., Vuilleumier, R., & Rotenberg, B. (2019). Computing three-dimensional densities from force densities improves statistical efficiency. *The Journal of Chemical Physics*, 151(6), 064124. [doi:10.1063/1.5111697](https://doi.org/10.1063/1.5111697)

- Rotenberg, B. (2020). Use the force! Reduced variance estimators for densities, radial distribution functions, and local mobilities in molecular simulations. *The Journal of Chemical Physics*, 153(15), 150902. [doi:10.1063/5.0029113](https://doi.org/10.1063/5.0029113)

- Coles, S. W., Mangaud, E., Frenkel, D., & Rotenberg, B. (2021). Reduced variance analysis of molecular dynamics simulations by linear combination of estimators. *The Journal of Chemical Physics*, 154(19), 191101. [doi:10.1063/5.0053737](https://doi.org/10.1063/5.0053737)
