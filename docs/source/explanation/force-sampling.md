# Force Sampling

Force sampling computes radial distribution functions (RDFs) and 3D density fields from molecular dynamics trajectories. Instead of counting how often particles visit a region, it uses the forces acting on particles to infer the underlying density. The result is the same quantity — $g(r)$ or $\rho(\mathbf{r})$ — but with lower variance, particularly in low-density or high-energy regions where histogram counts are sparse.

RevelsMD implements force sampling for both RDFs and 3D density fields, with an optional stage that combines force and counting estimates to minimise variance (see [Variance Reduction and Lambda Weighting](variance-reduction.md)).

## Histogram convergence

Histogram estimators for $g(r)$ and $\rho(\mathbf{r})$ converge as $1/\sqrt{N_\text{bin}}$, where $N_\text{bin}$ is the number of particle visits to a bin. In low-density regions — near energy barriers, in solvent-excluded volumes, or at distribution tails — bins accumulate counts slowly regardless of trajectory length. The estimator is unbiased but high-variance, and the variance is largest where physical interest is greatest.

## The force-based alternative

The potential of mean force $W(\mathbf{r})$ is related to the equilibrium density by:

$$W(\mathbf{r}) = -k_B T \ln \rho(\mathbf{r})$$

Taking the gradient:

$$\langle \mathbf{F}(\mathbf{r}) \rangle = -\nabla W(\mathbf{r}) = k_B T \nabla \ln \rho(\mathbf{r})$$

The mean force field encodes the same information as the density, but estimated from forces rather than counts. Forces are available at every particle at every frame, so the effective sample size is much larger.

## RDF from forces

Borgis et al. (2013) showed that the force density $F(r)$ — the mean radial force between particle pairs — can be integrated directly to yield $g(r)$ without passing through the potential of mean force. The key relation (Borgis et al. Eq. 9) is:

$$\rho_b\, h_{ab}(r) = -\beta \int_r^\infty F(r')\, dr'$$

where $h_{ab} = g_{ab} - 1$ and $\rho_b$ is the bulk number density. This gives $g(r)$ by linear integration of the force density, not by exponentiating a PMF.

In practice, this can be integrated in either direction, yielding two complementary estimators (Coles et al. 2021, Eqs. 6--7):

**Forward integration** (using $g(0) = 0$): accumulates force contributions from all pairs separated by a distance smaller than $r$.

**Backward integration** (using $g(\infty) = 1$): accumulates from all pairs separated by a distance larger than $r$, anchored at $g = 1$ at large separations.

Both are exact in principle but accumulate noise in different regions: forward is more accurate near $r = 0$, backward at large $r$. The [lambda method](variance-reduction.md) finds the optimal position-dependent combination of the two.

## 3D density from forces

For three-dimensional density fields, the analogous relationship is:

$$\nabla \rho(\mathbf{r}) = -\beta \rho(\mathbf{r}) \langle \mathbf{F}(\mathbf{r}) \rangle$$

This is solved in Fourier space. Treating the density as a small perturbation around the mean $\bar{\rho}$:

$$\delta\tilde{\rho}(\mathbf{k}) = \frac{i \beta}{k^2} \mathbf{k} \cdot \tilde{\mathbf{F}}(\mathbf{k})$$

where $\tilde{\mathbf{F}}(\mathbf{k})$ is the Fourier transform of the accumulated force field. The real-space density is then recovered by inverse FFT:

$$\rho(\mathbf{r}) = \bar{\rho} + \mathcal{F}^{-1}[\delta\tilde{\rho}(\mathbf{k})]$$

The mean density $\bar{\rho}$ comes from counting and anchors the absolute scale.

## Advantages

- **Lower variance in sparse regions.** The force field draws on all particles at all distances, not just those within a given bin.
- **Complementary error structure.** Force and counting estimators err differently by region, making combination worthwhile.
- **No additional simulation cost.** Forces are already computed during MD; force sampling requires changes only to the analysis.

## Implementation in revelsMD

RevelsMD implements force sampling as a two-stage process:

1. **Accumulation.** Forces are binned (for RDFs) or deposited onto a voxel grid (for density fields) and summed across trajectory frames.
2. **Conversion.** The accumulated force field is converted to $g(r)$ via numerical integration, or to $\rho(\mathbf{r})$ via FFT.

The lambda combination step is described in [Variance Reduction and Lambda Weighting](variance-reduction.md).

## Brief history

Borgis, Assaraf, Rotenberg, and Vuilleumier (2013) introduced force sampling for RDFs and 3D densities. Coles, Borgis, Vuilleumier, and Rotenberg (2019) extended it to rigid molecules, charge densities, and polarisation densities. Coles, Mangaud, Frenkel, and Rotenberg (2021) introduced the lambda combination, also implemented in RevelsMD. Rotenberg (2020) reviews the wider context.

## References

- Borgis, D., Assaraf, R., Rotenberg, B., & Vuilleumier, R. (2013). Computation of pair distribution functions and three-dimensional densities with a reduced variance principle. *Molecular Physics*, 111(22-23), 3486-3492. [doi:10.1080/00268976.2013.838316](https://doi.org/10.1080/00268976.2013.838316)

- Coles, S. W., Borgis, D., Vuilleumier, R., & Rotenberg, B. (2019). Computing three-dimensional densities from force densities improves statistical efficiency. *The Journal of Chemical Physics*, 151(6), 064124. [doi:10.1063/1.5111697](https://doi.org/10.1063/1.5111697)

- Rotenberg, B. (2020). Use the force! Reduced variance estimators for densities, radial distribution functions, and local mobilities in molecular simulations. *The Journal of Chemical Physics*, 153(15), 150902. [doi:10.1063/5.0029113](https://doi.org/10.1063/5.0029113)

- Coles, S. W., Mangaud, E., Frenkel, D., & Rotenberg, B. (2021). Reduced variance analysis of molecular dynamics simulations by linear combination of estimators. *The Journal of Chemical Physics*, 154(19), 191101. [doi:10.1063/5.0053737](https://doi.org/10.1063/5.0053737)
