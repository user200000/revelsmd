# Kernels and deposition methods

When accumulating particle data onto a grid, each particle's contribution must be
assigned to one or more voxels. RevelsMD supports two deposition kernels:

- **Triangular (CIC)** — distributes each particle's contribution among the 8
  surrounding voxel vertices using trilinear interpolation. This is the default
  and produces smoother density fields with reduced grid-aliasing artefacts.
- **Box** — assigns the full contribution to the single voxel containing the
  particle. Faster but more susceptible to discretisation noise.

The kernel is selected via the `kernel` parameter of `DensityGrid.deposit()` and
`DensityGrid.accumulate()`. The RDF accumulator always uses triangular deposition
for its histogram-based estimate.

## Triangular (CIC) kernel

The triangular kernel is the 3D equivalent of the one-dimensional "Cloud-in-Cell"
scheme. For a particle at fractional position $(x, y, z)$ within a voxel, the
contribution is split among the 8 neighbouring vertices with weights:

$$
w_{ijk} = f_x^i \, f_y^j \, f_z^k, \quad i,j,k \in \{0,1\}
$$

where $f_x^0 = 1 - \xi_x$, $f_x^1 = \xi_x$, and $\xi_x$ is the fractional
offset within the voxel along $x$ (similarly for $y$ and $z$). All eight weights
sum to 1, so total contribution is conserved. Periodic boundary conditions are
enforced by wrapping vertex indices modulo the grid dimensions.

## Box kernel

The box kernel places the entire contribution in the host voxel — the voxel whose
lower corner is closest to the particle in fractional coordinates. It is equivalent
to nearest-grid-point (NGP) assignment. Use it when speed matters more than
smoothness, or when post-processing will smooth the result anyway.

## CIC normalisation for the RDF

When the triangular kernel is used for histogram-based g(r) computation, the
effective shell volume seen by each bin edge differs from the simple spherical shell
formula. The derivation below gives the exact effective volume required for correct
normalisation.

### Triangular deposition weights

For a pair at distance $d$ falling between bin edges $r_i$ and $r_{i+1}$
(where $r_{i+1} = r_i + \Delta r$), the weights deposited to each edge are:

$$
w_\text{lower} = \frac{r_{i+1} - d}{\Delta r}, \qquad
w_\text{upper} = \frac{d - r_i}{\Delta r}
$$

These sum to 1, so total pair count is conserved.

### Deriving the effective volume

For a bin edge at position $r$, pairs contribute from two adjacent shells:

1. **Lower shell** $[r - \Delta r,\; r]$: a pair at distance $d$ contributes
   weight $(d - r + \Delta r)/\Delta r$.
2. **Upper shell** $[r,\; r + \Delta r]$: a pair at distance $d$ contributes
   weight $(r + \Delta r - d)/\Delta r$.

For a uniform density $\rho$, the number of pairs at distance $d$ in a thin shell
of thickness $\mathrm{d}d$ is $\mathrm{d}N = \rho \cdot 4\pi d^2 \,\mathrm{d}d$.
The expected count at bin edge $r$ is therefore:

$$
N(r) = \rho \left[
  \int_{r-\Delta r}^{r} \frac{d - r + \Delta r}{\Delta r} \, 4\pi d^2 \,\mathrm{d}d
  \;+\;
  \int_{r}^{r+\Delta r} \frac{r + \Delta r - d}{\Delta r} \, 4\pi d^2 \,\mathrm{d}d
\right]
$$

#### Lower integral

$$
I_\text{lower} = \int_{r-\Delta r}^{r} \frac{d - r + \Delta r}{\Delta r} \, 4\pi d^2 \,\mathrm{d}d
$$

Substituting $u = d - r + \Delta r$ (so $d = u + r - \Delta r$, $\mathrm{d}d = \mathrm{d}u$,
limits $u: 0 \to \Delta r$):

$$
I_\text{lower} = \frac{4\pi}{\Delta r}
  \int_0^{\Delta r} u \,(u + r - \Delta r)^2 \,\mathrm{d}u
$$

Expanding and integrating term by term:

$$
I_\text{lower} = \frac{4\pi}{\Delta r} \left[
  \frac{\Delta r^4}{4} + \frac{2(r-\Delta r)\,\Delta r^3}{3}
  + \frac{(r-\Delta r)^2\,\Delta r^2}{2}
\right]
$$

$$
= \frac{\pi\,\Delta r}{3}\left(\Delta r^2 - 4r\,\Delta r + 6r^2\right)
$$

#### Upper integral

By an analogous substitution:

$$
I_\text{upper} = \frac{\pi\,\Delta r}{3}\left(\Delta r^2 + 4r\,\Delta r + 6r^2\right)
$$

#### Total effective volume

Adding $I_\text{lower}$ and $I_\text{upper}$ (the $\pm 4r\,\Delta r$ terms cancel):

$$
N(r) = \rho \cdot \frac{\pi\,\Delta r}{3}
  \left[\left(\Delta r^2 - 4r\,\Delta r + 6r^2\right)
        + \left(\Delta r^2 + 4r\,\Delta r + 6r^2\right)\right]
= \rho \cdot \frac{2\pi}{3} \,\Delta r \left(\Delta r^2 + 6r^2\right)
$$

The **effective volume** for interior bin edges (both adjacent shells present) is:

$$
V_\text{eff}(r) = \frac{2\pi}{3}\,\Delta r\left(\Delta r^2 + 6r^2\right)
$$

or equivalently $\frac{2\pi}{3}\Delta r^3 + 4\pi\,\Delta r\,r^2$.

### Boundary treatment

The formula above assumes contributions from both adjacent shells. At the boundaries
this assumption breaks down.

**First bin edge ($r = 0$):** only the upper shell $[0, \Delta r]$ exists. Using
$I_\text{upper}$ with $r = 0$:

$$
V_\text{eff}(0) = \frac{\pi\,\Delta r^3}{3}
$$

This is exactly half the value the general formula gives at $r = 0$. The
implementation corrects for this explicitly.

**Last bin edge ($r = r_\text{max}$):** the implementation allocates an extra
overflow bin extending to $r_\text{max} + \Delta r$ and discards it before
returning results. This means the last *returned* bin edge is an interior edge
with both shells contributing, so no special handling is needed.

### Comparison with shell-volume approximations

| Method | Formula | Error vs exact |
|--------|---------|----------------|
| Exact (triangular) | $\frac{2\pi}{3}\,\Delta r\,(\Delta r^2 + 6r^2)$ | 0 |
| Centred shell | $\frac{\pi}{3}\,\Delta r\,(\Delta r^2 + 12r^2)$ | $-\pi\,\Delta r^3/3$ |
| Average adjacent shells | $\frac{4\pi}{3}\,\Delta r\,(\Delta r^2 + 3r^2)$ | $+2\pi\,\Delta r^3/3$ |

Both approximations have errors proportional to $\Delta r^3$, which is negligible
when $\Delta r \ll r$, but can matter near $r = 0$.

### Numerical validation ($\Delta r = 0.1$)

| $r$ | Exact | Centred shell | Error |
|-----|-------|---------------|-------|
| 0.5 | 0.3163 | 0.3152 | -0.33% |
| 1.0 | 1.2587 | 1.2577 | -0.08% |
| 2.0 | 5.0286 | 5.0276 | -0.02% |
| 5.0 | 31.418 | 31.417 | -0.003% |
