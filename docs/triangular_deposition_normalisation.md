# Normalisation for Triangular (CIC) Deposition in Histogram g(r)

## Background

When computing histogram-based g(r), we accumulate pair counts and normalise by the expected count for an ideal gas. With triangular (Cloud-in-Cell) deposition, each pair's contribution is distributed linearly between adjacent bin edges, which affects the normalisation.

## Triangular Deposition

For a pair at distance d between bin edges r_i and r_{i+1} (where r_{i+1} = r_i + delr):

- Weight to r_i: w_lower = (r_{i+1} - d) / delr
- Weight to r_{i+1}: w_upper = (d - r_i) / delr

Note that w_lower + w_upper = 1 (total weight conserved).

## Deriving the Effective Volume

For a bin edge at position r, pairs contribute from two adjacent shells:

1. **Lower shell [r - delr, r]**: A pair at distance d contributes weight (d - (r - delr)) / delr = (d - r + delr) / delr
2. **Upper shell [r, r + delr]**: A pair at distance d contributes weight ((r + delr) - d) / delr

For uniform density rho, the number of pairs at distance d in a thin shell of thickness dd is:

    dN = rho × 4*pi*d^2 dd

The expected count at bin edge r is the sum of weighted contributions from both shells:

    N(r) = rho × [ integral from r-delr to r of ((d - r + delr)/delr) × 4*pi*d^2 dd
                 + integral from r to r+delr of ((r + delr - d)/delr) × 4*pi*d^2 dd ]

### Evaluating the Lower Integral

    I_lower = integral from r-delr to r of ((d - r + delr)/delr) × 4*pi*d^2 dd

Let u = d - r + delr, so d = u + r - delr and dd = du.
When d = r - delr, u = 0; when d = r, u = delr.

    I_lower = (4*pi/delr) × integral from 0 to delr of u × (u + r - delr)^2 du

Expanding (u + r - delr)^2 = u^2 + 2u(r - delr) + (r - delr)^2:

    I_lower = (4*pi/delr) × integral of [u^3 + 2u^2(r - delr) + u(r - delr)^2] du

Integrating from 0 to delr:

    I_lower = (4*pi/delr) × [delr^4/4 + 2(r - delr)*delr^3/3 + (r - delr)^2*delr^2/2]

    I_lower = (4*pi/delr) × delr^2 × [delr^2/4 + 2(r - delr)*delr/3 + (r - delr)^2/2]

Simplifying:

    I_lower = 4*pi*delr × [delr^2/4 + 2r*delr/3 - 2*delr^2/3 + r^2/2 - r*delr + delr^2/2]

    I_lower = 4*pi*delr × [delr^2(1/4 - 2/3 + 1/2) + r*delr(2/3 - 1) + r^2/2]

    I_lower = 4*pi*delr × [delr^2(3/12 - 8/12 + 6/12) + r*delr(-1/3) + r^2/2]

    I_lower = 4*pi*delr × [delr^2/12 - r*delr/3 + r^2/2]

    I_lower = (pi*delr/3) × (delr^2 - 4*r*delr + 6*r^2)

### Evaluating the Upper Integral

    I_upper = integral from r to r+delr of ((r + delr - d)/delr) × 4*pi*d^2 dd

By symmetry of the calculation (or direct evaluation):

    I_upper = (pi*delr/3) × (delr^2 + 4*r*delr + 6*r^2)

### Total Expected Count

Adding the two contributions (note the +/-4*r*delr terms cancel):

    N(r) = rho × [I_lower + I_upper]
         = rho × (pi*delr/3) × [(delr^2 - 4*r*delr + 6*r^2) + (delr^2 + 4*r*delr + 6*r^2)]
         = rho × (pi*delr/3) × [2*delr^2 + 12*r^2]
         = rho × (2*pi/3) × delr × (delr^2 + 6*r^2)

## Effective Volume Formula

For interior bin edges (where both adjacent shells contribute), the effective volume is:

    V_eff(r) = (2*pi/3) × delr × (delr^2 + 6*r^2)

Or equivalently:

    V_eff(r) = (2*pi/3)*delr^3 + 4*pi*delr*r^2

### Boundary Treatment

The derivation assumes contributions from both adjacent shells. At boundary edges, only one shell contributes:

**First bin edge (r = 0):** Only the upper shell [0, delr] exists. Using I_upper with r = 0:

    V_eff(r=0) = (pi*delr/3) × (delr^2 + 0 + 0) = pi*delr^3/3

This is exactly half the value the general formula gives at r = 0. The implementation must handle this case explicitly.

**Last bin edge (r = r_max):** Only the lower shell [r_max - delr, r_max] exists. However, in our implementation we use n+1 bins internally (extending to r_max + delr) and discard the final bin from the returned results. This means the "last" returned bin edge is actually an interior edge with both shells contributing, so no special handling is needed — we simply use the general formula and then exclude the true boundary bin from output.

## Comparison with Shell Volume Approximations

| Method | Formula | Error vs Exact |
|--------|---------|----------------|
| Exact (triangular) | (2*pi/3) × delr × (delr^2 + 6*r^2) | 0 |
| Centred shell | (pi/3) × delr × (delr^2 + 12*r^2) | -pi*delr^3/3 |
| Average adjacent shells | (4*pi/3) × delr × (delr^2 + 3*r^2) | +2*pi*delr^3/3 |

The centred shell approximation has half the error magnitude of the average adjacent shells, and both have errors proportional to delr^3 (negligible when delr << r).

## Numerical Validation

For delr = 0.1:

| r | Exact | Centred Shell | Error |
|---|-------|---------------|-------|
| 0.5 | 0.3163 | 0.3152 | -0.33% |
| 1.0 | 1.2587 | 1.2577 | -0.08% |
| 2.0 | 5.0286 | 5.0276 | -0.02% |
| 5.0 | 31.418 | 31.417 | -0.003% |

## Implementation

The normalisation for histogram g(r) should use:

```python
# Exact effective volume for triangular deposition (interior edges)
# V_eff(r) = (2*pi/3) * delr * (delr^2 + 6*r^2)
eff_vol = (2.0 * np.pi / 3.0) * delr * (delr**2 + 6.0 * r_vals**2)

# Boundary correction: at r=0, only the upper shell contributes
# V_eff(0) = pi*delr^3/3 (half the general formula)
eff_vol[0] = np.pi * delr**3 / 3.0

# Note: The last bin edge (r_max + delr) also has only one shell, but we
# discard this bin from the returned results, so no correction is needed.

# Ideal count at each bin edge
# N_ideal = N_ref × (N_target / V_box) × V_eff × n_frames
#
# For unlike species (A-B): N_ref = N_A, N_target = N_B
# For like species (A-A):   N_ref × N_target -> N_A × (N_A - 1) / 2
#                           (counting each pair once)
rho_target = n_target / volume
if like_species:
    ideal_count = (n_ref * (n_ref - 1) / 2) * (1 / volume) * eff_vol * n_frames
else:
    ideal_count = n_ref * rho_target * eff_vol * n_frames

# Histogram g(r)
g_count = observed_count / ideal_count
```
