# Export results

## RDF data

RDF results are plain NumPy arrays on the returned object.

```python
from revelsMD.rdf import compute_rdf

rdf = compute_rdf(ts, "1", "1", integration="forward")

rdf.r        # r values: bin edges 0, delr, ..., rmax
rdf.g_count  # histogram-based g(r)
rdf.g_force  # force-based g(r)
```

With `integration="lambda"`, the lambda values are also available as `rdf.lam`.

Save to a plain text file:

```python
import numpy as np

np.savetxt("rdf.txt", np.column_stack([rdf.r, rdf.g_count, rdf.g_force]),
           header="r  g_count  g_force")
```

Save to a NumPy binary archive:

```python
np.savez("rdf.npz", r=rdf.r, g_count=rdf.g_count, g_force=rdf.g_force)
```

## Density data

Density grids expose accumulated arrays as NumPy arrays.

```python
grid.rho_count       # histogram-based density
grid.rho_force       # force-based density
grid.rho_lambda      # lambda-weighted density
grid.lambda_weights  # accumulated lambda weights
```

Save individual arrays with `np.save`:

```python
np.save("rho_count.npy", grid.rho_count)
np.save("rho_force.npy", grid.rho_force)
```

Or save them together:

```python
np.savez("density.npz",
         rho_count=grid.rho_count,
         rho_force=grid.rho_force,
         rho_lambda=grid.rho_lambda,
         lambda_weights=grid.lambda_weights)
```

## Cube files

Export to a Gaussian cube file for VESTA or VMD:

```python
grid.write_to_cube("force", "force_density.cube")
grid.write_to_cube("count", "count_density.cube")
grid.write_to_cube("lambda", "lambda_density.cube")
grid.write_to_cube("hybrid", "hybrid_density.cube", threshold=0.01)
```

The first argument selects which density to write: `"force"`, `"count"`,
`"lambda"`, or `"hybrid"`. `threshold` is required for `"hybrid"` and
controls the force/count crossover.

Cube files are in atomic units, set by the grid's `density_type` whichever
density is written: lengths in Bohr, and values in Bohr⁻³ for `'number'`,
e Bohr⁻³ for `'charge'` and e Bohr⁻² for `'polarisation'`. The grid's own
arrays are per Å³, so a value in the file is 0.148 times the corresponding
value in `rho_count` and the other arrays (0.280 for `'polarisation'`); choose
isosurface levels from the file's values. Lengths are taken to be in Å, so for a trajectory in `'lj'`
units the file's units are nominal.

## Grid metadata

Cell geometry and grid dimensions:

```python
grid.cell_matrix   # (3, 3) array, rows are lattice vectors
grid.voxel_volume  # volume of a single voxel
grid.nbinsx        # number of bins along x
grid.nbinsy        # number of bins along y
grid.nbinsz        # number of bins along z
```
