# Export results

## RDF data

After computing an RDF, results are plain NumPy arrays on the returned object.

```python
from revelsMD.rdf import compute_rdf

rdf = compute_rdf(ts, "1", "1", integration="forward")

rdf.r        # bin centres
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

Density grids expose their accumulated arrays directly as NumPy arrays.

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

Export a density grid to a Gaussian cube file for use with VESTA or VMD:

```python
grid.write_to_cube("force", "force_density.cube")
grid.write_to_cube("count", "count_density.cube")
grid.write_to_cube("lambda", "lambda_density.cube")
grid.write_to_cube("hybrid", "hybrid_density.cube", threshold=0.01)
```

The first argument selects which density to write: `"force"`, `"count"`,
`"lambda"`, or `"hybrid"`. The `threshold` argument is required for
`"hybrid"` and controls the crossover between force and count density.

## Grid metadata

Cell geometry and grid dimensions are available as attributes:

```python
grid.cell_matrix   # (3, 3) array, rows are lattice vectors
grid.voxel_volume  # volume of a single voxel
grid.nbinsx        # number of bins along x
grid.nbinsy        # number of bins along y
grid.nbinsz        # number of bins along z
```
