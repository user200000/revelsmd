# RevelsMD

RevelsMD (REduced Variance Estimators of the Local Structure in MD simulations)
computes reduced-variance radial distribution functions and 3D densities from
molecular dynamics trajectories using force sampling methods.

It processes output from LAMMPS, VASP, and any code supported by MDAnalysis,
and can be driven directly from NumPy arrays.

```{toctree}
:maxdepth: 2
:hidden:
:caption: Getting Started

getting-started/installation
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: Tutorials

tutorials/first-rdf
tutorials/3d-densities
tutorials/trajectory-formats
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: How-to Guides

how-to/density-types
how-to/lambda-weighting
how-to/blocking
how-to/numpy-interface
how-to/performance-tuning
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: Explanation

explanation/force-sampling
explanation/variance-reduction
explanation/kernels
explanation/block-averaging
explanation/architecture
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: API Reference

api/trajectories
api/density
api/rdf
api/supporting
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: Contributing

contributing/development-setup
contributing/conventions
contributing/new-trajectory-backend
```
