# RevelsMD

RevelsMD (REduced Variance Estimators of the Local Structure in MD simulations)
computes reduced-variance radial distribution functions and 3D densities from
molecular dynamics trajectories using force sampling methods.

It reads LAMMPS and VASP output, GROMACS, AMBER and H5MD trajectories with
forces (through MDAnalysis), and NumPy arrays.

```{toctree}
:maxdepth: 2
:hidden:
:caption: Getting Started

getting-started/installation
getting-started/quickstart
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
how-to/rigid-molecules
how-to/numpy-interface
how-to/export-results
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

