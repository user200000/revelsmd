# RevelsMD

[![Tests](https://github.com/user200000/revelsmd/actions/workflows/tests.yml/badge.svg)](https://github.com/user200000/revelsmd/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/revelsMD)](https://pypi.org/project/revelsMD/)
[![Python](https://img.shields.io/pypi/pyversions/revelsMD)](https://pypi.org/project/revelsMD/)
[![Documentation](https://readthedocs.org/projects/revelsmd/badge/?version=latest)](https://revelsmd.readthedocs.io)
[![Licence: MIT](https://img.shields.io/github/license/user200000/revelsmd)](https://github.com/user200000/revelsmd/blob/main/LICENSE)

RevelsMD (REduced Variance Estimators of the Local Structure in MD simulations)
computes 3D densities and radial distribution functions from molecular dynamics
trajectories by force sampling, which gives estimates with much lower variance
than histogram counting.

It reads LAMMPS and VASP output, GROMACS, AMBER and H5MD trajectories with
forces (through MDAnalysis), and NumPy arrays.

Version 0.2.0 is a beta release ahead of 1.0.

## Installation

RevelsMD requires Python 3.12 or later.

```bash
pip install revelsMD
```

## Examples

A 3D lithium density from a VASP run, written to a cube file:

```python
from revelsMD.trajectories import VaspTrajectory
from revelsMD.density import compute_density

ts = VaspTrajectory("vasprun.xml", temperature=500.0)
grid = compute_density(ts, "Li", nbins=100, compute_lambda=True)

grid.rho_count   # histogram density
grid.rho_force   # force-sampled density
grid.rho_lambda  # lambda-weighted density
grid.write_to_cube("lambda", "li_density.cube")
```

![Counting, force-sampled and lambda-weighted Li densities in Li6PS5I](https://raw.githubusercontent.com/user200000/revelsmd/main/docs/source/_static/images/tutorial_density.png)

An RDF from a LAMMPS trajectory with forces:

```python
from revelsMD.trajectories import LammpsTrajectory
from revelsMD.rdf import compute_rdf

ts = LammpsTrajectory(
    "dump.nh.lammps",
    topology_file="data.fin.nh.data",
    temperature=1.35,
    units="lj",
    atom_style="id resid type charge x y z ix iy iz",
)
rdf = compute_rdf(ts, species_a="1", species_b="1", integration="lambda")

rdf.r        # r values
rdf.g        # force-sampled g(r), lambda-weighted
rdf.g_count  # histogram g(r)
```

<img src="https://raw.githubusercontent.com/user200000/revelsmd/main/docs/source/_static/images/tutorial_rdf.png" alt="Histogram and force-sampled g(r) for a Lennard-Jones fluid" width="450">

## Documentation

Tutorials, how-to guides and the API reference are at
[revelsmd.readthedocs.io](https://revelsmd.readthedocs.io).

## Citing RevelsMD

If you use RevelsMD in published work, please cite

- S. W. Coles, B. J. Morgan and B. Rotenberg, RevelsMD: Reduced Variance
  Estimators of the Local Structure in Molecular Dynamics,
  [arXiv:2310.06149](https://arxiv.org/abs/2310.06149) (2023)

and the papers describing the methods you use, below.

## Methods

RevelsMD implements the force-sampling estimators of

- D. Borgis, R. Assaraf, B. Rotenberg and R. Vuilleumier, *Mol. Phys.* **111**,
  3486 (2013), [doi:10.1080/00268976.2013.838316](https://doi.org/10.1080/00268976.2013.838316)
- S. W. Coles, D. Borgis, R. Vuilleumier and B. Rotenberg, *J. Chem. Phys.*
  **151**, 064124 (2019), [doi:10.1063/1.5111697](https://doi.org/10.1063/1.5111697)
- S. W. Coles, E. Mangaud, D. Frenkel and B. Rotenberg, *J. Chem. Phys.*
  **154**, 191101 (2021), [doi:10.1063/5.0053737](https://doi.org/10.1063/5.0053737)

## Contributing and licence

See [CONTRIBUTING.md](https://github.com/user200000/revelsmd/blob/main/CONTRIBUTING.md)
and [CHANGELOG.md](https://github.com/user200000/revelsmd/blob/main/CHANGELOG.md).
RevelsMD is released under the
[MIT licence](https://github.com/user200000/revelsmd/blob/main/LICENSE).
