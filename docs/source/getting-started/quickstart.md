# Quickstart

RevelsMD computes reduced-variance radial distribution functions (RDFs) and 3D
number densities from molecular dynamics trajectories. It uses force sampling
to produce estimators with lower statistical noise than conventional histogram
counting. You need an MD trajectory that includes per-atom forces.

## Radial distribution functions

Compute an RDF from a LAMMPS trajectory with `compute_rdf`:

```python
from revelsMD.trajectories import LammpsTrajectory
from revelsMD.rdf import compute_rdf

ts = LammpsTrajectory(
    "dump.nh.lammps",
    topology_file="data.min.nh",
    temperature=1.0,
    units="lj",
)

rdf = compute_rdf(ts, species_a="1", species_b="1", integration="forward")
```

The result carries both estimators:

```python
rdf.g_count   # conventional histogram RDF
rdf.g_force   # force-sampled RDF
```

Plot them together to see the variance reduction:

```python
import matplotlib.pyplot as plt

plt.plot(rdf.r, rdf.g_count, label="histogram")
plt.plot(rdf.r, rdf.g_force, label="force-sampled")
plt.xlabel("r")
plt.ylabel("g(r)")
plt.legend()
plt.show()
```

## 3D densities

Compute a 3D density with the lambda estimator from a VASP trajectory:

```python
from revelsMD.trajectories import VaspTrajectory
from revelsMD.density import compute_density

ts = VaspTrajectory("vasprun.xml", temperature=600.0)

grid = compute_density(
    ts, atom_names="Li", nbins=50, compute_lambda=True,
)
```

The grid carries three estimators and a hybrid selector:

```python
grid.rho_count             # histogram density
grid.rho_force             # force-sampled density
grid.rho_lambda            # lambda-weighted density
grid.rho_hybrid(0.1)       # lambda where lambda < threshold, force elsewhere
```

Export to a Gaussian cube file for visualisation in VESTA or similar:

```python
grid.write_to_cube("lambda", "density.cube")
```

## Rigid molecules

For rigid molecules (e.g. water), accumulate across all atoms in the group and
deposit at the centre of mass:

```python
from revelsMD.trajectories import MDATrajectory
from revelsMD.density import compute_density

ts = MDATrajectory("prod.trr", "prod.tpr", temperature=300.0)

grid = compute_density(
    ts,
    atom_names=["Ow", "Hw1", "Hw2"],
    rigid=True,
    nbins=50,
)
```

Each atom must have a unique name. By default the density is deposited at the
centre of mass of the group. See the [rigid molecules how-to](../how-to/rigid-molecules) for
further options including charge and polarisation densities.

## Trajectory formats

All trajectory backends share the same interface and can be used with any
RevelsMD function.

**LAMMPS**

```python
from revelsMD.trajectories import LammpsTrajectory

ts = LammpsTrajectory(
    "dump.nh.lammps",
    topology_file="data.min.nh",
    temperature=1.0,
    units="lj",
)
```

**VASP**

```python
from revelsMD.trajectories import VaspTrajectory

ts = VaspTrajectory("vasprun.xml", temperature=600.0)
```

**MDAnalysis** (GROMACS, AMBER, NAMD, and any other format supported by MDAnalysis)

```python
from revelsMD.trajectories import MDATrajectory

ts = MDATrajectory("prod.trr", "prod.tpr", temperature=300.0)
```

**NumPy arrays**

```python
from revelsMD.trajectories import NumpyTrajectory

ts = NumpyTrajectory(
    positions=positions,  # shape (n_frames, n_atoms, 3)
    forces=forces,        # shape (n_frames, n_atoms, 3)
    cell_matrix=cell,     # shape (3, 3), rows are lattice vectors
    species_list=["O", "O", "H", "H"],
    temperature=300.0,
    units="real",
)
```
