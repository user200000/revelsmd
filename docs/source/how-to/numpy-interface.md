# Use the NumPy trajectory interface

## NumpyTrajectory

`NumpyTrajectory` wraps in-memory NumPy arrays for use with
`DensityGrid.accumulate()` and `RDF.accumulate()`, just like any
file-backed trajectory.

```python
import numpy as np
from revelsMD.trajectories import NumpyTrajectory
from revelsMD.density import DensityGrid

positions = np.load('positions.npy')   # shape (frames, atoms, 3)
forces    = np.load('forces.npy')      # shape (frames, atoms, 3)
species   = ['O', 'H', 'H'] * 100     # one label per atom

traj = NumpyTrajectory(
    positions, forces,
    box_x=20.0, box_y=20.0, box_z=20.0,
    species_list=species,
    temperature=300.0,
)

grid = DensityGrid(traj, density_type='number', nbins=50)
grid.accumulate(traj, atom_names='O')
```

For triclinic cells, pass `cell_matrix` (3x3, rows as lattice vectors)
instead of `box_x/y/z`.

Supply charges and masses via `charge_list` and `mass_list`:

```python
charges = np.array([-0.82, 0.41, 0.41] * 100)
masses  = np.array([15.999, 1.008, 1.008] * 100)

traj = NumpyTrajectory(
    positions, forces,
    box_x=20.0, box_y=20.0, box_z=20.0,
    species_list=species,
    temperature=300.0,
    charge_list=charges,
    mass_list=masses,
)
```

## A set of structures

A set of structures with the same atoms and cell, such as snapshots from
several independent runs, is a trajectory: stack their arrays and wrap them in
a `NumpyTrajectory`. Their order does not matter, since densities and RDFs are
equilibrium averages.

```python
traj = NumpyTrajectory(
    np.stack(positions_list),   # (structures, atoms, 3)
    np.stack(forces_list),      # (structures, atoms, 3)
    cell_matrix=cell,
    species_list=species,
    temperature=300.0,
)

grid = DensityGrid(traj, density_type='number', nbins=50)
grid.accumulate(traj, atom_names='O')
```

To add more structures later, call `accumulate()` again with another
trajectory with the same cell.
