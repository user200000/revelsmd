# Use the NumPy trajectory interface

## NumpyTrajectory

`NumpyTrajectory` wraps in-memory NumPy arrays so they can be used with
`DensityGrid.accumulate()` and `RDF.accumulate()` just like any
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

For triclinic cells, pass `cell_matrix` (a 3x3 array with rows as lattice
vectors) instead of `box_x/y/z`.

Charge and mass arrays can be supplied via `charge_list` and `mass_list`:

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

## Raw deposit()

For custom iteration loops you can call `deposit()` directly on a
`DensityGrid` or `RDF` object. Each call deposits a single frame.

```python
from revelsMD.frame_sources import Frame

# DensityGrid.deposit(positions, forces, weights)
for i in range(len(positions)):
    pos = positions[i]        # (atoms, 3)
    frc = forces[i]           # (atoms, 3)
    idx = species_indices     # precomputed integer indices for the species
    grid.deposit(pos[idx], frc[idx], weights=1.0)

# RDF.deposit(frame) — takes a Frame dataclass
from revelsMD.rdf import RDF

rdf = RDF(traj, 'O', 'O')
for i in range(len(positions)):
    frame = Frame(positions[i], forces[i])
    rdf.deposit(frame)
```

Using `accumulate()` via `NumpyTrajectory` is preferred because it
handles blocking, Welford statistics, and frame-range selection
automatically. Use `deposit()` only when you need fine-grained control
over which frames or pre-processed data to accumulate.
