# Compute charge and polarisation densities

## Charge density

Charge density requires per-atom charges on the trajectory. Pass
`density_type='charge'` to `DensityGrid`, then call `accumulate()` as normal.

```python
import numpy as np
from revelsMD.trajectories import NumpyTrajectory
from revelsMD.density import DensityGrid

positions = np.load('positions.npy')   # (frames, atoms, 3)
forces    = np.load('forces.npy')      # (frames, atoms, 3)
species   = ['O', 'H1', 'H2'] * 100
charges   = np.array([-0.82, 0.41, 0.41] * 100)

traj = NumpyTrajectory(
    positions, forces,
    box_x=20.0, box_y=20.0, box_z=20.0,
    species_list=species,
    temperature=300.0,
    charge_list=charges,
)

grid = DensityGrid(traj, density_type='charge', nbins=50)
grid.accumulate(traj, atom_names='O')

print(grid.rho_force)   # charge density at oxygen sites
```

For single-species accumulation, weights are per-atom charges. For
rigid-molecule accumulation, the summed molecular charge is deposited at the
centre location.

## Polarisation density

Polarisation density projects the molecular dipole moment along one
Cartesian axis. It requires `rigid=True` with charge and mass data on the
trajectory.

```python
masses = np.array([15.999, 1.008, 1.008] * 100)

traj = NumpyTrajectory(
    positions, forces,
    box_x=20.0, box_y=20.0, box_z=20.0,
    species_list=species,
    temperature=300.0,
    charge_list=charges,
    mass_list=masses,
)

grid = DensityGrid(traj, density_type='polarisation', nbins=50)
grid.accumulate(
    traj,
    atom_names=['O', 'H1', 'H2'],   # all atoms in the rigid group
    rigid=True,
    centre_location=True,          # deposit at centre of mass
    polarisation_axis=2,           # 0=x, 1=y, 2=z
)

print(grid.rho_force)   # polarisation density along z
```

`polarisation_axis` selects which dipole component is accumulated (default
0, i.e. x).

Valid density types are `'number'`, `'charge'`, and `'polarisation'`.
