# Trajectory formats

RevelsMD provides trajectory classes for LAMMPS, VASP, and MDAnalysis-compatible
formats, plus `NumpyTrajectory` for raw arrays. All backends expose the same interface,
so analysis code is format-independent.

## LAMMPS

`LammpsTrajectory` reads LAMMPS custom dump files, requiring both a dump file and a
topology (data) file. The dump must contain positions and forces:

```
dump 1 all custom 100 trajectory.dump id type x y z fx fy fz
```

```python
from revelsMD.trajectories import LammpsTrajectory

traj = LammpsTrajectory(
    'examples/example_1_LJ/dump.nh.lammps',
    'examples/example_1_LJ/data.fin.nh.data',
    temperature=1.35,
    units='lj',
    atom_style='id resid type q x y z ix iy iz',
)

print(traj.frames)        # number of frames
print(traj.cell_matrix)   # 3x3 lattice matrix, rows are lattice vectors
print(traj.temperature)   # 1.35
print(traj.beta)          # 1 / (kB * T) in the chosen unit system
```

Concatenate multiple dump files by passing a list:

```python
traj = LammpsTrajectory(
    ['run1.dump', 'run2.dump', 'run3.dump'],
    'topology.data',
    temperature=300.0,
    units='real',
)
```

Atoms are identified by LAMMPS type number, as a string:

```python
indices = traj.get_indices('1')  # atoms of type 1
```

## VASP

`VaspTrajectory` reads `vasprun.xml` from VASP MD runs. No configuration is needed
beyond a standard NVT or NVE run:

```python
from revelsMD.trajectories import VaspTrajectory

traj = VaspTrajectory(
    'examples/example_3_BaSnF4/r1/vasprun.xml',
    temperature=500.0,
)

print(traj.frames)
print(traj.cell_matrix)
```

The default unit system is `'metal'` (eV). Concatenate multiple `vasprun.xml` files
from sequential restarts:

```python
traj = VaspTrajectory(
    ['run1/vasprun.xml', 'run2/vasprun.xml'],
    temperature=500.0,
)
```

Atoms are selected by element symbol:

```python
indices = traj.get_indices('F')
```

`VaspTrajectory` requires pymatgen, which is installed with:

```
pip install revelsMD[vasp]
```

## MDAnalysis

`MDATrajectory` supports any format MDAnalysis can read: GROMACS (`.xtc`, `.trr`),
AMBER (`.nc`, `.mdcrd`), CHARMM/NAMD (`.dcd`), and others.

```python
from revelsMD.trajectories import MDATrajectory

traj = MDATrajectory(
    'examples/example_4_rigid_water/prod.trr',
    'examples/example_4_rigid_water/prod.tpr',
    temperature=300.0,
)

print(traj.frames)
print(traj.cell_matrix)
```

The default unit system is `'mda'` (kJ/mol). Atoms are selected by name from the
topology:

```python
indices = traj.get_indices('OW')  # oxygen atoms in SPC/E water
```

## NumPy arrays

`NumpyTrajectory` wraps in-memory position and force arrays, both of shape
`(frames, atoms, 3)`:

```python
import numpy as np
from revelsMD.trajectories import NumpyTrajectory

rng = np.random.default_rng(42)
n_frames, n_atoms = 500, 108
positions = rng.uniform(0, 20.0, (n_frames, n_atoms, 3))
forces = rng.normal(0, 1.0, (n_frames, n_atoms, 3))
species = ['Ar'] * n_atoms

traj = NumpyTrajectory(
    positions,
    forces,
    box_x=20.0,
    box_y=20.0,
    box_z=20.0,
    species_list=species,
    temperature=300.0,
    units='real',
)
```

For a triclinic cell, pass a 3x3 `cell_matrix` instead (rows are lattice vectors):

```python
cell = np.array([
    [20.0,  0.0,  0.0],
    [ 5.0, 18.0,  0.0],
    [ 0.0,  0.0, 20.0],
])

traj = NumpyTrajectory(
    positions,
    forces,
    cell_matrix=cell,
    species_list=species,
    temperature=300.0,
)
```

`box_x/box_y/box_z` and `cell_matrix` are mutually exclusive. Charges and masses
can be supplied as keyword arguments:

```python
charges = np.tile([-0.82, 0.41, 0.41], n_atoms // 3)
masses = np.tile([15.999, 1.008, 1.008], n_atoms // 3)

traj = NumpyTrajectory(
    positions,
    forces,
    box_x=20.0, box_y=20.0, box_z=20.0,
    species_list=species,
    temperature=300.0,
    charge_list=charges,
    mass_list=masses,
)
```

## The common interface

Every backend exposes the same attributes and methods.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `frames` | `int` | Number of frames |
| `cell_matrix` | `ndarray (3, 3)` | Lattice vectors (rows) |
| `temperature` | `float` | Temperature in Kelvin (or reduced T* for LJ) |
| `beta` | `float` | 1 / (kB T) in the trajectory's unit system |
| `units` | `str` | Unit system: `'real'`, `'metal'`, `'mda'`, or `'lj'` |

For orthorhombic cells, `traj.box_x`, `traj.box_y`, and `traj.box_z` return the diagonal
elements. For triclinic cells, use `traj.cell_matrix` directly.

### Iterating over frames

`iter_frames()` streams frames without loading the whole trajectory into memory. It
accepts `start`, `stop`, and `stride` following Python slice semantics:

```python
# All frames
for frame in traj.iter_frames():
    print(frame.positions.shape)  # (n_atoms, 3)
    print(frame.forces.shape)     # (n_atoms, 3)

# Every other frame, skipping the first 100
for frame in traj.iter_frames(start=100, stride=2):
    process(frame)
```

`Frame` is a frozen dataclass:

```python
frame = next(iter(traj.iter_frames()))

positions = frame.positions  # shape (n_atoms, 3), dtype float64
forces = frame.forces        # shape (n_atoms, 3), dtype float64

# The first atom's position
x, y, z = positions[0]
```

## Unit systems

The unit system controls how `beta` is computed from temperature.

| Value | Energy units | Typical use |
|-------|-------------|-------------|
| `'real'` | kcal/mol | LAMMPS real units |
| `'metal'` | eV | VASP, LAMMPS metal units |
| `'mda'` | kJ/mol | GROMACS, MDAnalysis |
| `'lj'` | epsilon (reduced) | Lennard-Jones simulations |

Each backend sets a sensible default: `LammpsTrajectory` defaults to `'real'`,
`VaspTrajectory` to `'metal'`, and `MDATrajectory` to `'mda'`. Override with `units`
when needed.

## Species conventions

The string passed to `get_indices()` follows the source simulation's convention:

- **LAMMPS**: type number as a string — `'1'`, `'2'`
- **VASP**: element symbol — `'Ba'`, `'F'`
- **MDAnalysis**: atom name from the topology — `'OW'`, `'HW1'`
- **NumPy**: the label provided in `species_list`
