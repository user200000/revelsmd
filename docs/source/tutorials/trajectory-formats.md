# Trajectory formats

RevelsMD provides trajectory classes for LAMMPS, VASP, and MDAnalysis-compatible
formats, plus `NumpyTrajectory` for raw arrays. All backends expose the same interface,
so analysis code is format-independent.

## Requirements for all formats

Trajectories must have a fixed, fully periodic cell; variable-cell (NPT)
trajectories are not supported. Forces are required in every frame.
Reading a frame whose positions or forces contain NaN or infinity raises
an error.

Several files are joined in the order given, so pass a list or tuple in
run order. They must be a run and its restarts: LAMMPS and MDAnalysis files
whose timesteps start again from zero, as independent runs do, raise an
error, and VASP files, which record no timesteps, are joined whatever their
origin. Load each independent run as its own trajectory and accumulate them
into the same grid or RDF. `sorted(glob(...))` puts `run10` before `run2`; name files
with leading zeros or list them explicitly. A set, or the same file
twice, raises an error.

## LAMMPS

`LammpsTrajectory` reads LAMMPS custom dump files, with a LAMMPS data file as
the topology. The dump must contain atom ids, positions and forces; rows may
be in any order, since frames are sorted by id on loading:

```
dump 1 all custom 100 trajectory.dump id type x y z fx fy fz
```

Positions may be written as `x y z`, unwrapped `xu yu zu`, or scaled
`xs ys zs` or `xsu ysu zsu`. Orthogonal and triclinic boxes are supported (not
the general triclinic `abc origin` format). All three boundaries must be
periodic (`pp pp pp`), and the box must be the same in every kept frame: a
dump whose box changes raises an error. If the dump has a `type` column, each
atom's type must match the data file: reading a frame raises an error if
any atom's type differs, so runs that change types (`fix atom/swap`,
`set type`) cannot be read.

Positions are returned as written. MDAnalysis's own dump reader shifts
them so the box starts at the origin, so the two can differ by the box
origin (`xlo ylo zlo`).

```python
from revelsMD.trajectories import LammpsTrajectory

traj = LammpsTrajectory(
    'tests/data/example_1_LJ/dump.nh.lammps',
    'tests/data/example_1_LJ/data.fin.nh.data',
    temperature=1.35,
    units='lj',
)

print(traj.frames)        # number of frames
print(traj.cell_matrix)   # 3x3 lattice matrix, rows are lattice vectors
print(traj.temperature)   # 1.35
print(traj.beta)          # 1 / (kB * T) in the chosen unit system
```

The column layout of the data file's `Atoms` section is read from the
`Atoms # <style>` comment that `write_data` writes. The styles `atomic`,
`charge`, `full`, `molecular`, `bond` and `angle` are recognised. For any
other style, pass the layout as a column string; for a data file without the
comment, pass a column string or one of the style names above. MDAnalysis
ignores column names it does not know; charges are read only from a column
named `charge` (not `q`):

```python
traj = LammpsTrajectory(
    'trajectory.dump',
    'topology.data',
    temperature=300.0,
    units='real',
    atom_style='id type x y z',
)
```

Pass a list of dump files to read a run and its restarts as one trajectory:

```python
traj = LammpsTrajectory(
    ['run1.dump', 'run2.dump', 'run3.dump'],
    'topology.data',
    temperature=300.0,
    units='real',
)
```

List the files in run order. Each file's frames replace the previous file's
frames from its own first timestep on, as after a restart from a restart
file. A warning names every replaced frame whose timestep is not in the
joined trajectory, as when a restart ends before the frames it replaces, or
dumps less often than the run it replaces. Replaced frames are not compared
with their file's first frame, so a run whose box or atom count went wrong
after the restart point joins with its restart. A file that starts at or
before the first timestep of the file before it raises an error, since that
would discard the whole earlier file. Gaps in timesteps between or within the
files as written are not detected.

The last frame of any file may be incomplete, as left by a job killed while
writing: it is dropped with a warning, or silently if the next file's frames
replace it. An empty file, or one holding only the start of a frame header,
from a job that stopped before writing a frame, is dropped in the same way.
An incomplete frame anywhere else raises an error, so a file written with
`dump_modify append yes` across a restart must be split at the restart.

A dump file must hold only frames: other text before, between or after
frames, apart from the `ITEM: UNITS` and `ITEM: TIME` lines written by
`dump_modify units yes` and `time yes`, raises an error. A dump whose
`ITEM: UNITS` names a unit style other than `units` raises an error, so
pass the style the run used.

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
    'tests/data/example_3_vasp/vasprun.xml',
    temperature=600.0,
)

print(traj.frames)
print(traj.cell_matrix)
```

The unit system is always `'metal'` (eV). Join `vasprun.xml` files from
sequential restarts by passing them in order:

```python
traj = VaspTrajectory(
    ['run1/vasprun.xml', 'run2/vasprun.xml'],
    temperature=600.0,
)
```

`vasprun.xml` records no timesteps, so the order of the files, and repeated or
missing steps between them, cannot be checked. Pass each file separately: a
file holding two `vasprun.xml` documents joined with `cat` raises an error.

A `vasprun.xml` from a run that was killed before it finished is read up to its
last complete ionic step; an ionic step cut short is dropped with a warning. A
file with no complete ionic step raises an error.

The lattice must be the same in every ionic step; a run whose lattice changes
raises an error.

Atoms are selected by element symbol:

```python
indices = traj.get_indices('F')
```

## MDAnalysis

`MDATrajectory` reads trajectories through MDAnalysis and converts positions
and forces to Angstrom and kJ/(mol Angstrom). The format must record its
length and force units: GROMACS `.trr` and `.tng`, AMBER NetCDF, and H5MD
files with units are accepted. Other formats raise an error, including LAMMPS
dumps, which `LammpsTrajectory` reads, and formats without forces such as
`.xtc` and `.dcd`. Reading TNG files needs the `pytng` package and H5MD files
the `h5py` package, which are not installed with RevelsMD.

Every frame in the file must hold both positions and forces. For a `.trr`,
set `nstfout` equal to `nstxout`, and `nstvout` to 0 or a multiple of
`nstxout` (GROMACS writes a velocity-only frame otherwise). An H5MD file must
save positions and forces at the same steps. Every frame must also have the
same box as the first frame; reading a frame that does not meet these
requirements raises an error.

```python
from revelsMD.trajectories import MDATrajectory

traj = MDATrajectory(
    'tests/data/example_4_water/prod.trr',
    'tests/data/example_4_water/prod.tpr',
    temperature=300.0,
)

print(traj.frames)
print(traj.cell_matrix)
```

Pass a list of files, in run order, to read a run and its restarts as one
trajectory. Every frame's step is read when the trajectory is constructed
(from the frame headers, for TRR files), and steps must strictly increase
within each file. The files are joined by step as LAMMPS dumps are by
timestep: each file's frames replace the previous file's frames from its own
first step on, so the frame GROMACS repeats at the start of each file written
with `-noappend` is read once. A restart must continue the step count, as
`gmx mdrun -cpi` does; files whose steps start again from zero, including
files written by MDAnalysis (which numbers steps from zero), cannot be joined.
If any file records no steps (AMBER NetCDF), all are joined by time instead,
and NetCDF files without times cannot be joined. As for LAMMPS dumps, a
warning names every replaced frame whose step is not in the joined
trajectory, and a file that starts at or before the first step of the file
before it raises an error. Gaps in steps between the files are not detected. A last TRR frame
cut short, as left by a job killed while writing, is dropped with a warning.

The unit system is always `'mda'` (kJ/mol). Atoms are selected by name from the
topology:

```python
indices = traj.get_indices('Ow')  # oxygen atoms in SPC/E water
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

# Every other frame, skipping the first two
for frame in traj.iter_frames(start=2, stride=2):
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

`LammpsTrajectory` and `NumpyTrajectory` take `units` (default `'real'`).
`VaspTrajectory` always uses `'metal'` and `MDATrajectory` always uses `'mda'`.

## Species conventions

The string passed to `get_indices()` follows the source simulation's convention:

- **LAMMPS**: type number as a string — `'1'`, `'2'`
- **VASP**: element symbol — `'Ba'`, `'F'`
- **MDAnalysis**: atom name from the topology — `'Ow'`, `'Hw1'`
- **NumPy**: the label provided in `species_list`
