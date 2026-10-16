# Rigid molecules

Compute densities for molecules where forces on constituent atoms are summed.

## Basic usage

Pass a list of atom names and set `rigid=True`:

```python
from revelsMD.density import DensityGrid

grid = DensityGrid(traj, density_type='number', nbins=50)
grid.accumulate(
    traj,
    atom_names=['Ow', 'Hw1', 'Hw2'],
    rigid=True,
)
```

Each atom name must be unique, and all selections must have the same number of
indices.

A rigid selection deposits one weight per molecule at the chosen centre, so
listing several atom names here selects the atoms that make up each molecule
rather than adding up separate species densities.

## Centre of mass vs atom site

By default, density is deposited at the centre of mass
(`centre_location=True`). To deposit at a specific atom, pass its index
into `atom_names`:

```python
grid.accumulate(
    traj,
    atom_names=['Ow', 'Hw1', 'Hw2'],
    rigid=True,
    centre_location=0,  # deposit at oxygen site
)
```

Centre-of-mass deposition requires masses, which are never guessed. They come
from a topology that records them (a GROMACS `.tpr`, or a LAMMPS data file with
a `Masses` section), or from the `masses` argument of `MDATrajectory` (by atom
name), `LammpsTrajectory` (by atom type) and `VaspTrajectory` (by element), or
the `mass_list` argument of `NumpyTrajectory`:

```python
traj = MDATrajectory('prod.trr', 'conf.gro', temperature=300.0,
                     masses={'Ow': 15.999, 'Hw1': 1.008, 'Hw2': 1.008})
```

## Charge density

Pass `density_type='charge'`. Charges come from the topology or, in the same
way as masses, from the `charges` argument.

```python
grid = DensityGrid(traj, density_type='charge', nbins=50)
grid.accumulate(
    traj,
    atom_names=['Ow', 'Hw1', 'Hw2'],
    rigid=True,
)
```

The summed molecular charge is deposited at the centre location.

## Polarisation density

Pass `density_type='polarisation'` and `polarisation_axis`. Requires
`rigid=True` with multiple atom names.

```python
grid = DensityGrid(traj, density_type='polarisation', nbins=50)
grid.accumulate(
    traj,
    atom_names=['Ow', 'Hw1', 'Hw2'],
    rigid=True,
    centre_location=True,
    polarisation_axis=2,   # 0=x, 1=y, 2=z
)
```

Requires both charge and mass data on the trajectory.

## Troubleshooting

`ValueError: Duplicate atom names detected`
: Each entry in `atom_names` must be unique.

`ValueError` about selection sizes
: All atom names must select the same number of atoms.

`DataUnavailableError` for masses
: Give them with `masses=`, use a topology that records them, or switch to
  `centre_location=<int>` to deposit at a named atom site instead.

`ValueError` about masses and the centre of mass
: Atom masses must be finite and not negative, and each molecule's total mass
  positive.

`ValueError: polarisation requires rigid`
: Set `rigid=True` and supply more than one atom name.
