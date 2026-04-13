# Rigid molecules

Compute densities for molecular species where forces on constituent atoms are
summed per molecule.

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

Each atom name must be unique. All selections must have the same number of
indices.

## Centre of mass vs atom site

By default, the density is deposited at the molecular centre of mass
(`centre_location=True`). To deposit at a specific atom instead, pass its
index into `atom_names`:

```python
grid.accumulate(
    traj,
    atom_names=['Ow', 'Hw1', 'Hw2'],
    rigid=True,
    centre_location=0,  # deposit at oxygen site
)
```

Centre-of-mass deposition requires mass data. `MDATrajectory` provides this
automatically. For `NumpyTrajectory`, pass `mass_list` to the constructor.

## Charge density

Pass `density_type='charge'` when constructing the grid. Charge data must be
available on the trajectory.

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

Pass `density_type='polarisation'` and specify `polarisation_axis`.
`rigid=True` with multiple atom names is required.

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

Both charge and mass data must be available on the trajectory.

## Troubleshooting

`ValueError: Duplicate atom names detected`
: Each entry in `atom_names` must be unique.

`ValueError` about selection sizes
: All atom names must select the same number of atoms.

`DataUnavailableError` for masses
: Use a backend that carries mass data (e.g. `MDATrajectory`), or switch to
  `centre_location=<int>` to deposit at a named atom site instead.

`ValueError: polarisation requires rigid`
: Set `rigid=True` and supply more than one atom name.
