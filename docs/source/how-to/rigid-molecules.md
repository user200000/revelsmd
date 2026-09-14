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

Centre-of-mass deposition requires mass data. `MDATrajectory` provides this
automatically; for `NumpyTrajectory`, pass `mass_list`.

## Charge density

Pass `density_type='charge'`. Charge data must be available on the trajectory.

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
: Use a backend that carries mass data (e.g. `MDATrajectory`), or switch to
  `centre_location=<int>` to deposit at a named atom site instead.

`ValueError: polarisation requires rigid`
: Set `rigid=True` and supply more than one atom name.
