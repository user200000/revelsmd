# Changelog

## 0.2.0 (unreleased)

The first release since 0.1.5. It replaces the 0.1.5 API
(`Revels3D.GridState`, `RevelsRDF`, the `*TrajectoryState` classes)
entirely; there is no compatibility layer.

### Calculations

- `RDF` and `compute_rdf` for force-sampled radial distribution functions:
  forward, backward and variance-minimised (lambda) integration, and a
  histogram estimate (`g_count`).
- `DensityGrid` and `compute_density` for force-sampled 3D number, charge and
  polarisation densities, with counting, force, lambda and hybrid estimates.
  Rigid molecules are supported, centred on their centre of mass or on a
  chosen atom.
- Lambda estimation for densities uses contiguous or interleaved blocks of
  frames; for RDFs each frame is a sample. Both keep running statistics.
- `accumulate()` adds frames from further trajectories of the same system,
  and checks that the cell, temperature and units (and for an RDF the
  species counts) match.
- Orthorhombic and triclinic cells.
- Cube files (`write_to_cube`) in atomic units.
- A numba backend (the default) and a numpy backend, selected with
  `REVELSMD_BACKEND`; `REVELSMD_FFT_WORKERS` sets the threads for density FFTs.

### Trajectories

- `LammpsTrajectory`: one or several dump files, including restarts,
  triclinic boxes, scaled or unwrapped coordinates, and runs that stopped
  mid-frame. Atoms are matched to the data file by id.
- `MDATrajectory`: any MDAnalysis format that carries forces in converted
  units (e.g. GROMACS TRR, H5MD).
- `VaspTrajectory`: `vasprun.xml` files, read directly; pymatgen is no longer
  needed.
- `NumpyTrajectory`: positions and forces from arrays.

### Changes in results from 0.1.5

- Forward-integrated g(r) was labelled one bin too early.
- Multi-species densities were averages over the species; they are now the
  total density of the selected atoms.
- MDAnalysis selections used atom ids rather than positions, selecting the
  wrong atoms for topologies numbered from 1 (e.g. GRO, PDB).
- LAMMPS dumps whose rows were not sorted by atom id were paired with the
  wrong atoms.

### Requirements

Python 3.12 or later, numpy 2.2, numba 0.61.2, scipy 1.15 and MDAnalysis 2.8
or later.
