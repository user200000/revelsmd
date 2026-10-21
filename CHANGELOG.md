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
- Lambda is estimated in each half in time of the frames and applied to the
  other half, so it is never estimated from the samples it weights. Densities
  use contiguous or interleaved blocks of frames within each half; for RDFs
  each frame is a sample. Both keep running statistics. Lambda needs at least
  2 samples in each half, and a density warns with fewer than 10 blocks in a
  half; forward and backward g(r) need only one frame.
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
- Lambda was estimated from the same frames it weighted, which biased the
  combination. The lambda density fell short of the particle count where
  voxels are sparsely sampled: on a 7500-frame ab initio trajectory on a
  200^3 grid it held 72 to 82% of the atoms. The lambda g(r) was biased at
  small numbers of frames. With lambda estimated in one half of the frames
  and applied to the other, both are unbiased; each half's lambda is noisier,
  so a density needs at least 10 blocks in each half for the lambda estimate
  to be reliably better than the force estimate alone.

### Requirements

Python 3.12 or later, numpy 2.2, numba 0.61.2, scipy 1.15 and MDAnalysis 2.8
or later.
