"""The FFT density conversion equals a naive k-space reference."""

import numpy as np

from revelsMD.trajectories import NumpyTrajectory
from revelsMD.density import DensityGrid


def _reference_rho_force(grid, fx, fy, fz, counter, count):
    """Convention-independent correct density: Re of the full complex
    reconstruction with the masked-divergence numerator and full k^2."""
    nx, ny, nz = grid.nbinsx, grid.nbinsy, grid.nbinsz
    vv = grid.voxel_volume
    scale = 1.0 / (count * vv)
    Fx = np.fft.fftn(fx * scale); Fy = np.fft.fftn(fy * scale); Fz = np.fft.fftn(fz * scale)
    ax = [np.fft.fftfreq(n, d=1.0 / n) for n in (nx, ny, nz)]
    M = np.stack(np.meshgrid(*ax, indexing="ij"), axis=-1)
    kfull = 2 * np.pi * np.einsum("ab,ijkb->ijka", grid.cell_inverse, M)
    ksq = np.sum(kfull ** 2, axis=-1); ksq[0, 0, 0] = 1.0
    Mn = M.copy()
    for a, n in enumerate((nx, ny, nz)):
        if n % 2 == 0:
            Mn[(slice(None),) * a + (n // 2,) + (slice(None),) * (2 - a) + (a,)] = 0.0
    knum = 2 * np.pi * np.einsum("ab,ijkb->ijka", grid.cell_inverse, Mn)
    delk = (1j * grid.beta / ksq) * (knum[..., 0] * Fx + knum[..., 1] * Fy + knum[..., 2] * Fz)
    delk[0, 0, 0] = 0.0
    return (-np.fft.ifftn(delk)).real + np.mean(counter * (1.0 / (vv * count)))


def test_rho_force_matches_naive_reference():
    rng = np.random.default_rng(1)
    box = 12.0
    n_atoms, n_frames, nbins = 40, 3, 16
    positions = rng.uniform(0, box, (n_frames, n_atoms, 3))
    forces = rng.normal(0, 1.0, (n_frames, n_atoms, 3))
    traj = NumpyTrajectory(
        positions, forces, box_x=box, box_y=box, box_z=box,
        species_list=["A"] * n_atoms, temperature=1.3, units="lj",
    )
    grid = DensityGrid(traj, density_type="number", nbins=nbins)
    grid.accumulate(traj, atom_names="A")
    reference = _reference_rho_force(
        grid, grid.force_x, grid.force_y, grid.force_z, grid.counter, grid.count
    )
    np.testing.assert_allclose(grid.rho_force, reference, rtol=1e-12, atol=1e-14)


def test_reconstructed_density_has_no_imaginary_part():
    """Production rho_force matches the reference reconstruction (the real
    part of the full complex inverse transform) for a non-cubic
    orthorhombic cell with distinct even bin counts per axis, so each
    axis's Nyquist plane is exercised separately -- the masked Nyquist
    numerator leaves no imaginary component large enough to affect the
    result."""
    rng = np.random.default_rng(2)
    box_x, box_y, box_z = 12.0, 9.0, 15.0
    n_atoms, n_frames = 50, 3
    nbins = (10, 14, 18)  # distinct and even on every axis -> three Nyquist planes
    positions = rng.uniform(0, 1, (n_frames, n_atoms, 3)) * np.array([box_x, box_y, box_z])
    forces = rng.normal(0, 1.0, (n_frames, n_atoms, 3))
    traj = NumpyTrajectory(
        positions, forces, box_x=box_x, box_y=box_y, box_z=box_z,
        species_list=["A"] * n_atoms, temperature=1.0, units="lj",
    )
    grid = DensityGrid(traj, density_type="number", nbins=nbins)
    grid.accumulate(traj, atom_names="A")
    reference = _reference_rho_force(
        grid, grid.force_x, grid.force_y, grid.force_z, grid.counter, grid.count
    )
    np.testing.assert_allclose(grid.rho_force, reference, rtol=1e-12, atol=1e-14)


def test_triclinic_reconstruction_matches_reference():
    """The point of this module: on a triclinic cell, delta_rho(k) is only
    Hermitian away from the Nyquist planes once the divergence numerator is
    masked there, and triclinic metric cross-terms make a real-FFT
    reconstruction's implicit Hermitian fill-in convention-dependent at
    those planes. Re of the full complex reconstruction is not, so
    production must match the reference oracle for a triclinic cell just as
    closely as it does for an orthorhombic one.

    Checks both a cubic bin count and a triclinic cell with distinct even
    bin counts per axis, so each axis's independent Nyquist mask is
    exercised separately.
    """
    rng = np.random.default_rng(3)
    n_atoms, n_frames = 80, 4

    def check(cell, nbins):
        cell = np.asarray(cell)
        frac = rng.random((n_frames, n_atoms, 3))
        positions = np.einsum("fai,ij->faj", frac, cell)
        forces = rng.normal(0, 1.0, (n_frames, n_atoms, 3))
        traj = NumpyTrajectory(
            positions, forces, cell_matrix=cell,
            species_list=["A"] * n_atoms, temperature=1.0, units="lj",
        )
        grid = DensityGrid(traj, density_type="number", nbins=nbins)
        grid.accumulate(traj, atom_names="A")
        reference = _reference_rho_force(
            grid, grid.force_x, grid.force_y, grid.force_z, grid.counter, grid.count
        )
        np.testing.assert_allclose(grid.rho_force, reference, rtol=1e-12, atol=1e-14)

    check([[10.0, 0.0, 0.0], [3.0, 9.0, 0.0], [1.0, 2.0, 8.0]], nbins=16)
    check([[10.0, 0.0, 0.0], [3.0, 8.0, 0.0], [1.0, 2.0, 12.0]], nbins=(12, 16, 20))
