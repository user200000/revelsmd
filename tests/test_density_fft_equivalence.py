"""The FFT density conversion equals a naive k-space reference."""

import numpy as np

from revelsMD.trajectories import NumpyTrajectory
from revelsMD.density import DensityGrid


def _reference_rho_force(grid, force_x, force_y, force_z, counter, count):
    """Independent reference for delta_rho(k) = i*beta*(k.F)/k^2, DC term 0.

    Uses the same rfft frequency convention as the production grid: rfftn on
    the last axis with rfftfreq for its frequencies. The numerator k.F is a
    divergence (first-derivative operator); its Nyquist Miller mode on each
    even axis is zeroed before the dot product, per the standard spectral
    first-derivative convention for a real field. ksq (an inverse Laplacian,
    a second-derivative operator) is formed from the full, unzeroed
    k-vectors.
    """
    vv = grid.voxel_volume
    scale = 1.0 / (count * vv)
    fx = np.fft.rfftn(force_x * scale)
    fy = np.fft.rfftn(force_y * scale)
    fz = np.fft.rfftn(force_z * scale)
    nx, ny, nz = grid.nbinsx, grid.nbinsy, grid.nbinsz
    m1 = np.fft.fftfreq(nx, d=1.0 / nx)
    m2 = np.fft.fftfreq(ny, d=1.0 / ny)
    m3 = np.fft.rfftfreq(nz, d=1.0 / nz)
    M = np.stack(np.meshgrid(m1, m2, m3, indexing="ij"), axis=-1)
    kvec_full = 2 * np.pi * np.einsum("ab,ijkb->ijka", grid.cell_inverse, M)
    ksq = np.sum(kvec_full * kvec_full, axis=-1)
    ksq[0, 0, 0] = 1.0
    Mnum = M.copy()
    if nx % 2 == 0:
        Mnum[nx // 2, :, :, 0] = 0.0
    if ny % 2 == 0:
        Mnum[:, ny // 2, :, 1] = 0.0
    if nz % 2 == 0:
        Mnum[:, :, nz // 2, 2] = 0.0
    kvec = 2 * np.pi * np.einsum("ab,ijkb->ijka", grid.cell_inverse, Mnum)
    k_dot_F = kvec[..., 0] * fx + kvec[..., 1] * fy + kvec[..., 2] * fz
    del_k = (1j * grid.beta / ksq) * k_dot_F
    del_k[0, 0, 0] = 0.0
    del_n = -np.fft.irfftn(del_k, s=(nx, ny, nz), axes=(0, 1, 2))
    rho_count = counter * (1.0 / (vv * count))
    return del_n + np.mean(rho_count)


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
    """Full-complex reconstruction of delta_rho is real once the numerator's
    Nyquist mode is zeroed; a nonzero Nyquist wavenumber would leave an
    imaginary (unphysical) component."""
    rng = np.random.default_rng(2)
    box = 10.0
    n_atoms, n_frames, nbins = 50, 3, 16  # even grid -> has a Nyquist bin
    positions = rng.uniform(0, box, (n_frames, n_atoms, 3))
    forces = rng.normal(0, 1.0, (n_frames, n_atoms, 3))
    traj = NumpyTrajectory(
        positions, forces, box_x=box, box_y=box, box_z=box,
        species_list=["A"] * n_atoms, temperature=1.0, units="lj",
    )
    grid = DensityGrid(traj, density_type="number", nbins=nbins)
    grid.accumulate(traj, atom_names="A")
    vv = grid.voxel_volume
    scale = 1.0 / (grid.count * vv)
    Fx = np.fft.fftn(grid.force_x * scale)
    Fy = np.fft.fftn(grid.force_y * scale)
    Fz = np.fft.fftn(grid.force_z * scale)
    m = np.fft.fftfreq(nbins, d=1.0 / nbins)
    M = np.stack(np.meshgrid(m, m, m, indexing="ij"), axis=-1)
    kfull = 2 * np.pi * np.einsum("ab,ijkb->ijka", grid.cell_inverse, M)
    ksq = np.sum(kfull * kfull, axis=-1); ksq[0, 0, 0] = 1.0
    Mnum = M.copy()
    nyq = nbins // 2
    Mnum[nyq, :, :, 0] = 0.0; Mnum[:, nyq, :, 1] = 0.0; Mnum[:, :, nyq, 2] = 0.0
    knum = 2 * np.pi * np.einsum("ab,ijkb->ijka", grid.cell_inverse, Mnum)
    kdotF = knum[..., 0] * Fx + knum[..., 1] * Fy + knum[..., 2] * Fz
    delk = (1j * grid.beta / ksq) * kdotF; delk[0, 0, 0] = 0.0
    deln = -np.fft.ifftn(delk)
    scale_real = np.max(np.abs(deln.real))
    assert np.max(np.abs(deln.imag)) / scale_real < 1e-12
