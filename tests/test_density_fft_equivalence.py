"""The FFT density conversion equals a naive k-space reference."""

import numpy as np

from revelsMD.trajectories import NumpyTrajectory
from revelsMD.density import DensityGrid


def _reference_rho_force(grid, force_x, force_y, force_z, counter, count):
    """Independent reference for delta_rho(k) = i*beta*(k.F)/k^2, DC term 0.

    Uses the same rfft frequency convention as the production grid: rfftn on
    the last axis with rfftfreq for its frequencies.
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
    kvec = 2 * np.pi * np.einsum("ab,ijkb->ijka", grid.cell_inverse, M)
    k_dot_F = kvec[..., 0] * fx + kvec[..., 1] * fy + kvec[..., 2] * fz
    ksq = np.sum(kvec * kvec, axis=-1)
    ksq[0, 0, 0] = 1.0
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
