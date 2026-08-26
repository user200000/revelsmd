"""
Cross-backend consistency tests for RevelsMD.

These tests verify that the same trajectory data loaded through different
backends produces analysis results consistent within tolerance. This helps
ensure that the different Trajectory implementations are consistent.

Tests include:
- LAMMPS vs NumPy (same data, different loaders): RDF and density
- MDA vs NumPy (same data, different loaders): RDF
- Different grid resolutions reproducing the analytic mean density
- Triangular vs box kernels reproducing the analytic mean density
"""

import pytest
import numpy as np

from revelsMD.rdf import compute_rdf
from revelsMD.density import DensityGrid
from .conftest import lammps_to_numpy, mda_to_numpy, assert_arrays_close


@pytest.mark.integration
@pytest.mark.requires_example1
class TestLammpsVsNumpyConsistency:
    """
    Test that LAMMPS and NumPy backends produce results consistent within
    tolerance.

    We load the same trajectory data via LAMMPS and convert it to NumPy,
    then verify that RDF and density calculations produce the same output.
    """

    def test_rdf_identical(self, example1_trajectory):
        """Same trajectory via LAMMPS and NumPy gives a consistent RDF."""
        lammps_ts = example1_trajectory

        # Convert to NumPy (first 5 frames for speed)
        numpy_ts = lammps_to_numpy(lammps_ts, start=0, stop=5, stride=1)

        # Use the same explicit frame range on both backends
        # (stop=4 processes frames 0,1,2,3)
        n_frames_to_use = 4
        rdf_lammps = compute_rdf(
            lammps_ts, '1', '1',
            delr=0.02, start=0, stop=n_frames_to_use
        )

        rdf_numpy = compute_rdf(
            numpy_ts, '1', '1',
            delr=0.02, start=0, stop=n_frames_to_use
        )

        assert rdf_lammps.r is not None
        assert rdf_numpy.r is not None

        # Results should be very close (small numerical differences expected
        # due to different frame iteration methods between LAMMPS/NumPy backends)
        assert_arrays_close(
            rdf_lammps.r, rdf_numpy.r,
            rtol=1e-10, context="r values"
        )
        assert_arrays_close(
            rdf_lammps.g, rdf_numpy.g,
            rtol=1e-3, atol=1e-3, context="g(r) values"
        )

    def test_density_identical(self, example1_trajectory):
        """Same trajectory via LAMMPS and NumPy gives a consistent density."""
        lammps_ts = example1_trajectory

        # Convert to NumPy
        numpy_ts = lammps_to_numpy(lammps_ts, start=0, stop=5, stride=1)

        # Compute density via both using same explicit frame range
        n_frames_to_use = 4
        gs_lammps = DensityGrid(lammps_ts, 'number', nbins=30)
        gs_lammps.accumulate(
            lammps_ts, '1', kernel='triangular', rigid=False,
            start=0, stop=n_frames_to_use
        )

        gs_numpy = DensityGrid(numpy_ts, 'number', nbins=30)
        gs_numpy.accumulate(
            numpy_ts, '1', kernel='triangular', rigid=False,
            start=0, stop=n_frames_to_use
        )

        # Results should be very close (small numerical differences expected
        # due to different frame iteration methods between LAMMPS/NumPy backends)
        assert_arrays_close(
            gs_lammps.rho_force, gs_numpy.rho_force,
            rtol=1e-2, atol=1e-4, context="density values"
        )


@pytest.mark.integration
@pytest.mark.requires_example4
class TestMDAVsNumpyConsistency:
    """
    Test that MDA and NumPy backends produce results consistent within
    tolerance.
    """

    def test_rdf_identical(self, example4_trajectory):
        """Same trajectory via MDA and NumPy gives a consistent RDF."""
        mda_ts = example4_trajectory

        # Convert to NumPy (first 5 frames)
        n_frames = 5
        numpy_ts = mda_to_numpy(mda_ts, start=0, stop=n_frames, stride=1)

        # Compute RDF via both using same frame range
        rdf_mda = compute_rdf(
            mda_ts, 'Ow', 'Ow',
            delr=0.1, start=0, stop=n_frames
        )

        rdf_numpy = compute_rdf(
            numpy_ts, 'Ow', 'Ow',
            delr=0.1, start=0, stop=None  # Process all frames in NumPy trajectory
        )

        assert rdf_mda.r is not None
        assert rdf_numpy.r is not None

        # Results should be very close (small numerical differences possible
        # due to different frame iteration between MDA and NumPy backends)
        assert_arrays_close(
            rdf_mda.r, rdf_numpy.r,
            rtol=1e-10, context="r values"
        )
        assert_arrays_close(
            rdf_mda.g, rdf_numpy.g,
            rtol=1e-3, atol=1e-3, context="g(r) values"
        )


# Note: Forward/backward consistency is tested via regression tests
# (test_rdf_forward_regression and test_rdf_backward_regression in test_regression.py)


# The uniform-gas fixture has 500 atoms in a 10x10x10 box, so its mean
# density is analytically 0.5. The spatial mean of rho_force is fixed by
# the deposited counts (the k=0 FFT component), so it reproduces N/V to
# machine precision: the measured relative deviation is ~1e-15 for every
# resolution and kernel. rtol=1e-12 keeps a margin of ~1000x over the
# measured deviation (well beyond the suite's >= 3x rule) while remaining
# many orders of magnitude tighter than any real regression.
UNIFORM_GAS_MEAN_DENSITY = 500 / 10.0**3
MEAN_DENSITY_RTOL = 1e-12


@pytest.mark.integration
class TestGridResolutionConsistency:
    """
    Test that different grid resolutions reproduce the analytic mean density.
    """

    def test_mean_density_resolution_independent(self, uniform_gas_trajectory):
        """Mean density equals the analytic N/V at every grid resolution."""
        ts = uniform_gas_trajectory

        resolutions = [20, 40, 60]
        densities = []
        for nbins in resolutions:
            gs = DensityGrid(ts, 'number', nbins=nbins)
            gs.accumulate(ts, '1', kernel='triangular', rigid=False)

            densities.append(np.mean(gs.rho_force))

        for nbins, mean_density in zip(resolutions, densities):
            assert mean_density == pytest.approx(
                UNIFORM_GAS_MEAN_DENSITY, rel=MEAN_DENSITY_RTOL
            ), f"Mean density at nbins={nbins} deviates from N/V: {mean_density}"

        # Cross-resolution consistency
        spread = max(densities) - min(densities)
        assert spread <= MEAN_DENSITY_RTOL * UNIFORM_GAS_MEAN_DENSITY, \
            f"Mean densities vary with resolution: {densities}"


@pytest.mark.integration
class TestKernelConsistency:
    """
    Test that triangular and box kernels reproduce the analytic mean density.
    """

    def test_kernel_mean_density_similar(self, uniform_gas_trajectory):
        """Both kernels give a mean density equal to the analytic N/V."""
        ts = uniform_gas_trajectory

        # Triangular kernel
        gs_tri = DensityGrid(ts, 'number', nbins=30)
        gs_tri.accumulate(ts, '1', kernel='triangular', rigid=False)

        # Box kernel
        gs_box = DensityGrid(ts, 'number', nbins=30)
        gs_box.accumulate(ts, '1', kernel='box', rigid=False)

        mean_tri = np.mean(gs_tri.rho_force)
        mean_box = np.mean(gs_box.rho_force)

        assert mean_tri == pytest.approx(
            UNIFORM_GAS_MEAN_DENSITY, rel=MEAN_DENSITY_RTOL
        ), f"Triangular-kernel mean density deviates from N/V: {mean_tri}"
        assert mean_box == pytest.approx(
            UNIFORM_GAS_MEAN_DENSITY, rel=MEAN_DENSITY_RTOL
        ), f"Box-kernel mean density deviates from N/V: {mean_box}"

        # Cross-kernel consistency
        assert abs(mean_tri - mean_box) <= MEAN_DENSITY_RTOL * UNIFORM_GAS_MEAN_DENSITY, \
            f"Kernel mean densities differ: tri={mean_tri}, box={mean_box}"
