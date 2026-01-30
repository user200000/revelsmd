"""
Cross-backend consistency tests for RevelsMD.

These tests verify that the same trajectory data loaded through different
backends produces identical (or very similar) analysis results. This helps
ensure that the different TrajectoryState implementations are consistent.

Tests include:
- LAMMPS vs NumPy (same data, different loaders)
- Forward vs backward RDF integration convergence
- Different grid resolutions producing consistent bulk density
"""

import pytest
import numpy as np

from revelsMD.revels_rdf import RevelsRDF
from revelsMD.density import GridState
from .conftest import lammps_to_numpy, mda_to_numpy, assert_arrays_close


@pytest.mark.integration
@pytest.mark.requires_example1
class TestLammpsVsNumpyConsistency:
    """
    Test that LAMMPS and NumPy backends produce identical results.

    We load the same trajectory data via LAMMPS and convert it to NumPy,
    then verify that RDF and density calculations produce the same output.
    """

    def test_rdf_identical(self, example1_trajectory):
        """Same trajectory via LAMMPS and NumPy should give identical RDF."""
        lammps_ts = example1_trajectory

        # Convert to NumPy (first 5 frames for speed)
        numpy_ts = lammps_to_numpy(lammps_ts, start=0, stop=5, stride=1)

        # Compute RDF via both using same explicit frame range
        # Note: stop=-1 gives (frames-1) due to stop % frames implementation
        # So use stop=4 on both to process frames 0,1,2,3
        n_frames_to_use = 4
        rdf_lammps = RevelsRDF.run_rdf(
            lammps_ts, '1', '1', temp=1.35,
            delr=0.02, start=0, stop=n_frames_to_use
        )

        rdf_numpy = RevelsRDF.run_rdf(
            numpy_ts, '1', '1', temp=1.35,
            delr=0.02, start=0, stop=n_frames_to_use
        )

        assert rdf_lammps is not None
        assert rdf_numpy is not None

        # Results should be very close (small numerical differences expected
        # due to different frame iteration methods between LAMMPS/NumPy backends)
        assert_arrays_close(
            rdf_lammps[0], rdf_numpy[0],
            rtol=1e-10, context="r values"
        )
        assert_arrays_close(
            rdf_lammps[1], rdf_numpy[1],
            rtol=1e-3, atol=1e-3, context="g(r) values"
        )

    def test_density_identical(self, example1_trajectory):
        """Same trajectory via LAMMPS and NumPy should give identical density."""
        lammps_ts = example1_trajectory

        # Convert to NumPy
        numpy_ts = lammps_to_numpy(lammps_ts, start=0, stop=5, stride=1)

        # Compute density via both using same explicit frame range
        n_frames_to_use = 4
        gs_lammps = GridState(lammps_ts, 'number', nbins=30, temperature=1.35)
        gs_lammps.make_force_grid(
            lammps_ts, '1', kernel='triangular', rigid=False,
            start=0, stop=n_frames_to_use
        )
        gs_lammps.get_real_density()

        gs_numpy = GridState(numpy_ts, 'number', nbins=30, temperature=1.35)
        gs_numpy.make_force_grid(
            numpy_ts, '1', kernel='triangular', rigid=False,
            start=0, stop=n_frames_to_use
        )
        gs_numpy.get_real_density()

        # Results should be very close (small numerical differences expected
        # due to different frame iteration methods between LAMMPS/NumPy backends)
        assert_arrays_close(
            gs_lammps.rho, gs_numpy.rho,
            rtol=1e-2, atol=1e-4, context="density values"
        )

    def test_frame_count_matches(self, example1_trajectory):
        """Converted NumPy trajectory should have correct frame count."""
        lammps_ts = example1_trajectory
        numpy_ts = lammps_to_numpy(lammps_ts, start=0, stop=10, stride=1)

        assert numpy_ts.frames == 10

    def test_box_dimensions_match(self, example1_trajectory):
        """Converted NumPy trajectory should have same box dimensions."""
        lammps_ts = example1_trajectory
        numpy_ts = lammps_to_numpy(lammps_ts, start=0, stop=5, stride=1)

        assert numpy_ts.box_x == lammps_ts.box_x
        assert numpy_ts.box_y == lammps_ts.box_y
        assert numpy_ts.box_z == lammps_ts.box_z


@pytest.mark.integration
@pytest.mark.requires_example4
class TestMDAVsNumpyConsistency:
    """
    Test that MDA and NumPy backends produce identical results.
    """

    def test_rdf_identical(self, example4_trajectory):
        """Same trajectory via MDA and NumPy should give identical RDF."""
        mda_ts = example4_trajectory

        # Convert to NumPy (first 5 frames)
        n_frames = 5
        numpy_ts = mda_to_numpy(mda_ts, start=0, stop=n_frames, stride=1)

        # Compute RDF via both using same frame range
        rdf_mda = RevelsRDF.run_rdf(
            mda_ts, 'Ow', 'Ow', temp=300,
            delr=0.1, start=0, stop=n_frames
        )

        rdf_numpy = RevelsRDF.run_rdf(
            numpy_ts, 'Ow', 'Ow', temp=300,
            delr=0.1, start=0, stop=None  # Process all frames in NumPy trajectory
        )

        assert rdf_mda is not None
        assert rdf_numpy is not None

        # Results should be very close (small numerical differences possible
        # due to different frame iteration between MDA and NumPy backends)
        assert_arrays_close(
            rdf_mda[0], rdf_numpy[0],
            rtol=1e-10, context="r values"
        )
        assert_arrays_close(
            rdf_mda[1], rdf_numpy[1],
            rtol=1e-3, atol=1e-3, context="g(r) values"
        )


# Note: Forward/backward consistency is tested via regression tests
# (test_rdf_forward_regression and test_rdf_backward_regression in test_regression.py)


@pytest.mark.integration
class TestGridResolutionConsistency:
    """
    Test that different grid resolutions produce consistent bulk densities.
    """

    def test_mean_density_resolution_independent(self, uniform_gas_trajectory):
        """Mean density should be similar regardless of grid resolution."""
        ts = uniform_gas_trajectory

        densities = []
        for nbins in [20, 40, 60]:
            gs = GridState(ts, 'number', nbins=nbins, temperature=1.0)
            gs.make_force_grid(ts, '1', kernel='triangular', rigid=False)
            gs.get_real_density()

            densities.append(np.mean(gs.rho))

        # All mean densities should be similar
        max_diff = max(densities) - min(densities)
        mean_density = np.mean(densities)

        if mean_density > 0:
            relative_diff = max_diff / mean_density
            assert relative_diff < 0.5, \
                f"Mean densities vary too much with resolution: {densities}"


@pytest.mark.integration
class TestKernelConsistency:
    """
    Test that triangular and box kernels produce consistent results.
    """

    def test_kernel_mean_density_similar(self, uniform_gas_trajectory):
        """Triangular and box kernels should give similar mean density."""
        ts = uniform_gas_trajectory

        # Triangular kernel
        gs_tri = GridState(ts, 'number', nbins=30, temperature=1.0)
        gs_tri.make_force_grid(ts, '1', kernel='triangular', rigid=False)
        gs_tri.get_real_density()

        # Box kernel
        gs_box = GridState(ts, 'number', nbins=30, temperature=1.0)
        gs_box.make_force_grid(ts, '1', kernel='box', rigid=False)
        gs_box.get_real_density()

        mean_tri = np.mean(gs_tri.rho)
        mean_box = np.mean(gs_box.rho)

        # Should be within 50% of each other
        if max(abs(mean_tri), abs(mean_box)) > 0:
            relative_diff = abs(mean_tri - mean_box) / max(abs(mean_tri), abs(mean_box))
            assert relative_diff < 0.5, \
                f"Kernel mean densities differ: tri={mean_tri:.4f}, box={mean_box:.4f}"


@pytest.mark.integration
@pytest.mark.requires_example1
class TestSpeciesConsistency:
    """
    Test that atom type/species handling is consistent across backends.
    """

    def test_species_count_matches(self, example1_trajectory):
        """Species counts should match between backends."""
        lammps_ts = example1_trajectory
        numpy_ts = lammps_to_numpy(lammps_ts, start=0, stop=5, stride=1)

        # Get type 1 indices from both
        lammps_indices = lammps_ts.get_indices('1')
        numpy_indices = numpy_ts.get_indices('1')

        assert len(lammps_indices) == len(numpy_indices), \
            f"Species count mismatch: LAMMPS={len(lammps_indices)}, NumPy={len(numpy_indices)}"

    def test_total_atoms_match(self, example1_trajectory):
        """Total atom count should match between backends."""
        lammps_ts = example1_trajectory
        numpy_ts = lammps_to_numpy(lammps_ts, start=0, stop=5, stride=1)

        # NumPy trajectory stores positions with shape (frames, atoms, 3)
        n_atoms_numpy = numpy_ts.positions.shape[1]

        # LAMMPS uses MDAnalysis universe
        n_atoms_lammps = len(lammps_ts.mdanalysis_universe.atoms)

        assert n_atoms_lammps == n_atoms_numpy, \
            f"Atom count mismatch: LAMMPS={n_atoms_lammps}, NumPy={n_atoms_numpy}"
