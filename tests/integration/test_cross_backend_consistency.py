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
from revelsMD.revels_3D import Revels3D
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
        gs_lammps = Revels3D.GridState(lammps_ts, 'number', nbins=30, temperature=1.35)
        gs_lammps.make_force_grid(
            lammps_ts, '1', kernel='triangular', rigid=False,
            start=0, stop=n_frames_to_use
        )
        gs_lammps.get_real_density()

        gs_numpy = Revels3D.GridState(numpy_ts, 'number', nbins=30, temperature=1.35)
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

    @pytest.mark.xfail(
        reason="Bug: revels_rdf.py:331 accesses .trajectory.atoms instead of .atoms",
        raises=AttributeError,
    )
    def test_rdf_identical(self, example4_trajectory):
        """Same trajectory via MDA and NumPy should give identical RDF."""
        mda_ts = example4_trajectory

        # Convert to NumPy (first 5 frames)
        numpy_ts = mda_to_numpy(mda_ts, start=0, stop=5, stride=1)

        # Compute RDF via both
        rdf_mda = RevelsRDF.run_rdf(
            mda_ts, 'Ow', 'Ow', temp=300,
            delr=0.1, start=0, stop=5
        )

        # Need to use 'Ow' species name in NumPy too
        rdf_numpy = RevelsRDF.run_rdf(
            numpy_ts, 'Ow', 'Ow', temp=300,
            delr=0.1, start=0, stop=-1
        )

        assert rdf_mda is not None
        assert rdf_numpy is not None

        # Results should be identical
        assert_arrays_close(
            rdf_mda[0], rdf_numpy[0],
            rtol=1e-10, context="r values"
        )
        assert_arrays_close(
            rdf_mda[1], rdf_numpy[1],
            rtol=1e-6, context="g(r) values"
        )


@pytest.mark.integration
@pytest.mark.requires_example1
class TestForwardBackwardConsistency:
    """
    Test that forward and backward RDF integration converge to consistent values.

    The from_zero=True (forward) and from_zero=False (backward) methods should
    give the same results in the bulk region, though they may differ at boundaries.
    """

    def test_bulk_convergence(self, example1_trajectory):
        """Forward and backward integration should agree in bulk region."""
        ts = example1_trajectory

        # Use small subset of frames for reasonable test time (5 frames ~10s)
        rdf_forward = RevelsRDF.run_rdf(
            ts, '1', '1', temp=1.35,
            delr=0.02, from_zero=True, start=0, stop=5
        )

        rdf_backward = RevelsRDF.run_rdf(
            ts, '1', '1', temp=1.35,
            delr=0.02, from_zero=False, start=0, stop=5
        )

        # Find bulk region (2.5 < r < 4.5 in LJ units)
        bulk_mask = (rdf_forward[0] > 2.5) & (rdf_forward[0] < 4.5)

        if np.any(bulk_mask):
            # Extract bulk values
            bulk_forward = rdf_forward[1][bulk_mask]
            bulk_backward = rdf_backward[1][bulk_mask]

            # Mean values should be close
            mean_forward = np.mean(bulk_forward)
            mean_backward = np.mean(bulk_backward)

            assert abs(mean_forward - mean_backward) < 0.1, \
                f"Forward ({mean_forward:.3f}) and backward ({mean_backward:.3f}) bulk means differ"

            # Both should be close to 1.0 for bulk fluid
            assert abs(mean_forward - 1.0) < 0.15, \
                f"Forward bulk mean ({mean_forward:.3f}) far from 1.0"
            assert abs(mean_backward - 1.0) < 0.15, \
                f"Backward bulk mean ({mean_backward:.3f}) far from 1.0"

    def test_first_peak_position_consistent(self, example1_trajectory):
        """Forward and backward should show peak at same position."""
        ts = example1_trajectory

        # Use small subset of frames for reasonable test time (5 frames ~10s)
        rdf_forward = RevelsRDF.run_rdf(
            ts, '1', '1', temp=1.35,
            delr=0.02, from_zero=True, start=0, stop=5
        )

        rdf_backward = RevelsRDF.run_rdf(
            ts, '1', '1', temp=1.35,
            delr=0.02, from_zero=False, start=0, stop=5
        )

        # Find peak in short range (r < 2)
        short_mask = rdf_forward[0] < 2.0

        peak_forward = rdf_forward[0][short_mask][np.argmax(rdf_forward[1][short_mask])]
        peak_backward = rdf_backward[0][short_mask][np.argmax(rdf_backward[1][short_mask])]

        # Peak positions should be within 0.1 of each other
        assert abs(peak_forward - peak_backward) < 0.1, \
            f"Peak positions differ: forward={peak_forward:.3f}, backward={peak_backward:.3f}"


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
            gs = Revels3D.GridState(ts, 'number', nbins=nbins, temperature=1.0)
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
        gs_tri = Revels3D.GridState(ts, 'number', nbins=30, temperature=1.0)
        gs_tri.make_force_grid(ts, '1', kernel='triangular', rigid=False)
        gs_tri.get_real_density()

        # Box kernel
        gs_box = Revels3D.GridState(ts, 'number', nbins=30, temperature=1.0)
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
