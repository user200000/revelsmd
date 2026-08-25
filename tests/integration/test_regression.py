"""
Integration tests comparing current results against committed reference data.

Reference .npz files under tests/reference_data/ are committed to git.
Regenerating them (scripts/generate_reference_data.py) is a REVIEWED
change: commit the new baselines with the change that motivated them and
explain the numerical shift in the pull request. Never regenerate simply
to make a red test pass -- that defeats the test (it is exactly how an
inverted lambda combination was once certified by this suite).

Tolerances are tiered for cross-machine robustness, while remaining
orders of magnitude tighter than any real regression:

- r grids: rtol 1e-10 (pure bin arithmetic)
- g(r) and density arrays: rtol 1e-7
- lambda-combined arrays: rtol 1e-5 (variance ratios amplify float noise)
- all tiers except r grids additionally carry the assert_arrays_close
  absolute floor of atol=1e-8 for near-zero values

The cross-machine risk differs by baseline type: density baselines are
FFT-based, where SIMD dispatch can vary between CPUs; RDF arrays are
cumsum-based, where the risk is numpy version drift rather than SIMD
dispatch.

The LAMMPS, MDA and VASP tests read trimmed trajectory subsets committed
in tests/data/, so CI runs the complete regression suite: a green CI run
verifies every committed baseline. The full-length trajectories live
outside the repository and are used only by scripts/validate_*.py.

The semantic guard for the lambda combination itself is
tests/test_rdf_lambda_invariant.py, which needs no reference data.
"""

import pytest
import numpy as np
from pathlib import Path

from revelsMD.rdf import compute_rdf
from revelsMD.density import DensityGrid
from .conftest import assert_arrays_close

REFERENCE_DIR = Path(__file__).parent.parent / "reference_data"


def load_reference(subdir: str, filename: str):
    """Load reference data, failing the test if it is missing."""
    ref_path = REFERENCE_DIR / subdir / filename
    if not ref_path.exists():
        pytest.fail(
            f"Reference data missing: {ref_path} — baselines are committed "
            f"to git; a missing file means a broken checkout or deleted "
            f"baseline, not a skippable condition."
        )
    return dict(np.load(ref_path, allow_pickle=True))


# ---------------------------------------------------------------------------
# LAMMPS Regression Tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.regression
@pytest.mark.requires_example1
class TestLammpsRegression:
    """Regression tests against stored LAMMPS Example 1 results."""

    def test_rdf_forward_regression(self, example1_trajectory):
        """RDF forward integration matches stored reference and has valid physics."""
        ref = load_reference("lammps_example1", "rdf_forward.npz")

        result = compute_rdf(
            example1_trajectory, '1', '1',
            delr=0.02, integration='forward', start=0, stop=5
        )

        # Regression check against stored data
        assert_arrays_close(
            result.r, ref['r'],
            rtol=1e-10, atol=0.0, context="r values"
        )
        assert_arrays_close(
            result.g, ref['g_r'],
            rtol=1e-7, context="g(r) forward"
        )
        assert_arrays_close(
            result.g_count, ref['g_count'],
            rtol=1e-7, context="g_count histogram estimator"
        )

        # Physical property checks (saves computing RDF twice)
        # g(r) should have a first peak (LJ fluid)
        assert np.max(result.g) > 1.0, "LJ fluid should have g(r) peak > 1"

        # Check first peak position (LJ sigma ~ 1.0 in reduced units)
        short_range_mask = result.r < 2.0
        short_range_r = result.r[short_range_mask]
        short_range_gr = result.g[short_range_mask]
        if len(short_range_gr) > 0:
            peak_idx = np.argmax(short_range_gr)
            peak_r = short_range_r[peak_idx]
            assert 0.8 < peak_r < 1.5, f"First peak at r = {peak_r}, expected near 1.0"

        # Check normalisation in bulk region (r > 3 sigma)
        bulk_mask = result.r > 3.0
        if np.any(bulk_mask):
            bulk_gr = result.g[bulk_mask]
            mean_bulk = np.mean(bulk_gr)
            assert abs(mean_bulk - 1.0) < 0.2, f"Bulk g(r) = {mean_bulk}, expected ~1.0"

    def test_rdf_forward_strided_regression(self, example1_trajectory):
        """RDF forward integration with frame stride matches stored reference."""
        ref = load_reference("lammps_example1", "rdf_forward_strided.npz")

        result = compute_rdf(
            example1_trajectory, '1', '1',
            delr=0.05, integration='forward', start=0, stop=10, period=2
        )

        assert_arrays_close(
            result.r, ref['r'],
            rtol=1e-10, atol=0.0, context="r values"
        )
        assert_arrays_close(
            result.g, ref['g'],
            rtol=1e-7, context="g(r) forward strided"
        )

    def test_rdf_backward_regression(self, example1_trajectory):
        """RDF backward integration matches stored reference."""
        ref = load_reference("lammps_example1", "rdf_backward.npz")

        result = compute_rdf(
            example1_trajectory, '1', '1',
            delr=0.02, integration='backward', start=0, stop=5
        )

        assert_arrays_close(
            result.r, ref['r'],
            rtol=1e-10, atol=0.0, context="r values"
        )
        assert_arrays_close(
            result.g, ref['g_r'],
            rtol=1e-7, context="g(r) backward"
        )

    def test_rdf_lambda_regression(self, example1_trajectory):
        """RDF lambda combination matches stored reference."""
        ref = load_reference("lammps_example1", "rdf_lambda.npz")

        result = compute_rdf(
            example1_trajectory, '1', '1',
            delr=0.02, start=0, stop=5, integration='lambda'
        )

        assert_arrays_close(
            result.r, ref['r'],
            rtol=1e-10, atol=0.0, context="r values"
        )
        assert_arrays_close(
            result.g, ref['g'],
            rtol=1e-5, context="RDF lambda g(r)"
        )
        assert_arrays_close(
            result.lam, ref['lam'],
            rtol=1e-5, context="RDF lambda weights"
        )

    def test_number_density_regression(self, example1_trajectory):
        """3D number density matches stored reference."""
        ref = load_reference("lammps_example1", "number_density.npz")

        gs = DensityGrid(
            example1_trajectory, 'number', nbins=30
        )
        gs.accumulate(
            example1_trajectory, '1', kernel='triangular',
            rigid=False, start=0, stop=5
        )


        assert_arrays_close(
            gs.rho_force, ref['rho'],
            rtol=1e-7, context="number density"
        )

    def test_number_density_box_regression(self, example1_trajectory):
        """3D number density with the box kernel matches stored reference."""
        ref = load_reference("lammps_example1", "number_density_box.npz")

        gs = DensityGrid(
            example1_trajectory, 'number', nbins=30
        )
        gs.accumulate(
            example1_trajectory, '1', kernel='box',
            rigid=False, start=0, stop=5
        )

        assert_arrays_close(
            gs.rho_force, ref['rho'],
            rtol=1e-7, context="number density box kernel"
        )


# ---------------------------------------------------------------------------
# LAMMPS Example 2 Regression Tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.regression
@pytest.mark.requires_example2
class TestLammpsExample2Regression:
    """Regression tests against stored LAMMPS Example 2 results."""

    def test_number_density_lambda_regression(self, example2_trajectory):
        """Lambda-combined number density matches stored reference."""
        ref = load_reference("lammps_example2", "number_density_lambda.npz")

        gs = DensityGrid(
            example2_trajectory, 'number', nbins=30
        )
        gs.accumulate(
            example2_trajectory, '2', kernel='triangular', rigid=False,
            start=0, stop=10, compute_lambda=True, blocking='contiguous',
            block_size=int(ref['lambda_block_size']),
        )

        assert_arrays_close(
            gs.rho_force, ref['rho_force'],
            rtol=1e-7, context="Example 2 rho_force"
        )
        assert_arrays_close(
            gs.rho_count, ref['rho_count'],
            rtol=1e-7, context="Example 2 rho_count"
        )
        assert_arrays_close(
            gs.rho_lambda, ref['rho_lambda'],
            rtol=1e-5, context="Example 2 rho_lambda"
        )
        assert_arrays_close(
            gs.lambda_weights, ref['lambda_weights'],
            rtol=1e-5, context="Example 2 lambda weights"
        )


# ---------------------------------------------------------------------------
# MDA/GROMACS Regression Tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.regression
@pytest.mark.requires_example4
class TestMDARegression:
    """Regression tests against stored MDA Example 4 results."""

    def test_rdf_lambda_regression(self, example4_trajectory):
        """RDF lambda matches stored reference."""
        ref = load_reference("mda_example4", "rdf_lambda_ow.npz")

        result = compute_rdf(
            example4_trajectory, 'Ow', 'Ow',
            delr=0.1, start=0, stop=5, integration='lambda'
        )

        assert_arrays_close(
            result.r, ref['r'],
            rtol=1e-10, atol=0.0, context="r values"
        )
        assert_arrays_close(
            result.g, ref['g'],
            rtol=1e-5, context="RDF lambda g(r) Ow-Ow"
        )
        assert_arrays_close(
            result.lam, ref['lam'],
            rtol=1e-5, context="RDF lambda weights Ow-Ow"
        )

    def test_rdf_forward_unlike_regression(self, example4_trajectory):
        """Unlike-pair RDF (Ow-Hw1, forward) matches stored reference.

        Pins the pair-enumeration path, which is loader-agnostic, so this
        single instance covers unlike pairs for all loaders.
        """
        ref = load_reference("mda_example4", "rdf_forward_ow_hw1.npz")

        result = compute_rdf(
            example4_trajectory, 'Ow', 'Hw1',
            delr=0.1, integration='forward', start=0, stop=5
        )

        assert_arrays_close(
            result.r, ref['r'],
            rtol=1e-10, atol=0.0, context="r values"
        )
        assert_arrays_close(
            result.g, ref['g'],
            rtol=1e-7, context="g(r) forward Ow-Hw1"
        )

    def test_number_density_regression(self, example4_trajectory):
        """3D number density matches stored reference."""
        ref = load_reference("mda_example4", "number_density_ow.npz")

        gs = DensityGrid(
            example4_trajectory, 'number', nbins=30
        )
        gs.accumulate(
            example4_trajectory, 'Ow', kernel='triangular',
            rigid=False, start=0, stop=5
        )


        assert_arrays_close(
            gs.rho_force, ref['rho'],
            rtol=1e-7, context="number density Ow"
        )

    def test_rigid_density_regression(self, example4_trajectory):
        """Rigid molecule number density matches stored reference."""
        ref = load_reference("mda_example4", "number_density_rigid.npz")

        gs = DensityGrid(
            example4_trajectory, 'number', nbins=30
        )
        gs.accumulate(
            example4_trajectory, ['Ow', 'Hw1', 'Hw2'], kernel='triangular',
            rigid=True, start=0, stop=5
        )


        assert_arrays_close(
            gs.rho_force, ref['rho'],
            rtol=1e-7, context="rigid number density"
        )

    def test_polarisation_density_regression(self, example4_trajectory):
        """Polarisation density matches stored reference."""
        ref = load_reference("mda_example4", "polarisation_density.npz")

        gs = DensityGrid(
            example4_trajectory, 'polarisation', nbins=30
        )
        gs.accumulate(
            example4_trajectory, ['Ow', 'Hw1', 'Hw2'], kernel='triangular',
            rigid=True, start=0, stop=5
        )


        assert_arrays_close(
            gs.rho_force, ref['rho'],
            rtol=1e-7, context="polarisation density"
        )


# ---------------------------------------------------------------------------
# VASP Regression Tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.regression
@pytest.mark.requires_vasp
class TestVASPRegression:
    """Regression tests against stored VASP results (BaSnF4)."""

    def test_rdf_lambda_regression(self, vasp_trajectory):
        """RDF lambda matches stored reference."""
        ref = load_reference("vasp_example3", "rdf_lambda_f_f.npz")

        result = compute_rdf(
            vasp_trajectory, 'F', 'F',
            delr=0.1, start=0, stop=10, integration='lambda'
        )

        assert_arrays_close(
            result.r, ref['r'],
            rtol=1e-10, atol=0.0, context="r values"
        )
        assert_arrays_close(
            result.g, ref['g'],
            rtol=1e-5, context="RDF lambda g(r) F-F"
        )
        assert_arrays_close(
            result.lam, ref['lam'],
            rtol=1e-5, context="RDF lambda weights F-F"
        )

    def test_number_density_regression(self, vasp_trajectory):
        """3D number density matches stored reference."""
        ref = load_reference("vasp_example3", "number_density_f.npz")

        gs = DensityGrid(
            vasp_trajectory, 'number', nbins=30
        )
        gs.accumulate(
            vasp_trajectory, 'F', kernel='triangular',
            rigid=False, start=0, stop=10
        )


        assert_arrays_close(
            gs.rho_force, ref['rho'],
            rtol=1e-7, context="number density F"
        )


# ---------------------------------------------------------------------------
# Synthetic Trajectory Regression Tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.regression
@pytest.mark.analytical
class TestSyntheticRegression:
    """
    Regression tests against stored synthetic trajectory results.

    Synthetic trajectories are generated from deterministic random seeds,
    so any drift beyond the tiered tolerances indicates a real change.
    """

    def test_uniform_gas_rdf_regression(self, uniform_gas_trajectory):
        """Uniform gas RDF matches stored reference."""
        ref = load_reference("synthetic", "uniform_gas_rdf.npz")

        result = compute_rdf(
            uniform_gas_trajectory, '1', '1',
            delr=0.1, start=0, stop=None, integration='lambda'
        )

        assert_arrays_close(
            result.r, ref['r'],
            rtol=1e-10, atol=0.0, context="r values"
        )
        assert_arrays_close(
            result.g, ref['g'],
            rtol=1e-7, context="uniform gas RDF g(r)"
        )
        assert_arrays_close(
            result.lam, ref['lam'],
            rtol=1e-5, context="uniform gas RDF lambda weights"
        )

    def test_uniform_gas_density_regression(self, uniform_gas_trajectory):
        """Uniform gas density matches stored reference."""
        ref = load_reference("synthetic", "uniform_gas_density.npz")

        gs = DensityGrid(
            uniform_gas_trajectory, 'number', nbins=30
        )
        gs.accumulate(
            uniform_gas_trajectory, '1', kernel='triangular', rigid=False
        )


        assert_arrays_close(
            gs.rho_force, ref['rho'],
            rtol=1e-7, context="uniform gas density"
        )

    def test_uniform_gas_density_lambda_regression(self, uniform_gas_trajectory):
        """Uniform gas lambda density and weights match stored reference."""
        ref = load_reference("synthetic", "uniform_gas_density.npz")

        gs = DensityGrid(
            uniform_gas_trajectory, 'number', nbins=30
        )
        gs.accumulate(
            uniform_gas_trajectory, '1', kernel='triangular', rigid=False,
            compute_lambda=True, blocking='contiguous',
            block_size=int(ref['lambda_block_size']),
        )

        assert_arrays_close(
            gs.rho_lambda, ref['rho_lambda'],
            rtol=1e-5, context="uniform gas rho_lambda"
        )
        assert_arrays_close(
            gs.lambda_weights, ref['lambda_weights'],
            rtol=1e-5, context="uniform gas lambda weights"
        )


# ---------------------------------------------------------------------------
# Meta-tests for reference data integrity
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.regression
class TestReferenceDataIntegrity:
    """Tests that verify the reference data files are valid."""

    def test_lammps_references_exist(self):
        """LAMMPS reference files exist and are loadable."""
        ref_dir = REFERENCE_DIR / "lammps_example1"
        if not ref_dir.exists():
            pytest.fail(
                f"Reference data missing: {ref_dir} — baselines are committed "
                f"to git; a missing directory means a broken checkout or "
                f"deleted baseline, not a skippable condition."
            )

        expected_files = [
            "rdf_forward.npz",
            "rdf_forward_strided.npz",
            "rdf_backward.npz",
            "rdf_lambda.npz",
            "number_density.npz",
            "number_density_box.npz",
        ]

        for filename in expected_files:
            ref_path = ref_dir / filename
            assert ref_path.exists(), f"Missing reference: {ref_path}"
            data = np.load(ref_path)
            assert len(data.files) > 0, f"Empty reference: {ref_path}"

    def test_lammps_example2_references_exist(self):
        """LAMMPS Example 2 reference files exist and are loadable."""
        ref_dir = REFERENCE_DIR / "lammps_example2"
        if not ref_dir.exists():
            pytest.fail(
                f"Reference data missing: {ref_dir} — baselines are committed "
                f"to git; a missing directory means a broken checkout or "
                f"deleted baseline, not a skippable condition."
            )

        expected_files = [
            "number_density_lambda.npz",
        ]

        for filename in expected_files:
            ref_path = ref_dir / filename
            assert ref_path.exists(), f"Missing reference: {ref_path}"
            data = np.load(ref_path)
            assert len(data.files) > 0, f"Empty reference: {ref_path}"

    def test_mda_references_exist(self):
        """MDA reference files exist and are loadable."""
        ref_dir = REFERENCE_DIR / "mda_example4"
        if not ref_dir.exists():
            pytest.fail(
                f"Reference data missing: {ref_dir} — baselines are committed "
                f"to git; a missing directory means a broken checkout or "
                f"deleted baseline, not a skippable condition."
            )

        expected_files = [
            "rdf_lambda_ow.npz",
            "rdf_forward_ow_hw1.npz",
            "number_density_ow.npz",
            "number_density_rigid.npz",
            "polarisation_density.npz",
        ]

        for filename in expected_files:
            ref_path = ref_dir / filename
            assert ref_path.exists(), f"Missing reference: {ref_path}"
            data = np.load(ref_path)
            assert len(data.files) > 0, f"Empty reference: {ref_path}"

    def test_vasp_references_exist(self):
        """VASP reference files exist and are loadable."""
        ref_dir = REFERENCE_DIR / "vasp_example3"
        if not ref_dir.exists():
            pytest.fail(
                f"Reference data missing: {ref_dir} — baselines are committed "
                f"to git; a missing directory means a broken checkout or "
                f"deleted baseline, not a skippable condition."
            )

        expected_files = [
            "rdf_lambda_f_f.npz",
            "number_density_f.npz",
        ]

        for filename in expected_files:
            ref_path = ref_dir / filename
            assert ref_path.exists(), f"Missing reference: {ref_path}"
            data = np.load(ref_path)
            assert len(data.files) > 0, f"Empty reference: {ref_path}"

    def test_synthetic_references_exist(self):
        """Synthetic reference files exist and are loadable."""
        ref_dir = REFERENCE_DIR / "synthetic"
        if not ref_dir.exists():
            pytest.fail(
                f"Reference data missing: {ref_dir} — baselines are committed "
                f"to git; a missing directory means a broken checkout or "
                f"deleted baseline, not a skippable condition."
            )

        expected_files = [
            "uniform_gas_rdf.npz",
            "uniform_gas_density.npz",
        ]

        for filename in expected_files:
            ref_path = ref_dir / filename
            assert ref_path.exists(), f"Missing reference: {ref_path}"
            data = np.load(ref_path)
            assert len(data.files) > 0, f"Empty reference: {ref_path}"
