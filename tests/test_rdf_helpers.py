"""
Unit tests for RDF helper functions.

These tests verify the correctness of the vectorised helper functions
used in RDF calculations, following TDD principles.
"""

from __future__ import annotations

import os
import numpy as np
import pytest

from revelsMD.revels_rdf import RevelsRDF


class TestBackendSelection:
    """Test the backend selection mechanism."""

    def test_default_backend_is_numba(self):
        """Default backend should be numba."""
        from revelsMD.rdf_helpers import get_backend_functions

        pairwise_fn, accum_fn = get_backend_functions()

        assert 'numba' in pairwise_fn.__module__

    def test_explicit_numpy_backend(self):
        """Explicit numpy backend selection."""
        from revelsMD.rdf_helpers import get_backend_functions

        pairwise_fn, _ = get_backend_functions('numpy')

        assert 'numba' not in pairwise_fn.__module__

    def test_explicit_numba_backend(self):
        """Explicit numba backend selection."""
        pytest.importorskip('numba')
        from revelsMD.rdf_helpers import get_backend_functions

        pairwise_fn, _ = get_backend_functions('numba')

        assert 'numba' in pairwise_fn.__module__

    def test_invalid_backend_raises(self):
        """Invalid backend should raise ValueError."""
        from revelsMD.rdf_helpers import get_backend_functions

        with pytest.raises(ValueError, match="Unknown RDF backend"):
            get_backend_functions('invalid_backend')

    def test_environment_variable_backend(self):
        """Backend can be set via environment variable."""
        pytest.importorskip('numba')
        from revelsMD.rdf_helpers import get_backend_functions

        old_val = os.environ.get('REVELSMD_BACKEND')
        try:
            os.environ['REVELSMD_BACKEND'] = 'numba'
            pairwise_fn, _ = get_backend_functions()
            assert 'numba' in pairwise_fn.__module__
        finally:
            if old_val is None:
                os.environ.pop('REVELSMD_BACKEND', None)
            else:
                os.environ['REVELSMD_BACKEND'] = old_val

    def test_backends_produce_identical_results(self):
        """NumPy and Numba backends should produce identical results."""
        pytest.importorskip('numba')
        from revelsMD.rdf_helpers import get_backend_functions

        np.random.seed(42)
        pos = np.random.uniform(0, 10, (50, 3))
        forces = np.random.randn(50, 3)
        box = (10.0, 10.0, 10.0)
        bins = np.arange(0, 5, 0.1)

        np_pairwise, np_accum = get_backend_functions('numpy')
        nb_pairwise, nb_accum = get_backend_functions('numba')

        r_np, dot_np = np_pairwise(pos, pos, forces, forces, box)
        r_nb, dot_nb = nb_pairwise(pos, pos, forces, forces, box)

        storage_np = np_accum(dot_np, r_np, bins)
        storage_nb = nb_accum(dot_nb, r_nb, bins)

        np.testing.assert_allclose(r_np, r_nb, rtol=1e-14)
        np.testing.assert_allclose(storage_np, storage_nb, rtol=1e-10)


class TestMinimumImage:
    """Test the minimum image convention helper function."""

    def test_no_wrapping_within_half_box(self):
        """Displacement within half-box should be unchanged."""
        from revelsMD.rdf_helpers import apply_minimum_image

        box = np.array([10.0, 10.0, 10.0])
        displacement = np.array([2.0, -3.0, 4.0])

        result = apply_minimum_image(displacement, box)

        np.testing.assert_array_almost_equal(result, displacement)

    def test_wrap_positive_displacement(self):
        """Large positive displacement should wrap to negative."""
        from revelsMD.rdf_helpers import apply_minimum_image

        box = np.array([10.0, 10.0, 10.0])
        # Displacement of 7.0 in x should wrap to -3.0
        displacement = np.array([7.0, 0.0, 0.0])

        result = apply_minimum_image(displacement, box)

        np.testing.assert_array_almost_equal(result, np.array([-3.0, 0.0, 0.0]))

    def test_wrap_negative_displacement(self):
        """Large negative displacement should wrap to positive."""
        from revelsMD.rdf_helpers import apply_minimum_image

        box = np.array([10.0, 10.0, 10.0])
        # Displacement of -8.0 in y should wrap to +2.0
        displacement = np.array([0.0, -8.0, 0.0])

        result = apply_minimum_image(displacement, box)

        np.testing.assert_array_almost_equal(result, np.array([0.0, 2.0, 0.0]))

    def test_exactly_half_box(self):
        """Displacement of exactly half-box should remain unchanged."""
        from revelsMD.rdf_helpers import apply_minimum_image

        box = np.array([10.0, 10.0, 10.0])
        displacement = np.array([5.0, -5.0, 5.0])

        result = apply_minimum_image(displacement, box)

        # At exactly half-box, no wrapping should occur
        np.testing.assert_array_almost_equal(result, displacement)

    def test_3d_diagonal_wrapping(self):
        """Wrapping in all three dimensions simultaneously."""
        from revelsMD.rdf_helpers import apply_minimum_image

        box = np.array([10.0, 10.0, 10.0])
        displacement = np.array([7.0, -8.0, 6.0])

        result = apply_minimum_image(displacement, box)

        # 7 -> -3, -8 -> 2, 6 -> -4
        np.testing.assert_array_almost_equal(result, np.array([-3.0, 2.0, -4.0]))

    def test_non_cubic_box(self):
        """MIC with non-cubic orthorhombic box."""
        from revelsMD.rdf_helpers import apply_minimum_image

        box = np.array([8.0, 10.0, 12.0])
        displacement = np.array([5.0, 7.0, 8.0])

        result = apply_minimum_image(displacement, box)

        # 5 > 4 -> -3, 7 > 5 -> -3, 8 > 6 -> -4
        np.testing.assert_array_almost_equal(result, np.array([-3.0, -3.0, -4.0]))

    def test_batch_displacements(self):
        """Apply MIC to multiple displacements at once."""
        from revelsMD.rdf_helpers import apply_minimum_image

        box = np.array([10.0, 10.0, 10.0])
        displacements = np.array([
            [2.0, 3.0, 4.0],    # No wrapping
            [7.0, 0.0, 0.0],    # Wrap x
            [0.0, -8.0, 0.0],   # Wrap y
        ])

        result = apply_minimum_image(displacements, box)

        expected = np.array([
            [2.0, 3.0, 4.0],
            [-3.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
        ])
        np.testing.assert_array_almost_equal(result, expected)

    def test_matches_original_formula(self):
        """Results must match the original ceil-based formula exactly."""
        from revelsMD.rdf_helpers import apply_minimum_image

        # Original formula from revels_rdf.py:
        # rx -= (np.ceil((np.abs(rx) - box_x / 2) / box_x)) * (box_x) * np.sign(rx)

        np.random.seed(42)
        box = np.array([10.0, 12.0, 8.0])
        displacements = np.random.uniform(-15, 15, (100, 3))

        result = apply_minimum_image(displacements, box)

        # Compute expected using original formula
        expected = displacements.copy()
        for i in range(3):
            expected[:, i] -= (
                np.ceil((np.abs(expected[:, i]) - box[i] / 2) / box[i])
                * box[i]
                * np.sign(expected[:, i])
            )

        np.testing.assert_array_almost_equal(result, expected)


class TestPairwiseContributions:
    """Test pairwise distance and force projection calculation."""

    def test_two_atoms_like_species_distance(self):
        """Two atoms of the same species with known separation."""
        from revelsMD.rdf_helpers import compute_pairwise_contributions

        pos = np.array([
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ])
        forces = np.array([
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ])
        box = (10.0, 10.0, 10.0)

        r_flat, dot_prod_flat = compute_pairwise_contributions(pos, pos, forces, forces, box)

        # For like-species, should have n*(n-1)/2 = 1 pair
        assert r_flat.shape == (1,)
        assert dot_prod_flat.shape == (1,)

        # Distance between atoms 0 and 1 should be 3.0
        np.testing.assert_almost_equal(r_flat[0], 3.0)

    def test_force_difference_projection(self):
        """(F_j - F_i) . r_ij / |r|^3 computed correctly for like-species."""
        from revelsMD.rdf_helpers import compute_pairwise_contributions

        # Atom 0 at origin, atom 1 at (3, 0, 0)
        # Force on atom 0 is (0, 0, 0), force on atom 1 is (2, 0, 0)
        pos = np.array([
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ])
        forces = np.array([
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ])
        box = (10.0, 10.0, 10.0)

        r_flat, dot_prod_flat = compute_pairwise_contributions(pos, pos, forces, forces, box)

        # r_ij = pos[1] - pos[0] = (3, 0, 0), |r| = 3
        # F_j - F_i = (2, 0, 0) - (0, 0, 0) = (2, 0, 0)
        # dot_prod = (F_j - F_i) . r_ij / |r|^3 = (2*3) / 27 = 6/27 = 2/9
        expected = 2.0 / 9.0
        np.testing.assert_almost_equal(dot_prod_flat[0], expected)

    def test_force_projection_perpendicular(self):
        """Force perpendicular to r gives zero projection."""
        from revelsMD.rdf_helpers import compute_pairwise_contributions

        # Atom 0 at origin, atom 1 at (3, 0, 0)
        # Force difference perpendicular to r
        pos = np.array([
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ])
        forces = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],  # Perpendicular force difference
        ])
        box = (10.0, 10.0, 10.0)

        r_flat, dot_prod_flat = compute_pairwise_contributions(pos, pos, forces, forces, box)

        # Perpendicular force should give zero contribution
        np.testing.assert_almost_equal(dot_prod_flat[0], 0.0)

    def test_wrapped_distance(self):
        """Atoms requiring MIC wrapping."""
        from revelsMD.rdf_helpers import compute_pairwise_contributions

        # Atom 0 at (1, 0, 0), atom 1 at (9, 0, 0) in box of 10
        # Wrapped distance should be 2.0, not 8.0
        pos = np.array([
            [1.0, 0.0, 0.0],
            [9.0, 0.0, 0.0],
        ])
        forces = np.array([
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ])
        box = (10.0, 10.0, 10.0)

        r_flat, dot_prod_flat = compute_pairwise_contributions(pos, pos, forces, forces, box)

        np.testing.assert_almost_equal(r_flat[0], 2.0)

    def test_unlike_species_one_each(self):
        """Simplest unlike-species case: one A, one B."""
        from revelsMD.rdf_helpers import compute_pairwise_contributions

        pos_a = np.array([[0.0, 0.0, 0.0]])
        pos_b = np.array([[4.0, 0.0, 0.0]])
        forces_a = np.array([[1.0, 0.0, 0.0]])
        forces_b = np.array([[0.5, 0.0, 0.0]])
        box = (10.0, 10.0, 10.0)

        r_flat, dot_prod_flat = compute_pairwise_contributions(
            pos_a, pos_b, forces_a, forces_b, box
        )

        # Should have 1 pair
        assert r_flat.shape == (1,)
        assert dot_prod_flat.shape == (1,)
        np.testing.assert_almost_equal(r_flat[0], 4.0)

    def test_unlike_force_difference_projection(self):
        """(F_A - F_B) . r_AB / |r|^3 computed correctly for unlike-species."""
        from revelsMD.rdf_helpers import compute_pairwise_contributions

        pos_a = np.array([[0.0, 0.0, 0.0]])
        pos_b = np.array([[3.0, 0.0, 0.0]])
        forces_a = np.array([[2.0, 0.0, 0.0]])  # F_A
        forces_b = np.array([[0.5, 0.0, 0.0]])  # F_B
        box = (10.0, 10.0, 10.0)

        r_flat, dot_prod_flat = compute_pairwise_contributions(
            pos_a, pos_b, forces_a, forces_b, box
        )

        # r = pos_a - pos_b = (-3, 0, 0), |r| = 3
        # F_A - F_B = (1.5, 0, 0)
        # dot_prod = (1.5 * -3) / 27 = -4.5 / 27 = -1/6
        expected = 1.5 * (-3) / 27
        np.testing.assert_almost_equal(dot_prod_flat[0], expected)

    def test_unlike_multiple_atoms(self):
        """Multiple atoms in each species."""
        from revelsMD.rdf_helpers import compute_pairwise_contributions

        pos_a = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ])  # 2 A atoms
        pos_b = np.array([
            [5.0, 0.0, 0.0],
            [6.0, 0.0, 0.0],
            [7.0, 0.0, 0.0],
        ])  # 3 B atoms
        forces_a = np.ones((2, 3))
        forces_b = np.ones((3, 3))
        box = (10.0, 10.0, 10.0)

        r_flat, dot_prod_flat = compute_pairwise_contributions(
            pos_a, pos_b, forces_a, forces_b, box
        )

        # Should have 2 * 3 = 6 pairs
        assert r_flat.shape == (6,)
        assert dot_prod_flat.shape == (6,)

    def test_like_species_uses_upper_triangle(self):
        """For like-species, function should only compute n(n-1)/2 pairs."""
        from revelsMD.rdf_helpers import compute_pairwise_contributions

        n = 10
        np.random.seed(42)
        pos = np.random.uniform(0, 10, (n, 3))
        forces = np.random.randn(n, 3)
        box = (10.0, 10.0, 10.0)

        r_flat, dot_flat = compute_pairwise_contributions(
            pos, pos, forces, forces, box
        )

        # For like-species (same array), should have n*(n-1)/2 unique pairs
        expected_pairs = n * (n - 1) // 2
        assert r_flat.shape[0] == expected_pairs, (
            f"Expected {expected_pairs} pairs for n={n}, got {r_flat.shape[0]}"
        )

    def test_unlike_species_uses_full_matrix(self):
        """For unlike-species, function computes all n1*n2 pairs."""
        from revelsMD.rdf_helpers import compute_pairwise_contributions

        n1, n2 = 10, 15
        np.random.seed(43)
        pos_a = np.random.uniform(0, 10, (n1, 3))
        pos_b = np.random.uniform(0, 10, (n2, 3))
        forces_a = np.random.randn(n1, 3)
        forces_b = np.random.randn(n2, 3)
        box = (10.0, 10.0, 10.0)

        r_flat, dot_flat = compute_pairwise_contributions(
            pos_a, pos_b, forces_a, forces_b, box
        )

        # For unlike-species, should have n1 * n2 pairs
        expected_pairs = n1 * n2
        assert r_flat.shape[0] == expected_pairs, (
            f"Expected {expected_pairs} pairs, got {r_flat.shape[0]}"
        )


class TestBinnedAccumulation:
    """Test the bincount-based accumulation."""

    def test_single_value_single_bin(self):
        """One value lands in one bin."""
        from revelsMD.rdf_helpers import accumulate_binned_contributions

        values = np.array([1.5])
        distances = np.array([0.5])
        bins = np.array([0.0, 1.0, 2.0, 3.0])

        result = accumulate_binned_contributions(values, distances, bins)

        # Distance 0.5 should go in bin 0 (0-1)
        assert result.shape == (4,)
        np.testing.assert_almost_equal(result[0], 1.5)
        np.testing.assert_almost_equal(result[1], 0.0)

    def test_multiple_values_same_bin(self):
        """Multiple values accumulate in same bin."""
        from revelsMD.rdf_helpers import accumulate_binned_contributions

        values = np.array([1.0, 2.0, 3.0])
        distances = np.array([0.5, 0.7, 0.9])  # All in bin 0
        bins = np.array([0.0, 1.0, 2.0, 3.0])

        result = accumulate_binned_contributions(values, distances, bins)

        np.testing.assert_almost_equal(result[0], 6.0)  # 1 + 2 + 3

    def test_values_in_last_bin_zeroed(self):
        """Values in final bin are excluded (matching original behaviour)."""
        from revelsMD.rdf_helpers import accumulate_binned_contributions

        values = np.array([1.0, 2.0])
        distances = np.array([0.5, 2.5])  # Second value in last bin
        bins = np.array([0.0, 1.0, 2.0, 3.0])

        result = accumulate_binned_contributions(values, distances, bins)

        np.testing.assert_almost_equal(result[0], 1.0)
        np.testing.assert_almost_equal(result[3], 0.0)  # Last bin zeroed

    def test_out_of_range_distances(self):
        """Distances outside bin range handled correctly."""
        from revelsMD.rdf_helpers import accumulate_binned_contributions

        values = np.array([1.0, 2.0, 3.0])
        distances = np.array([-0.5, 1.5, 10.0])  # -0.5 before bins, 10 after
        bins = np.array([0.0, 1.0, 2.0, 3.0])

        result = accumulate_binned_contributions(values, distances, bins)

        # Only distance 1.5 is in valid range (bin 1)
        # -0.5 goes to bin -1 (clipped to 0)
        # 10.0 goes to last bin (zeroed)
        np.testing.assert_almost_equal(result[1], 2.0)

    def test_nan_handling(self):
        """NaN values converted to zero."""
        from revelsMD.rdf_helpers import accumulate_binned_contributions

        values = np.array([1.0, np.nan, 3.0])
        distances = np.array([0.5, 1.5, 2.5])
        bins = np.array([0.0, 1.0, 2.0, 3.0])

        result = accumulate_binned_contributions(values, distances, bins)

        # Result should have no NaN values
        assert np.all(np.isfinite(result))

    def test_matches_original_loop(self):
        """Output matches the original backward loop implementation."""
        from revelsMD.rdf_helpers import accumulate_binned_contributions

        np.random.seed(42)
        n_pairs = 1000
        n_bins = 50
        values = np.random.randn(n_pairs)
        distances = np.random.uniform(0, 5, n_pairs)
        bins = np.linspace(0, 5, n_bins)

        result = accumulate_binned_contributions(values, distances, bins)

        # Compute expected using original loop logic
        digitized = np.digitize(distances, bins) - 1
        values_copy = values.copy()
        values_copy[digitized == n_bins - 1] = 0

        expected = np.zeros(n_bins, dtype=np.longdouble)
        expected[n_bins - 1] = np.sum(values_copy[digitized == n_bins - 1])
        for l in range(n_bins - 2, -1, -1):
            mask = digitized == l
            if np.any(mask):
                expected[l] = np.sum(values_copy[mask])

        np.testing.assert_array_almost_equal(result, expected)


class TestComparisonWithOriginal:
    """Compare vectorised helpers against the original implementation."""

    def test_like_pairs_matches_original(self):
        """Vectorised like-pair calculation matches original loop output."""
        from revelsMD.rdf_helpers import (
            compute_pairwise_contributions,
            accumulate_binned_contributions,
        )

        np.random.seed(42)
        ns = 50  # Number of atoms
        box_x, box_y, box_z = 10.0, 10.0, 10.0

        # Random positions and forces
        pos_ang = np.random.uniform(0, box_x, (ns, 3))
        force_total = np.random.randn(ns, 3)

        bins = np.arange(0, 5, 0.1)
        n_bins = len(bins)

        # --- Original implementation (from revels_rdf.py lines 91-134) ---
        rx = np.zeros((ns, ns))
        ry = np.zeros((ns, ns))
        rz = np.zeros((ns, ns))
        Fx = np.zeros((ns, ns))
        Fy = np.zeros((ns, ns))
        Fz = np.zeros((ns, ns))

        for x in range(ns):
            ry[x, :] = pos_ang[:, 1] - pos_ang[x, 1]
            rx[x, :] = pos_ang[:, 0] - pos_ang[x, 0]
            rz[x, :] = pos_ang[:, 2] - pos_ang[x, 2]
            Fx[x, :] = force_total[:, 0]
            Fy[x, :] = force_total[:, 1]
            Fz[x, :] = force_total[:, 2]

        # Minimum image convention
        rx -= (np.ceil((np.abs(rx) - box_x / 2) / box_x)) * (box_x) * np.sign(rx)
        ry -= (np.ceil((np.abs(ry) - box_y / 2) / box_y)) * (box_y) * np.sign(ry)
        rz -= (np.ceil((np.abs(rz) - box_z / 2) / box_z)) * (box_z) * np.sign(rz)

        r_orig = (rx * rx + ry * ry + rz * rz) ** 0.5

        with np.errstate(divide="ignore", invalid="ignore"):
            dot_prod_orig = ((Fz * rz) + (Fy * ry) + (Fx * rx)) / r_orig / r_orig / r_orig

        dot_prod_orig[(rx > box_x / 2) + (ry > box_y / 2) + (rz > box_z / 2)] = 0

        dp_orig = dot_prod_orig.reshape(-1)
        rn_orig = r_orig.reshape(-1)

        digtized_array = np.digitize(rn_orig, bins) - 1
        dp_orig_copy = dp_orig.copy()
        dp_orig_copy[digtized_array == n_bins - 1] = 0

        storage_orig = np.zeros(n_bins, dtype=np.longdouble)
        storage_orig[n_bins - 1] = np.sum(dp_orig_copy[digtized_array == n_bins - 1])
        for l in range(n_bins - 2, -1, -1):
            mask = digtized_array == l
            if np.any(mask):
                storage_orig[l] = np.sum(dp_orig_copy[mask])

        storage_orig = np.nan_to_num(storage_orig, nan=0.0, posinf=0.0, neginf=0.0)

        # --- Vectorised implementation ---
        # The unified function returns upper-triangle pairs only,
        # but accumulation should give same result because (i,j) and (j,i)
        # pairs sum to (F_j - F_i) . r_ij which is what unified computes.
        r_vec, dot_vec = compute_pairwise_contributions(
            pos_ang, pos_ang, force_total, force_total, (box_x, box_y, box_z)
        )
        storage_vec = accumulate_binned_contributions(dot_vec, r_vec, bins)

        # Compare results
        np.testing.assert_array_almost_equal(
            storage_vec, storage_orig,
            decimal=10,
            err_msg="Vectorised like-pair calculation differs from original"
        )

    def test_unlike_pairs_matches_original(self):
        """Vectorised unlike-pair calculation matches original loop output."""
        from revelsMD.rdf_helpers import (
            compute_pairwise_contributions,
            accumulate_binned_contributions,
        )

        np.random.seed(43)
        n1, n2 = 30, 40  # Different species counts
        box_x, box_y, box_z = 10.0, 12.0, 8.0

        # Random positions and forces
        pos_ang_1 = np.random.uniform(0, box_x, (n1, 3))
        pos_ang_2 = np.random.uniform(0, box_x, (n2, 3))
        force_total_1 = np.random.randn(n1, 3)
        force_total_2 = np.random.randn(n2, 3)

        bins = np.arange(0, 4, 0.1)
        n_bins = len(bins)

        # --- Original implementation (from revels_rdf.py lines 187-227) ---
        rx = np.zeros((n2, n1))
        ry = np.zeros((n2, n1))
        rz = np.zeros((n2, n1))
        Fx = np.zeros((n2, n1))
        Fy = np.zeros((n2, n1))
        Fz = np.zeros((n2, n1))

        for x in range(n2):
            ry[x, :] = pos_ang_1[:, 1] - pos_ang_2[x, 1]
            rx[x, :] = pos_ang_1[:, 0] - pos_ang_2[x, 0]
            rz[x, :] = pos_ang_1[:, 2] - pos_ang_2[x, 2]
            Fx[x, :] = force_total_1[:, 0] - force_total_2[x, 0]
            Fy[x, :] = force_total_1[:, 1] - force_total_2[x, 1]
            Fz[x, :] = force_total_1[:, 2] - force_total_2[x, 2]

        # Minimum image convention
        rx -= (np.ceil((np.abs(rx) - box_x / 2) / box_x)) * (box_x) * np.sign(rx)
        ry -= (np.ceil((np.abs(ry) - box_y / 2) / box_y)) * (box_y) * np.sign(ry)
        rz -= (np.ceil((np.abs(rz) - box_z / 2) / box_z)) * (box_z) * np.sign(rz)

        r_orig = (rx * rx + ry * ry + rz * rz) ** 0.5

        with np.errstate(divide="ignore", invalid="ignore"):
            dot_prod_orig = ((Fz * rz) + (Fy * ry) + (Fx * rx)) / r_orig / r_orig / r_orig

        dot_prod_orig[(rx > box_x / 2) + (ry > box_y / 2) + (rz > box_z / 2)] = 0

        dp_orig = dot_prod_orig.reshape(-1)
        rn_orig = r_orig.reshape(-1)

        digtized_array = np.digitize(rn_orig, bins) - 1
        dp_orig_copy = dp_orig.copy()
        dp_orig_copy[digtized_array == n_bins - 1] = 0

        storage_orig = np.zeros(n_bins, dtype=np.longdouble)
        storage_orig[n_bins - 1] = np.sum(dp_orig_copy[digtized_array == n_bins - 1])
        for l in range(n_bins - 2, -1, -1):
            mask = digtized_array == l
            if np.any(mask):
                storage_orig[l] = np.sum(dp_orig_copy[mask])

        storage_orig = np.nan_to_num(storage_orig, nan=0.0, posinf=0.0, neginf=0.0)

        # --- Vectorised implementation ---
        r_vec, dot_vec = compute_pairwise_contributions(
            pos_ang_1, pos_ang_2, force_total_1, force_total_2,
            (box_x, box_y, box_z)
        )
        storage_vec = accumulate_binned_contributions(dot_vec, r_vec, bins)

        # Compare results
        np.testing.assert_array_almost_equal(
            storage_vec, storage_orig,
            decimal=10,
            err_msg="Vectorised unlike-pair calculation differs from original"
        )

    def test_single_frame_rdf_like_full_comparison(self):
        """Full single_frame_rdf output matches helpers for like-species."""
        from revelsMD.rdf_helpers import (
            compute_pairwise_contributions,
            accumulate_binned_contributions,
        )

        np.random.seed(44)
        n_atoms = 100
        box_x, box_y, box_z = 15.0, 15.0, 15.0

        pos_array = np.random.uniform(0, box_x, (n_atoms, 3))
        force_array = np.random.randn(n_atoms, 3)
        indices = np.arange(n_atoms)
        bins = np.arange(0, 7, 0.05)

        # Using RevelsRDF.single_frame_rdf
        rdf_result = RevelsRDF.single_frame_rdf(
            pos_array, force_array, [indices, indices],
            box_x, box_y, box_z, bins
        )

        # Direct helper implementation
        pos_ang = pos_array[indices, :]
        force_total = force_array[indices, :]
        r_vec, dot_vec = compute_pairwise_contributions(
            pos_ang, pos_ang, force_total, force_total, (box_x, box_y, box_z)
        )
        helper_result = accumulate_binned_contributions(dot_vec, r_vec, bins)

        np.testing.assert_array_almost_equal(
            helper_result, rdf_result,
            decimal=10,
            err_msg="Helper-based single_frame_rdf differs from direct helper call"
        )

    def test_single_frame_rdf_unlike_full_comparison(self):
        """Full single_frame_rdf output matches helpers for unlike-species."""
        from revelsMD.rdf_helpers import (
            compute_pairwise_contributions,
            accumulate_binned_contributions,
        )

        np.random.seed(45)
        n_atoms = 120
        n_type1 = 70
        box_x, box_y, box_z = 12.0, 14.0, 10.0

        pos_array = np.random.uniform(0, min(box_x, box_y, box_z), (n_atoms, 3))
        force_array = np.random.randn(n_atoms, 3)
        indices = [np.arange(n_type1), np.arange(n_type1, n_atoms)]
        bins = np.arange(0, 5, 0.05)

        # Using RevelsRDF.single_frame_rdf
        rdf_result = RevelsRDF.single_frame_rdf(
            pos_array, force_array, indices,
            box_x, box_y, box_z, bins
        )

        # Direct helper implementation
        pos_ang_1 = pos_array[indices[0], :]
        pos_ang_2 = pos_array[indices[1], :]
        force_total_1 = force_array[indices[0], :]
        force_total_2 = force_array[indices[1], :]

        r_vec, dot_vec = compute_pairwise_contributions(
            pos_ang_1, pos_ang_2, force_total_1, force_total_2,
            (box_x, box_y, box_z)
        )
        helper_result = accumulate_binned_contributions(dot_vec, r_vec, bins)

        np.testing.assert_array_almost_equal(
            helper_result, rdf_result,
            decimal=10,
            err_msg="Helper-based single_frame_rdf differs from direct helper call"
        )


class TestPairwiseContributionsNumba:
    """Test Numba implementation of pairwise contributions."""

    @pytest.fixture(autouse=True)
    def skip_if_no_numba(self):
        """Skip tests if numba is not available."""
        pytest.importorskip('numba')

    def test_like_species_shape(self):
        """For like-species, Numba function should compute n(n-1)/2 pairs."""
        from revelsMD.rdf_helpers_numba import compute_pairwise_contributions_numba

        n = 10
        np.random.seed(44)
        pos = np.random.uniform(0, 10, (n, 3))
        forces = np.random.randn(n, 3)
        box = (10.0, 10.0, 10.0)

        r_flat, dot_flat = compute_pairwise_contributions_numba(
            pos, pos, forces, forces, box
        )

        expected_pairs = n * (n - 1) // 2
        assert r_flat.shape[0] == expected_pairs, (
            f"Expected {expected_pairs} pairs for n={n}, got {r_flat.shape[0]}"
        )

    def test_unlike_species_shape(self):
        """For unlike-species, Numba function computes all n1*n2 pairs."""
        from revelsMD.rdf_helpers_numba import compute_pairwise_contributions_numba

        n1, n2 = 10, 15
        np.random.seed(45)
        pos_a = np.random.uniform(0, 10, (n1, 3))
        pos_b = np.random.uniform(0, 10, (n2, 3))
        forces_a = np.random.randn(n1, 3)
        forces_b = np.random.randn(n2, 3)
        box = (10.0, 10.0, 10.0)

        r_flat, dot_flat = compute_pairwise_contributions_numba(
            pos_a, pos_b, forces_a, forces_b, box
        )

        expected_pairs = n1 * n2
        assert r_flat.shape[0] == expected_pairs, (
            f"Expected {expected_pairs} pairs, got {r_flat.shape[0]}"
        )

    def test_matches_numpy_implementation(self):
        """Numba function should match NumPy function."""
        from revelsMD.rdf_helpers import compute_pairwise_contributions
        from revelsMD.rdf_helpers_numba import compute_pairwise_contributions_numba

        np.random.seed(46)
        pos_a = np.random.uniform(0, 10, (25, 3))
        pos_b = np.random.uniform(0, 10, (30, 3))
        forces_a = np.random.randn(25, 3)
        forces_b = np.random.randn(30, 3)
        box = (10.0, 10.0, 10.0)

        # NumPy
        r_numpy, dot_numpy = compute_pairwise_contributions(
            pos_a, pos_b, forces_a, forces_b, box
        )

        # Numba
        r_numba, dot_numba = compute_pairwise_contributions_numba(
            pos_a, pos_b, forces_a, forces_b, box
        )

        np.testing.assert_allclose(
            r_numpy, r_numba, rtol=1e-14,
            err_msg="Numba distances don't match NumPy"
        )
        np.testing.assert_allclose(
            dot_numpy, dot_numba, rtol=1e-14,
            err_msg="Numba dot products don't match NumPy"
        )

    def test_like_species_matches_numpy(self):
        """Numba like-species result should match NumPy."""
        from revelsMD.rdf_helpers import compute_pairwise_contributions
        from revelsMD.rdf_helpers_numba import (
            compute_pairwise_contributions_numba,
            accumulate_binned_contributions_numba,
        )
        from revelsMD.rdf_helpers import accumulate_binned_contributions

        np.random.seed(42)
        pos = np.random.uniform(0, 10, (50, 3))
        forces = np.random.randn(50, 3)
        box = (10.0, 10.0, 10.0)
        bins = np.arange(0, 5, 0.1)

        # NumPy
        r_np, dot_np = compute_pairwise_contributions(pos, pos, forces, forces, box)
        acc_np = accumulate_binned_contributions(dot_np, r_np, bins)

        # Numba
        r_nb, dot_nb = compute_pairwise_contributions_numba(pos, pos, forces, forces, box)
        acc_nb = accumulate_binned_contributions_numba(dot_nb, r_nb, bins)

        np.testing.assert_allclose(
            acc_np, acc_nb, rtol=1e-10,
            err_msg="Numba like-species accumulated contributions don't match NumPy"
        )
