"""Tests for revelsMD.cell coordinate transforms and cell geometry."""

import numpy as np
import pytest

from revelsMD.cell import (
    apply_minimum_image,
    apply_minimum_image_orthorhombic,
    cartesian_to_fractional,
    cells_are_compatible,
    fractional_to_cartesian,
    inscribed_sphere_radius,
    is_orthorhombic,
    wrap_fractional,
    ORTHORHOMBIC_TOLERANCE,
)


class TestIsOrthorhombic:
    """Tests for is_orthorhombic()."""

    def test_diagonal_matrix_is_orthorhombic(self):
        cell = np.diag([10.0, 8.0, 6.0])
        assert is_orthorhombic(cell) is True

    def test_identity_is_orthorhombic(self):
        assert is_orthorhombic(np.eye(3)) is True

    def test_triclinic_cell_is_not_orthorhombic(self):
        cell = np.array([
            [10.0, 0.0, 0.0],
            [3.0, 9.0, 0.0],
            [0.0, 0.0, 8.0],
        ])
        assert is_orthorhombic(cell) is False

    def test_small_off_diagonal_within_tolerance_is_orthorhombic(self):
        cell = np.diag([10.0, 8.0, 6.0])
        cell[0, 1] = ORTHORHOMBIC_TOLERANCE / 10  # well within tolerance
        assert is_orthorhombic(cell) is True

    def test_off_diagonal_at_tolerance_boundary_is_not_orthorhombic(self):
        cell = np.diag([10.0, 8.0, 6.0])
        cell[0, 1] = ORTHORHOMBIC_TOLERANCE * 10  # above tolerance
        assert is_orthorhombic(cell) is False

    def test_custom_tolerance(self):
        cell = np.diag([10.0, 8.0, 6.0])
        cell[1, 2] = 0.1
        assert is_orthorhombic(cell, atol=0.01) is False
        assert is_orthorhombic(cell, atol=1.0) is True


class TestCoordinateTransforms:
    """Tests for cartesian_to_fractional and fractional_to_cartesian."""

    def test_round_trip_orthorhombic(self):
        cell = np.diag([10.0, 8.0, 6.0])
        cell_inv = np.linalg.inv(cell)
        positions = np.array([[5.0, 4.0, 3.0], [1.0, 2.0, 5.5]])

        fractional = cartesian_to_fractional(positions, cell_inv)
        recovered = fractional_to_cartesian(fractional, cell)

        np.testing.assert_allclose(recovered, positions)

    def test_round_trip_triclinic(self):
        cell = np.array([
            [10.0, 0.0, 0.0],
            [3.0, 9.0, 0.0],
            [1.0, 2.0, 8.0],
        ])
        cell_inv = np.linalg.inv(cell)
        positions = np.array([[7.0, 3.5, 2.0], [0.5, 8.0, 6.5]])

        fractional = cartesian_to_fractional(positions, cell_inv)
        recovered = fractional_to_cartesian(fractional, cell)

        np.testing.assert_allclose(recovered, positions)

    def test_fractional_values_orthorhombic(self):
        cell = np.diag([10.0, 8.0, 6.0])
        cell_inv = np.linalg.inv(cell)
        positions = np.array([[5.0, 4.0, 3.0]])

        fractional = cartesian_to_fractional(positions, cell_inv)
        np.testing.assert_allclose(fractional, [[0.5, 0.5, 0.5]])

    def test_origin_maps_to_zero_fractional(self):
        cell = np.array([
            [10.0, 0.0, 0.0],
            [3.0, 9.0, 0.0],
            [0.0, 0.0, 8.0],
        ])
        cell_inv = np.linalg.inv(cell)
        origin = np.array([[0.0, 0.0, 0.0]])

        fractional = cartesian_to_fractional(origin, cell_inv)
        np.testing.assert_allclose(fractional, [[0.0, 0.0, 0.0]], atol=1e-15)

    def test_lattice_vector_maps_to_unit_fractional(self):
        cell = np.array([
            [10.0, 0.0, 0.0],
            [3.0, 9.0, 0.0],
            [1.0, 2.0, 8.0],
        ])
        cell_inv = np.linalg.inv(cell)

        # Each lattice vector should map to (1,0,0), (0,1,0), (0,0,1)
        for i in range(3):
            vec = cell[i:i+1, :]  # shape (1, 3)
            frac = cartesian_to_fractional(vec, cell_inv)
            expected = np.zeros((1, 3))
            expected[0, i] = 1.0
            np.testing.assert_allclose(frac, expected, atol=1e-14)


class TestWrapFractional:
    """Tests for wrap_fractional()."""

    def test_already_in_range(self):
        frac = np.array([[0.25, 0.5, 0.75]])
        result = wrap_fractional(frac)
        np.testing.assert_allclose(result, [[0.25, 0.5, 0.75]])

    def test_negative_values_wrapped(self):
        frac = np.array([[-0.1, -0.5, -1.3]])
        result = wrap_fractional(frac)
        np.testing.assert_allclose(result, [[0.9, 0.5, 0.7]])

    def test_values_above_one_wrapped(self):
        frac = np.array([[1.2, 2.7, 3.0]])
        result = wrap_fractional(frac)
        np.testing.assert_allclose(result, [[0.2, 0.7, 0.0]])

    def test_exactly_one_maps_to_zero(self):
        frac = np.array([[1.0, 0.0, 0.0]])
        result = wrap_fractional(frac)
        np.testing.assert_allclose(result, [[0.0, 0.0, 0.0]])

    def test_exactly_zero_stays_zero(self):
        frac = np.array([[0.0, 0.0, 0.0]])
        result = wrap_fractional(frac)
        np.testing.assert_allclose(result, [[0.0, 0.0, 0.0]])

    def test_multiple_positions(self):
        frac = np.array([
            [0.5, -0.2, 1.1],
            [2.0, 0.0, -0.5],
        ])
        result = wrap_fractional(frac)
        np.testing.assert_allclose(result, [
            [0.5, 0.8, 0.1],
            [0.0, 0.0, 0.5],
        ])


class TestApplyMinimumImage:
    """Tests for apply_minimum_image (general triclinic MIC)."""

    def test_small_displacement_unchanged(self):
        """A displacement well within half the cell should be unchanged."""
        cell = np.diag([10.0, 8.0, 6.0])
        cell_inv = np.linalg.inv(cell)
        disp = np.array([[1.0, 2.0, -1.0]])

        result = apply_minimum_image(disp, cell, cell_inv)
        np.testing.assert_allclose(result, [[1.0, 2.0, -1.0]])

    def test_displacement_across_orthorhombic_boundary(self):
        """Displacement > half the box should be wrapped to nearest image."""
        cell = np.diag([10.0, 10.0, 10.0])
        cell_inv = np.linalg.inv(cell)
        # Displacement of 7.0 in x: nearest image is 7.0 - 10.0 = -3.0
        disp = np.array([[7.0, 0.0, 0.0]])

        result = apply_minimum_image(disp, cell, cell_inv)
        np.testing.assert_allclose(result, [[-3.0, 0.0, 0.0]])

    def test_triclinic_cell_known_displacement(self):
        """Hand-calculated MIC for a triclinic cell.

        Cell: a = (10, 0, 0), b = (3, 9, 0), c = (0, 0, 8)
        Displacement: (8, 1, 0) in Cartesian.

        Fractional: s = (8, 1, 0) @ inv(M)
          inv(M) = [[0.1, 0, 0], [-1/30, 1/9, 0], [0, 0, 0.125]]
          s = (0.8 - 1/30, 1/9, 0) = (23/30, 1/9, 0)
        round(s) = (1, 0, 0)
        s - round(s) = (-7/30, 1/9, 0)
        Back to Cartesian: (-7/30, 1/9, 0) @ M
          = (-7/30*10 + 1/9*3, 1/9*9, 0)
          = (-7/3 + 1/3, 1, 0)
          = (-2, 1, 0)
        """
        cell = np.array([
            [10.0, 0.0, 0.0],
            [3.0, 9.0, 0.0],
            [0.0, 0.0, 8.0],
        ])
        cell_inv = np.linalg.inv(cell)
        disp = np.array([[8.0, 1.0, 0.0]])

        result = apply_minimum_image(disp, cell, cell_inv)

        np.testing.assert_allclose(result, [[-2.0, 1.0, 0.0]], atol=1e-14)

    def test_agrees_with_orthorhombic_for_diagonal_cell(self):
        """General MIC should agree with orthorhombic MIC for diagonal cells."""
        cell = np.diag([10.0, 8.0, 6.0])
        cell_inv = np.linalg.inv(cell)
        box = np.array([10.0, 8.0, 6.0])

        displacements = np.array([
            [6.0, -5.0, 3.5],
            [-1.0, 2.0, -4.0],
            [0.0, 4.0, 0.0],
        ])

        result_general = apply_minimum_image(displacements, cell, cell_inv)
        result_ortho = apply_minimum_image_orthorhombic(displacements, box)

        np.testing.assert_allclose(result_general, result_ortho, atol=1e-12)


class TestApplyMinimumImageOrthorhombic:
    """Tests for apply_minimum_image_orthorhombic."""

    def test_small_displacement_unchanged(self):
        box = np.array([10.0, 10.0, 10.0])
        disp = np.array([[1.0, -2.0, 3.0]])

        result = apply_minimum_image_orthorhombic(disp, box)
        np.testing.assert_allclose(result, [[1.0, -2.0, 3.0]])

    def test_wraps_to_nearest_image(self):
        box = np.array([10.0, 10.0, 10.0])
        disp = np.array([[7.0, -8.0, 0.0]])

        result = apply_minimum_image_orthorhombic(disp, box)
        np.testing.assert_allclose(result, [[-3.0, 2.0, 0.0]])

    def test_matches_existing_rdf_helpers(self):
        """Should produce same results as revelsMD.rdf.rdf_helpers.apply_minimum_image."""
        from revelsMD.rdf.rdf_helpers import apply_minimum_image as rdf_mic

        box = np.array([10.0, 8.0, 6.0])
        displacements = np.array([
            [6.0, -5.0, 3.5],
            [-1.0, 2.0, -4.0],
            [0.0, 4.001, 0.0],
            [5.0, -4.0, 3.0],
        ])

        result_cell = apply_minimum_image_orthorhombic(displacements, box)
        result_rdf = rdf_mic(displacements, box)

        np.testing.assert_allclose(result_cell, result_rdf)


class TestInscribedSphereRadius:
    """Tests for inscribed_sphere_radius()."""

    def test_orthorhombic_cell(self):
        """For orthorhombic (10, 8, 6), rmax = min(10, 8, 6) / 2 = 3.0."""
        cell = np.diag([10.0, 8.0, 6.0])
        assert inscribed_sphere_radius(cell) == pytest.approx(3.0)

    def test_cubic_cell(self):
        """For cubic cell with side 5, rmax = 5 / 2 = 2.5."""
        cell = np.diag([5.0, 5.0, 5.0])
        assert inscribed_sphere_radius(cell) == pytest.approx(2.5)

    def test_hexagonal_cell(self):
        """Hexagonal cell: a=(10,0,0), b=(5, 5*sqrt(3), 0), c=(0,0,8).

        V = |det(M)| = 10 * 5*sqrt(3) * 8 = 400*sqrt(3)
        |b x c| = |(5*sqrt(3)*8, -5*8, 0)| = |(40*sqrt(3), -40, 0)| = 80
        |c x a| = |(0, 0, 80)| -> wait, need to compute properly.

        a x b = (0, 0, 10*5*sqrt(3) - 0) = (0, 0, 50*sqrt(3))
        |a x b| = 50*sqrt(3)
        h_ab = V / |a x b| = 400*sqrt(3) / (50*sqrt(3)) = 8

        b x c = (5*sqrt(3)*8 - 0, 0 - 10*0*0, ...) -- let me just use cross.
        Actually: b=(5, 5*sqrt(3), 0), c=(0, 0, 8)
        b x c = (5*sqrt(3)*8, -5*8, 0) = (40*sqrt(3), -40, 0)
        |b x c| = sqrt(4800 + 1600) = sqrt(6400) = 80
        h_bc = 400*sqrt(3) / 80 = 5*sqrt(3) ~ 8.66

        c x a = (0, 0, 8) x (10, 0, 0) = (0*0-8*0, 8*10-0*0, 0*0-0*10)
             = (0, 80, 0)
        |c x a| = 80
        h_ca = 400*sqrt(3) / 80 = 5*sqrt(3) ~ 8.66

        rmax = min(8.66, 8.66, 8) / 2 = 4.0
        """
        cell = np.array([
            [10.0, 0.0, 0.0],
            [5.0, 5.0 * np.sqrt(3), 0.0],
            [0.0, 0.0, 8.0],
        ])
        assert inscribed_sphere_radius(cell) == pytest.approx(4.0)

    def test_identity_cell(self):
        """Unit cell: rmax = 0.5."""
        assert inscribed_sphere_radius(np.eye(3)) == pytest.approx(0.5)


class TestCellsAreCompatible:
    """Tests for cells_are_compatible()."""

    def test_identical_cells(self):
        cell = np.diag([10.0, 8.0, 6.0])
        assert cells_are_compatible(cell, cell.copy()) is True

    def test_cells_within_tolerance(self):
        cell_a = np.diag([10.0, 8.0, 6.0])
        cell_b = cell_a + 1e-8
        assert cells_are_compatible(cell_a, cell_b) is True

    def test_cells_outside_tolerance(self):
        cell_a = np.diag([10.0, 8.0, 6.0])
        cell_b = np.diag([10.0, 8.0, 6.1])
        assert cells_are_compatible(cell_a, cell_b) is False

    def test_triclinic_cells_match(self):
        cell = np.array([
            [10.0, 0.0, 0.0],
            [3.0, 9.0, 0.0],
            [0.0, 0.0, 8.0],
        ])
        assert cells_are_compatible(cell, cell.copy()) is True

    def test_custom_tolerance(self):
        cell_a = np.diag([10.0, 8.0, 6.0])
        cell_b = np.diag([10.0, 8.0, 6.01])
        assert cells_are_compatible(cell_a, cell_b, atol=0.001) is False
        assert cells_are_compatible(cell_a, cell_b, atol=0.1) is True
