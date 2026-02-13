"""Tests for revelsMD.cell coordinate transforms and cell geometry."""

import numpy as np
import pytest

from revelsMD.cell import (
    cartesian_to_fractional,
    fractional_to_cartesian,
    is_orthorhombic,
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
