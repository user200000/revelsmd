"""Tests for revelsMD.cell coordinate transforms and cell geometry."""

import numpy as np
import pytest

from revelsMD.cell import is_orthorhombic, ORTHORHOMBIC_TOLERANCE


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
