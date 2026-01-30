"""Tests for revelsMD.density package and its module structure."""

import pytest
import warnings
import numpy as np


# ---------------------------------------------------------------------------
# Mock trajectory for SelectionState tests
# ---------------------------------------------------------------------------

class MockTrajectory:
    """Mock trajectory for testing SelectionState methods."""

    def __init__(self):
        self.box_x = self.box_y = self.box_z = 10.0

    def get_indices(self, atom_name):
        # 3 molecules, each with atoms O, H1, H2
        indices = {
            'O': np.array([0, 3, 6]),
            'H1': np.array([1, 4, 7]),
            'H2': np.array([2, 5, 8]),
        }
        return indices[atom_name]

    def get_masses(self, atom_name):
        masses = {
            'O': np.array([16.0, 16.0, 16.0]),
            'H1': np.array([1.0, 1.0, 1.0]),
            'H2': np.array([1.0, 1.0, 1.0]),
        }
        return masses[atom_name]

    def get_charges(self, atom_name):
        charges = {
            'O': np.array([-0.8, -0.8, -0.8]),
            'H1': np.array([0.4, 0.4, 0.4]),
            'H2': np.array([0.4, 0.4, 0.4]),
        }
        return charges[atom_name]


# ---------------------------------------------------------------------------
# SelectionState.get_positions() tests
# ---------------------------------------------------------------------------

class TestSelectionStateGetWeights:
    """Tests for SelectionState.get_weights() method."""

    @pytest.fixture
    def trajectory(self):
        return MockTrajectory()

    @pytest.fixture
    def positions(self):
        """9 atoms: 3 water molecules arranged along x-axis."""
        return np.array([
            # Molecule 0: O at x=0, H1 at x=1, H2 at x=-1
            [0.0, 5.0, 5.0],   # O (index 0)
            [1.0, 5.0, 5.0],   # H1 (index 1)
            [-1.0, 5.0, 5.0],  # H2 (index 2)
            # Molecule 1: O at x=3, H1 at x=4, H2 at x=2
            [3.0, 5.0, 5.0],   # O (index 3)
            [4.0, 5.0, 5.0],   # H1 (index 4)
            [2.0, 5.0, 5.0],   # H2 (index 5)
            # Molecule 2: O at x=6, H1 at x=7, H2 at x=5
            [6.0, 5.0, 5.0],   # O (index 6)
            [7.0, 5.0, 5.0],   # H1 (index 7)
            [5.0, 5.0, 5.0],   # H2 (index 8)
        ], dtype=float)

    def test_number_density_returns_one(self, trajectory):
        """Number density should return weight of 1.0."""
        from revelsMD.density import SelectionState

        ss = SelectionState(trajectory, 'O', centre_location=True, rigid=False)
        ss.density_type = 'number'
        result = ss.get_weights()

        assert result == 1.0

    def test_charge_single_species_returns_charges(self, trajectory):
        """Charge density for single species should return charge array."""
        from revelsMD.density import SelectionState

        ss = SelectionState(trajectory, 'O', centre_location=True, rigid=False)
        ss.density_type = 'charge'
        result = ss.get_weights()

        expected = np.array([-0.8, -0.8, -0.8])  # O charges
        np.testing.assert_array_equal(result, expected)

    def test_charge_multi_species_not_rigid_returns_list(self, trajectory):
        """Charge density for multi-species non-rigid should return list of charge arrays."""
        from revelsMD.density import SelectionState

        ss = SelectionState(trajectory, ['O', 'H1', 'H2'], centre_location=True, rigid=False)
        ss.density_type = 'charge'
        result = ss.get_weights()

        assert isinstance(result, list)
        assert len(result) == 3
        np.testing.assert_array_equal(result[0], np.array([-0.8, -0.8, -0.8]))  # O
        np.testing.assert_array_equal(result[1], np.array([0.4, 0.4, 0.4]))    # H1
        np.testing.assert_array_equal(result[2], np.array([0.4, 0.4, 0.4]))    # H2

    def test_charge_rigid_returns_summed_charges(self, trajectory):
        """Charge density for rigid should return total charge per molecule."""
        from revelsMD.density import SelectionState

        ss = SelectionState(trajectory, ['O', 'H1', 'H2'], centre_location=True, rigid=True)
        ss.density_type = 'charge'
        result = ss.get_weights()

        # Total charge per molecule: -0.8 + 0.4 + 0.4 = 0.0
        expected = np.array([0.0, 0.0, 0.0])
        np.testing.assert_allclose(result, expected)

    def test_polarisation_returns_dipole_projection(self, trajectory, positions):
        """Polarisation density should return dipole projected along axis."""
        from revelsMD.density import SelectionState

        ss = SelectionState(trajectory, ['O', 'H1', 'H2'], centre_location=True, rigid=True)
        ss.density_type = 'polarisation'
        ss.polarisation_axis = 0  # x-axis

        result = ss.get_weights(positions)

        # COM for each molecule is at x = (16*x_O + 1*x_H1 + 1*x_H2) / 18
        # Molecule 0: COM_x = (16*0 + 1*1 + 1*(-1)) / 18 = 0
        # Dipole_x = q_O*(x_O - COM_x) + q_H1*(x_H1 - COM_x) + q_H2*(x_H2 - COM_x)
        #          = -0.8*(0 - 0) + 0.4*(1 - 0) + 0.4*(-1 - 0)
        #          = 0 + 0.4 - 0.4 = 0.0
        # Similarly for molecules 1 and 2
        expected = np.array([0.0, 0.0, 0.0])
        np.testing.assert_allclose(result, expected, atol=1e-10)


class TestSelectionStateGetForces:
    """Tests for SelectionState.get_forces() method."""

    @pytest.fixture
    def trajectory(self):
        return MockTrajectory()

    @pytest.fixture
    def forces(self):
        """Forces for 9 atoms (3 water molecules)."""
        return np.array([
            # Molecule 0
            [1.0, 0.0, 0.0],   # O (index 0)
            [0.5, 0.0, 0.0],   # H1 (index 1)
            [0.5, 0.0, 0.0],   # H2 (index 2)
            # Molecule 1
            [2.0, 0.0, 0.0],   # O (index 3)
            [1.0, 0.0, 0.0],   # H1 (index 4)
            [1.0, 0.0, 0.0],   # H2 (index 5)
            # Molecule 2
            [3.0, 0.0, 0.0],   # O (index 6)
            [1.5, 0.0, 0.0],   # H1 (index 7)
            [1.5, 0.0, 0.0],   # H2 (index 8)
        ], dtype=float)

    def test_single_species_returns_forces_at_indices(self, trajectory, forces):
        """Single species should return forces at selected indices."""
        from revelsMD.density import SelectionState

        ss = SelectionState(trajectory, 'O', centre_location=True, rigid=False)
        result = ss.get_forces(forces)

        expected = forces[[0, 3, 6], :]  # O atoms
        np.testing.assert_array_equal(result, expected)

    def test_multi_species_not_rigid_returns_list(self, trajectory, forces):
        """Multi-species, non-rigid should return list of force arrays."""
        from revelsMD.density import SelectionState

        ss = SelectionState(trajectory, ['O', 'H1', 'H2'], centre_location=True, rigid=False)
        result = ss.get_forces(forces)

        assert isinstance(result, list)
        assert len(result) == 3
        np.testing.assert_array_equal(result[0], forces[[0, 3, 6], :])  # O
        np.testing.assert_array_equal(result[1], forces[[1, 4, 7], :])  # H1
        np.testing.assert_array_equal(result[2], forces[[2, 5, 8], :])  # H2

    def test_rigid_sums_forces_across_molecule(self, trajectory, forces):
        """Rigid molecule should sum forces across all atoms in molecule."""
        from revelsMD.density import SelectionState

        ss = SelectionState(trajectory, ['O', 'H1', 'H2'], centre_location=True, rigid=True)
        result = ss.get_forces(forces)

        # Sum forces for each molecule
        # Molecule 0: [1,0,0] + [0.5,0,0] + [0.5,0,0] = [2,0,0]
        # Molecule 1: [2,0,0] + [1,0,0] + [1,0,0] = [4,0,0]
        # Molecule 2: [3,0,0] + [1.5,0,0] + [1.5,0,0] = [6,0,0]
        assert result.shape == (3, 3)
        np.testing.assert_allclose(result[:, 0], [2.0, 4.0, 6.0])
        np.testing.assert_allclose(result[:, 1], [0.0, 0.0, 0.0])


class TestSelectionStateGetPositions:
    """Tests for SelectionState.get_positions() method."""

    @pytest.fixture
    def trajectory(self):
        return MockTrajectory()

    @pytest.fixture
    def positions(self):
        """9 atoms: 3 water molecules arranged along x-axis."""
        return np.array([
            # Molecule 0: O at x=0, H1 at x=0.5, H2 at x=-0.5
            [0.0, 5.0, 5.0],   # O (index 0)
            [0.5, 5.0, 5.0],   # H1 (index 1)
            [-0.5, 5.0, 5.0],  # H2 (index 2)
            # Molecule 1: O at x=3, H1 at x=3.5, H2 at x=2.5
            [3.0, 5.0, 5.0],   # O (index 3)
            [3.5, 5.0, 5.0],   # H1 (index 4)
            [2.5, 5.0, 5.0],   # H2 (index 5)
            # Molecule 2: O at x=6, H1 at x=6.5, H2 at x=5.5
            [6.0, 5.0, 5.0],   # O (index 6)
            [6.5, 5.0, 5.0],   # H1 (index 7)
            [5.5, 5.0, 5.0],   # H2 (index 8)
        ], dtype=float)

    def test_single_species_returns_positions_at_indices(self, trajectory, positions):
        """Single species should return positions at selected indices."""
        from revelsMD.density import SelectionState

        ss = SelectionState(trajectory, 'O', centre_location=True, rigid=False)
        result = ss.get_positions(positions)

        expected = positions[[0, 3, 6], :]  # O atoms
        np.testing.assert_array_equal(result, expected)

    def test_multi_species_not_rigid_returns_list(self, trajectory, positions):
        """Multi-species, non-rigid should return list of position arrays."""
        from revelsMD.density import SelectionState

        ss = SelectionState(trajectory, ['O', 'H1', 'H2'], centre_location=True, rigid=False)
        result = ss.get_positions(positions)

        assert isinstance(result, list)
        assert len(result) == 3
        np.testing.assert_array_equal(result[0], positions[[0, 3, 6], :])  # O
        np.testing.assert_array_equal(result[1], positions[[1, 4, 7], :])  # H1
        np.testing.assert_array_equal(result[2], positions[[2, 5, 8], :])  # H2

    def test_rigid_com_returns_center_of_mass(self, trajectory, positions):
        """Rigid molecule with COM should return mass-weighted center positions."""
        from revelsMD.density import SelectionState

        ss = SelectionState(trajectory, ['O', 'H1', 'H2'], centre_location=True, rigid=True)
        result = ss.get_positions(positions)

        # COM for each molecule (masses: O=16, H1=1, H2=1, total=18)
        # Molecule 0: (16*0 + 1*0.5 + 1*(-0.5)) / 18 = 0/18 = 0.0
        # Molecule 1: (16*3 + 1*3.5 + 1*2.5) / 18 = 54/18 = 3.0
        # Molecule 2: (16*6 + 1*6.5 + 1*5.5) / 18 = 108/18 = 6.0
        assert result.shape == (3, 3)
        np.testing.assert_allclose(result[:, 0], [0.0, 3.0, 6.0], rtol=1e-10)
        np.testing.assert_allclose(result[:, 1], [5.0, 5.0, 5.0], rtol=1e-10)

    def test_rigid_specific_atom_returns_that_atoms_positions(self, trajectory, positions):
        """Rigid molecule with specific atom index should return that atom's positions."""
        from revelsMD.density import SelectionState

        # centre_location=1 means use H1 positions
        ss = SelectionState(trajectory, ['O', 'H1', 'H2'], centre_location=1, rigid=True)
        result = ss.get_positions(positions)

        expected = positions[[1, 4, 7], :]  # H1 atoms
        np.testing.assert_array_equal(result, expected)


# ---------------------------------------------------------------------------
# Import tests
# ---------------------------------------------------------------------------

def test_selectionstate_importable_from_density():
    """SelectionState should be importable from revelsMD.density."""
    from revelsMD.density import SelectionState
    assert SelectionState is not None


def test_selectionstate_importable_from_submodule():
    """SelectionState should be importable from revelsMD.density.selection_state."""
    from revelsMD.density.selection_state import SelectionState
    assert SelectionState is not None


def test_selectionstate_backward_compatible_via_revels3d():
    """Revels3D.SelectionState should still work but emit deprecation warning."""
    from revelsMD.revels_3D import Revels3D
    from revelsMD.density import SelectionState
    with pytest.warns(DeprecationWarning, match="Revels3D.SelectionState is deprecated"):
        assert Revels3D.SelectionState is SelectionState


def test_helperfunctions_importable_from_density():
    """HelperFunctions should be importable from revelsMD.density."""
    from revelsMD.density import HelperFunctions
    assert HelperFunctions is not None


def test_helperfunctions_importable_from_submodule():
    """HelperFunctions should be importable from revelsMD.density.helper_functions."""
    from revelsMD.density.helper_functions import HelperFunctions
    assert HelperFunctions is not None


def test_helperfunctions_backward_compatible_via_revels3d():
    """Revels3D.HelperFunctions should still work but emit deprecation warning."""
    from revelsMD.revels_3D import Revels3D
    from revelsMD.density import HelperFunctions
    with pytest.warns(DeprecationWarning, match="Revels3D.HelperFunctions is deprecated"):
        assert Revels3D.HelperFunctions is HelperFunctions


def test_estimators_importable_from_density():
    """Estimators should be importable from revelsMD.density."""
    from revelsMD.density import Estimators
    assert Estimators is not None


def test_estimators_importable_from_submodule():
    """Estimators should be importable from revelsMD.density.estimators."""
    from revelsMD.density.estimators import Estimators
    assert Estimators is not None


def test_estimators_backward_compatible_via_revels3d():
    """Revels3D.Estimators should still work but emit deprecation warning."""
    from revelsMD.revels_3D import Revels3D
    from revelsMD.density import Estimators
    with pytest.warns(DeprecationWarning, match="Revels3D.Estimators is deprecated"):
        assert Revels3D.Estimators is Estimators


def test_gridstate_importable_from_density():
    """GridState should be importable from revelsMD.density."""
    from revelsMD.density import GridState
    assert GridState is not None


def test_gridstate_importable_from_submodule():
    """GridState should be importable from revelsMD.density.grid_state."""
    from revelsMD.density.grid_state import GridState
    assert GridState is not None


def test_gridstate_backward_compatible_via_revels3d():
    """Revels3D.GridState should still work but emit deprecation warning."""
    from revelsMD.revels_3D import Revels3D
    from revelsMD.density import GridState
    with pytest.warns(DeprecationWarning, match="Revels3D.GridState is deprecated"):
        assert Revels3D.GridState is GridState
