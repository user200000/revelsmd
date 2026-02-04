"""Tests for revelsMD.density package and its module structure."""

import numpy as np
import pytest

from revelsMD.density import DensityGrid, Selection


# ---------------------------------------------------------------------------
# Mock trajectory for Selection tests
# ---------------------------------------------------------------------------

class MockTrajectory:
    """Mock trajectory for testing Selection methods."""

    def __init__(self):
        self.box_x = self.box_y = self.box_z = 10.0
        self.units = 'real'
        self.beta = 1.0 / (300.0 * 0.0019872041)  # 1/(kB*T) for T=300K in real units

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
# Selection.get_positions() tests
# ---------------------------------------------------------------------------

class TestDeposit:
    """Tests for DensityGrid.deposit with Selection."""

    @pytest.fixture
    def trajectory(self):
        return MockTrajectory()

    @pytest.fixture
    def positions(self):
        """9 atoms: 3 water molecules."""
        return np.array([
            [1.0, 5.0, 5.0],   # O (index 0)
            [1.5, 5.0, 5.0],   # H1 (index 1)
            [0.5, 5.0, 5.0],   # H2 (index 2)
            [4.0, 5.0, 5.0],   # O (index 3)
            [4.5, 5.0, 5.0],   # H1 (index 4)
            [3.5, 5.0, 5.0],   # H2 (index 5)
            [7.0, 5.0, 5.0],   # O (index 6)
            [7.5, 5.0, 5.0],   # H1 (index 7)
            [6.5, 5.0, 5.0],   # H2 (index 8)
        ], dtype=float)

    @pytest.fixture
    def forces(self):
        """Forces for 9 atoms."""
        return np.array([
            [1.0, 0.1, 0.0],   # O
            [0.5, 0.05, 0.0],  # H1
            [0.5, 0.05, 0.0],  # H2
            [2.0, 0.2, 0.0],   # O
            [1.0, 0.1, 0.0],   # H1
            [1.0, 0.1, 0.0],   # H2
            [3.0, 0.3, 0.0],   # O
            [1.5, 0.15, 0.0],  # H1
            [1.5, 0.15, 0.0],  # H2
        ], dtype=float)

    def test_deposit_single_species_number_density(self, trajectory, positions, forces):
        """deposit with single species populates grid."""

        gs = DensityGrid(trajectory, "number", nbins=5)
        ss = Selection(trajectory, 'O', centre_location=True, rigid=False, density_type='number')
        gs.deposit(ss.get_positions(positions), ss.get_forces(forces), ss.get_weights(), kernel="triangular")

        assert np.any(gs.forceX != 0)
        assert np.any(gs.counter != 0)
        assert gs.count == 1

    def test_deposit_multi_species_non_rigid(self, trajectory, positions, forces):
        """deposit with multi-species non-rigid deposits each species."""

        gs = DensityGrid(trajectory, "number", nbins=5)
        ss = Selection(trajectory, ['O', 'H1', 'H2'], centre_location=True, rigid=False, density_type='number')
        gs.deposit(ss.get_positions(positions), ss.get_forces(forces), ss.get_weights(), kernel="triangular")

        # 3 species deposited = 3 calls to _process_frame
        assert gs.count == 3
        assert np.any(gs.counter != 0)

    def test_deposit_rigid_com_number_density(self, trajectory, positions, forces):
        """deposit with rigid molecule at COM populates grid."""

        gs = DensityGrid(trajectory, "number", nbins=5)
        ss = Selection(trajectory, ['O', 'H1', 'H2'], centre_location=True, rigid=True, density_type='number')
        gs.deposit(ss.get_positions(positions), ss.get_forces(forces), ss.get_weights(), kernel="triangular")

        # Rigid deposits once per molecule group
        assert gs.count == 1
        assert np.any(gs.counter != 0)
        assert np.any(gs.forceX != 0)

    def test_deposit_charge_density_single_species(self, trajectory, positions, forces):
        """deposit with charge density uses charge weights."""

        gs = DensityGrid(trajectory, "charge", nbins=5)
        ss = Selection(trajectory, 'O', centre_location=True, rigid=False, density_type='charge')
        gs.deposit(ss.get_positions(positions), ss.get_forces(forces), ss.get_weights(), kernel="triangular")

        # O has negative charge, so counter contributions are negative
        assert np.any(gs.counter != 0)
        assert gs.count == 1


class TestMakeForceGridUnified:
    """Test that accumulate using unified approach gives same results."""

    @pytest.fixture
    def trajectory(self):
        """Create mock trajectory that supports iteration."""
        class IterableMockTrajectory(MockTrajectory):
            def __init__(self):
                super().__init__()
                self.frames = 2
                self._positions = [
                    np.array([
                        [1.0, 5.0, 5.0], [1.5, 5.0, 5.0], [0.5, 5.0, 5.0],
                        [4.0, 5.0, 5.0], [4.5, 5.0, 5.0], [3.5, 5.0, 5.0],
                        [7.0, 5.0, 5.0], [7.5, 5.0, 5.0], [6.5, 5.0, 5.0],
                    ], dtype=float),
                    np.array([
                        [1.1, 5.1, 5.0], [1.6, 5.1, 5.0], [0.6, 5.1, 5.0],
                        [4.1, 5.1, 5.0], [4.6, 5.1, 5.0], [3.6, 5.1, 5.0],
                        [7.1, 5.1, 5.0], [7.6, 5.1, 5.0], [6.6, 5.1, 5.0],
                    ], dtype=float),
                ]
                self._forces = [
                    np.array([
                        [1.0, 0.1, 0.0], [0.5, 0.05, 0.0], [0.5, 0.05, 0.0],
                        [2.0, 0.2, 0.0], [1.0, 0.1, 0.0], [1.0, 0.1, 0.0],
                        [3.0, 0.3, 0.0], [1.5, 0.15, 0.0], [1.5, 0.15, 0.0],
                    ], dtype=float),
                    np.array([
                        [1.1, 0.11, 0.0], [0.55, 0.055, 0.0], [0.55, 0.055, 0.0],
                        [2.2, 0.22, 0.0], [1.1, 0.11, 0.0], [1.1, 0.11, 0.0],
                        [3.3, 0.33, 0.0], [1.65, 0.165, 0.0], [1.65, 0.165, 0.0],
                    ], dtype=float),
                ]

            def iter_frames(self, start, stop, period):
                for i in range(start, stop or self.frames, period):
                    yield self._positions[i], self._forces[i]

        return IterableMockTrajectory()

    def test_accumulate_single_species_number(self, trajectory):
        """accumulate with single species number density produces correct grid."""

        gs = DensityGrid(trajectory, "number", nbins=5)
        gs.accumulate(trajectory, atom_names="O", rigid=False, start=0, stop=2)

        # Verify grid was populated
        assert gs.count == 2
        assert gs.counter.sum() > 0
        assert gs.grid_progress == "Allocated"


class TestSelectionGetWeights:
    """Tests for Selection.get_weights() method."""

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

        ss = Selection(trajectory, 'O', centre_location=True, rigid=False, density_type='number')
        result = ss.get_weights()

        assert result == 1.0

    def test_charge_single_species_returns_charges(self, trajectory):
        """Charge density for single species should return charge array."""

        ss = Selection(trajectory, 'O', centre_location=True, rigid=False, density_type='charge')
        result = ss.get_weights()

        expected = np.array([-0.8, -0.8, -0.8])  # O charges
        np.testing.assert_array_equal(result, expected)

    def test_charge_multi_species_not_rigid_returns_list(self, trajectory):
        """Charge density for multi-species non-rigid should return list of charge arrays."""

        ss = Selection(trajectory, ['O', 'H1', 'H2'], centre_location=True, rigid=False, density_type='charge')
        result = ss.get_weights()

        assert isinstance(result, list)
        assert len(result) == 3
        np.testing.assert_array_equal(result[0], np.array([-0.8, -0.8, -0.8]))  # O
        np.testing.assert_array_equal(result[1], np.array([0.4, 0.4, 0.4]))    # H1
        np.testing.assert_array_equal(result[2], np.array([0.4, 0.4, 0.4]))    # H2

    def test_charge_rigid_returns_summed_charges(self, trajectory):
        """Charge density for rigid should return total charge per molecule."""

        ss = Selection(trajectory, ['O', 'H1', 'H2'], centre_location=True, rigid=True, density_type='charge')
        result = ss.get_weights()

        # Total charge per molecule: -0.8 + 0.4 + 0.4 = 0.0
        expected = np.array([0.0, 0.0, 0.0])
        np.testing.assert_allclose(result, expected)

    def test_polarisation_returns_dipole_projection(self, trajectory, positions):
        """Polarisation density should return dipole projected along axis."""

        ss = Selection(
            trajectory, ['O', 'H1', 'H2'], centre_location=True, rigid=True,
            density_type='polarisation', polarisation_axis=0
        )
        result = ss.get_weights(positions)

        # COM for each molecule is at x = (16*x_O + 1*x_H1 + 1*x_H2) / 18
        # Molecule 0: COM_x = (16*0 + 1*1 + 1*(-1)) / 18 = 0
        # Dipole_x = q_O*(x_O - COM_x) + q_H1*(x_H1 - COM_x) + q_H2*(x_H2 - COM_x)
        #          = -0.8*(0 - 0) + 0.4*(1 - 0) + 0.4*(-1 - 0)
        #          = 0 + 0.4 - 0.4 = 0.0
        # Similarly for molecules 1 and 2
        expected = np.array([0.0, 0.0, 0.0])
        np.testing.assert_allclose(result, expected, atol=1e-10)


class TestSelectionValidation:
    """Tests for Selection input validation."""

    @pytest.fixture
    def trajectory(self):
        return MockTrajectory()

    def test_density_type_validation_called(self, trajectory, mocker):
        """Selection should call validate_density_type with the provided value."""
        mock_validate = mocker.patch(
            'revelsMD.density.selection.validate_density_type',
            return_value='number'
        )

        Selection(
            trajectory,
            atom_names='O',
            centre_location=True,
            density_type='NUMBER',
        )

        mock_validate.assert_called_once_with('NUMBER')


class TestSelectionGetForces:
    """Tests for Selection.get_forces() method."""

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

        ss = Selection(trajectory, 'O', centre_location=True, rigid=False)
        result = ss.get_forces(forces)

        expected = forces[[0, 3, 6], :]  # O atoms
        np.testing.assert_array_equal(result, expected)

    def test_multi_species_not_rigid_returns_list(self, trajectory, forces):
        """Multi-species, non-rigid should return list of force arrays."""

        ss = Selection(trajectory, ['O', 'H1', 'H2'], centre_location=True, rigid=False)
        result = ss.get_forces(forces)

        assert isinstance(result, list)
        assert len(result) == 3
        np.testing.assert_array_equal(result[0], forces[[0, 3, 6], :])  # O
        np.testing.assert_array_equal(result[1], forces[[1, 4, 7], :])  # H1
        np.testing.assert_array_equal(result[2], forces[[2, 5, 8], :])  # H2

    def test_rigid_sums_forces_across_molecule(self, trajectory, forces):
        """Rigid molecule should sum forces across all atoms in molecule."""

        ss = Selection(trajectory, ['O', 'H1', 'H2'], centre_location=True, rigid=True)
        result = ss.get_forces(forces)

        # Sum forces for each molecule
        # Molecule 0: [1,0,0] + [0.5,0,0] + [0.5,0,0] = [2,0,0]
        # Molecule 1: [2,0,0] + [1,0,0] + [1,0,0] = [4,0,0]
        # Molecule 2: [3,0,0] + [1.5,0,0] + [1.5,0,0] = [6,0,0]
        assert result.shape == (3, 3)
        np.testing.assert_allclose(result[:, 0], [2.0, 4.0, 6.0])
        np.testing.assert_allclose(result[:, 1], [0.0, 0.0, 0.0])


class TestSelectionGetPositionsPeriodicBoundary:
    """Tests for Selection.get_positions() with molecules spanning periodic boundaries."""

    @pytest.fixture
    def trajectory(self):
        """Single molecule trajectory."""
        class SingleMolTrajectory:
            def __init__(self):
                self.box_x = self.box_y = self.box_z = 10.0
                self.units = 'real'

            def get_indices(self, atom_name):
                return {'O': np.array([0]), 'H1': np.array([1]), 'H2': np.array([2])}[atom_name]

            def get_masses(self, atom_name):
                return {'O': np.array([16.0]), 'H1': np.array([1.0]), 'H2': np.array([1.0])}[atom_name]

            def get_charges(self, atom_name):
                return {'O': np.array([-0.8]), 'H1': np.array([0.4]), 'H2': np.array([0.4])}[atom_name]

        return SingleMolTrajectory()

    @pytest.fixture
    def positions_across_boundary(self):
        """Molecule spanning periodic boundary: O at x=9.5, H1 at x=0.3, H2 at x=0.5."""
        return np.array([
            [9.5, 5.0, 5.0],   # O (index 0) - near right edge
            [0.3, 5.0, 5.0],   # H1 (index 1) - wrapped to left edge
            [0.5, 5.0, 5.0],   # H2 (index 2) - wrapped to left edge
        ], dtype=float)

    def test_com_with_molecule_spanning_periodic_boundary(self, trajectory, positions_across_boundary):
        """COM should handle molecules that span periodic boundaries correctly."""

        # Box is 10x10x10, molecule spans boundary in x
        # O at x=9.5, H1 at x=0.3 (really at x=10.3, i.e. 0.8 from O)
        # H2 at x=0.5 (really at x=10.5, i.e. 1.0 from O)
        ss = Selection(
            trajectory, ['O', 'H1', 'H2'], centre_location=True, rigid=True
        )
        result = ss.get_positions(positions_across_boundary)

        # With minimum image: H1 is at x=10.3, H2 is at x=10.5 relative to O
        # COM_x = (16*9.5 + 1*10.3 + 1*10.5) / 18 = (152 + 10.3 + 10.5) / 18 = 172.8 / 18 = 9.6
        # Then wrapped to box: 9.6 (already in box)
        #
        # Without minimum image (BUG):
        # COM_x = (16*9.5 + 1*0.3 + 1*0.5) / 18 = (152 + 0.8) / 18 = 8.49 (WRONG!)

        # The COM should be near x=9.6, not x=8.49
        assert result[0, 0] > 9.0, f"COM x={result[0, 0]} should be > 9.0 (near the O atom)"

    def test_dipole_with_molecule_spanning_periodic_boundary(self, trajectory, positions_across_boundary):
        """Dipole calculation should handle molecules that span periodic boundaries."""

        ss = Selection(
            trajectory, ['O', 'H1', 'H2'], centre_location=True, rigid=True,
            density_type='polarisation', polarisation_axis=0
        )
        result = ss.get_weights(positions_across_boundary)

        # The molecule is symmetric around COM in y and z, but asymmetric in x
        # H1 is at x=10.3 (0.7 from COM at 9.6), H2 is at x=10.5 (0.9 from COM)
        # O is at x=9.5 (-0.1 from COM)
        # Dipole_x = q_O*(x_O - COM_x) + q_H1*(x_H1 - COM_x) + q_H2*(x_H2 - COM_x)
        #          = -0.8*(-0.1) + 0.4*(0.7) + 0.4*(0.9)
        #          = 0.08 + 0.28 + 0.36 = 0.72
        #
        # Without minimum image (BUG), H atoms would appear far from COM:
        # Dipole_x = -0.8*(-0.1) + 0.4*(0.3-9.6) + 0.4*(0.5-9.6)
        #          = 0.08 + 0.4*(-9.3) + 0.4*(-9.1) = 0.08 - 3.72 - 3.64 = -7.28

        # The dipole should be small and positive, not large and negative
        assert result[0] > 0, f"Dipole x={result[0]} should be > 0"
        assert result[0] < 1.0, f"Dipole x={result[0]} should be < 1.0 (small molecule)"


class TestSelectionExtract:
    """Tests for Selection.extract() method."""

    @pytest.fixture
    def trajectory(self):
        return MockTrajectory()

    @pytest.fixture
    def positions(self):
        """9 atoms: 3 water molecules."""
        return np.array([
            [1.0, 5.0, 5.0], [1.5, 5.0, 5.0], [0.5, 5.0, 5.0],
            [4.0, 5.0, 5.0], [4.5, 5.0, 5.0], [3.5, 5.0, 5.0],
            [7.0, 5.0, 5.0], [7.5, 5.0, 5.0], [6.5, 5.0, 5.0],
        ], dtype=float)

    @pytest.fixture
    def forces(self):
        """Forces for 9 atoms."""
        return np.array([
            [1.0, 0.1, 0.0], [0.5, 0.05, 0.0], [0.5, 0.05, 0.0],
            [2.0, 0.2, 0.0], [1.0, 0.1, 0.0], [1.0, 0.1, 0.0],
            [3.0, 0.3, 0.0], [1.5, 0.15, 0.0], [1.5, 0.15, 0.0],
        ], dtype=float)

    def test_extract_returns_tuple_of_three(self, trajectory, positions, forces):
        """extract() should return a tuple of (positions, forces, weights)."""

        ss = Selection(trajectory, 'O', centre_location=True, rigid=False, density_type='number')
        result = ss.extract(positions, forces)

        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_extract_matches_individual_methods_single_species(self, trajectory, positions, forces):
        """extract() should return same values as calling get_positions, get_forces, get_weights."""

        ss = Selection(trajectory, 'O', centre_location=True, rigid=False, density_type='number')
        extracted_positions, extracted_forces, extracted_weights = ss.extract(positions, forces)

        np.testing.assert_array_equal(extracted_positions, ss.get_positions(positions))
        np.testing.assert_array_equal(extracted_forces, ss.get_forces(forces))
        assert extracted_weights == ss.get_weights(positions)

    def test_extract_matches_individual_methods_rigid_com(self, trajectory, positions, forces):
        """extract() should match individual methods for rigid COM case."""

        ss = Selection(trajectory, ['O', 'H1', 'H2'], centre_location=True, rigid=True, density_type='number')
        extracted_positions, extracted_forces, extracted_weights = ss.extract(positions, forces)

        np.testing.assert_array_equal(extracted_positions, ss.get_positions(positions))
        np.testing.assert_array_equal(extracted_forces, ss.get_forces(forces))
        assert extracted_weights == ss.get_weights(positions)

    def test_extract_matches_individual_methods_charge_density(self, trajectory, positions, forces):
        """extract() should match individual methods for charge density."""

        ss = Selection(trajectory, 'O', centre_location=True, rigid=False, density_type='charge')
        extracted_positions, extracted_forces, extracted_weights = ss.extract(positions, forces)

        np.testing.assert_array_equal(extracted_positions, ss.get_positions(positions))
        np.testing.assert_array_equal(extracted_forces, ss.get_forces(forces))
        np.testing.assert_array_equal(extracted_weights, ss.get_weights(positions))


class TestSelectionGetPositions:
    """Tests for Selection.get_positions() method."""

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

        ss = Selection(trajectory, 'O', centre_location=True, rigid=False)
        result = ss.get_positions(positions)

        expected = positions[[0, 3, 6], :]  # O atoms
        np.testing.assert_array_equal(result, expected)

    def test_multi_species_not_rigid_returns_list(self, trajectory, positions):
        """Multi-species, non-rigid should return list of position arrays."""

        ss = Selection(trajectory, ['O', 'H1', 'H2'], centre_location=True, rigid=False)
        result = ss.get_positions(positions)

        assert isinstance(result, list)
        assert len(result) == 3
        np.testing.assert_array_equal(result[0], positions[[0, 3, 6], :])  # O
        np.testing.assert_array_equal(result[1], positions[[1, 4, 7], :])  # H1
        np.testing.assert_array_equal(result[2], positions[[2, 5, 8], :])  # H2

    def test_rigid_com_returns_center_of_mass(self, trajectory, positions):
        """Rigid molecule with COM should return mass-weighted center positions."""

        ss = Selection(trajectory, ['O', 'H1', 'H2'], centre_location=True, rigid=True)
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

        # centre_location=1 means use H1 positions
        ss = Selection(trajectory, ['O', 'H1', 'H2'], centre_location=1, rigid=True)
        result = ss.get_positions(positions)

        expected = positions[[1, 4, 7], :]  # H1 atoms
        np.testing.assert_array_equal(result, expected)


# ---------------------------------------------------------------------------
# Import tests
# ---------------------------------------------------------------------------

def test_selectionstate_importable_from_density():
    """Selection should be importable from revelsMD.density."""
    assert Selection is not None


def test_selectionstate_importable_from_submodule():
    """Selection should be importable from revelsMD.density.selection."""
    from revelsMD.density.selection import Selection
    assert Selection is not None


def test_selectionstate_backward_compatible_via_revels3d():
    """Revels3D.SelectionState should still work but emit deprecation warning."""
    from revelsMD.revels_3D import Revels3D
    with pytest.warns(DeprecationWarning, match="Revels3D.SelectionState is deprecated.*import Selection"):
        assert Revels3D.SelectionState is Selection


def test_gridstate_importable_from_density():
    """DensityGrid should be importable from revelsMD.density."""
    assert DensityGrid is not None


def test_gridstate_importable_from_submodule():
    """DensityGrid should be importable from revelsMD.density.density_grid."""
    from revelsMD.density.density_grid import DensityGrid
    assert DensityGrid is not None


def test_gridstate_backward_compatible_via_revels3d():
    """Revels3D.GridState should still work but emit deprecation warning."""
    from revelsMD.revels_3D import Revels3D
    with pytest.warns(DeprecationWarning, match="Revels3D.GridState is deprecated.*import DensityGrid"):
        assert Revels3D.GridState is DensityGrid


# ---------------------------------------------------------------------------
# compute_density() convenience function
# ---------------------------------------------------------------------------

class TestComputeDensity:
    """Tests for compute_density() convenience function."""

    @pytest.fixture
    def trajectory(self):
        """Create mock trajectory that supports iteration."""
        class IterableMockTrajectory(MockTrajectory):
            def __init__(self):
                super().__init__()
                self.frames = 2
                self._positions = [
                    np.array([
                        [1.0, 5.0, 5.0], [1.5, 5.0, 5.0], [0.5, 5.0, 5.0],
                        [4.0, 5.0, 5.0], [4.5, 5.0, 5.0], [3.5, 5.0, 5.0],
                        [7.0, 5.0, 5.0], [7.5, 5.0, 5.0], [6.5, 5.0, 5.0],
                    ], dtype=float),
                    np.array([
                        [1.1, 5.1, 5.0], [1.6, 5.1, 5.0], [0.6, 5.1, 5.0],
                        [4.1, 5.1, 5.0], [4.6, 5.1, 5.0], [3.6, 5.1, 5.0],
                        [7.1, 5.1, 5.0], [7.6, 5.1, 5.0], [6.6, 5.1, 5.0],
                    ], dtype=float),
                ]
                self._forces = [
                    np.array([
                        [1.0, 0.1, 0.0], [0.5, 0.05, 0.0], [0.5, 0.05, 0.0],
                        [2.0, 0.2, 0.0], [1.0, 0.1, 0.0], [1.0, 0.1, 0.0],
                        [3.0, 0.3, 0.0], [1.5, 0.15, 0.0], [1.5, 0.15, 0.0],
                    ], dtype=float),
                    np.array([
                        [1.1, 0.11, 0.0], [0.55, 0.055, 0.0], [0.55, 0.055, 0.0],
                        [2.2, 0.22, 0.0], [1.1, 0.11, 0.0], [1.1, 0.11, 0.0],
                        [3.3, 0.33, 0.0], [1.65, 0.165, 0.0], [1.65, 0.165, 0.0],
                    ], dtype=float),
                ]

            def iter_frames(self, start, stop, period):
                for i in range(start, stop or self.frames, period):
                    yield self._positions[i], self._forces[i]

        return IterableMockTrajectory()

    def test_compute_density_returns_densitygrid(self, trajectory):
        """compute_density should return a DensityGrid with computed density."""
        from revelsMD.density import compute_density

        result = compute_density(trajectory, atom_names='O', nbins=5)

        assert isinstance(result, DensityGrid)
        assert hasattr(result, 'rho_force')
        assert result.rho_force.shape == (5, 5, 5)
        assert np.all(np.isfinite(result.rho_force))

    def test_compute_density_with_rigid_molecules(self, trajectory):
        """compute_density should work with rigid molecules."""
        from revelsMD.density import compute_density

        result = compute_density(
            trajectory,
            atom_names=['O', 'H1', 'H2'],
            rigid=True,
            nbins=5
        )

        assert hasattr(result, 'rho_force')
        assert result.rho_force.shape == (5, 5, 5)

    def test_compute_density_importable_from_density(self):
        """compute_density should be importable from revelsMD.density."""
        from revelsMD.density import compute_density
        assert compute_density is not None

    @pytest.fixture
    def trajectory_with_get_frame(self):
        """Create mock trajectory that supports both iteration and get_frame."""
        class IterableMockTrajectoryWithGetFrame(MockTrajectory):
            def __init__(self):
                super().__init__()
                self.frames = 4
                # 9 atoms: 3 water molecules (O, H1, H2 each)
                self._positions = [
                    np.array([
                        [1.0, 5.0, 5.0], [1.5, 5.0, 5.0], [0.5, 5.0, 5.0],
                        [4.0, 5.0, 5.0], [4.5, 5.0, 5.0], [3.5, 5.0, 5.0],
                        [7.0, 5.0, 5.0], [7.5, 5.0, 5.0], [6.5, 5.0, 5.0],
                    ], dtype=float),
                    np.array([
                        [1.1, 5.1, 5.0], [1.6, 5.1, 5.0], [0.6, 5.1, 5.0],
                        [4.1, 5.1, 5.0], [4.6, 5.1, 5.0], [3.6, 5.1, 5.0],
                        [7.1, 5.1, 5.0], [7.6, 5.1, 5.0], [6.6, 5.1, 5.0],
                    ], dtype=float),
                    np.array([
                        [1.2, 5.2, 5.0], [1.7, 5.2, 5.0], [0.7, 5.2, 5.0],
                        [4.2, 5.2, 5.0], [4.7, 5.2, 5.0], [3.7, 5.2, 5.0],
                        [7.2, 5.2, 5.0], [7.7, 5.2, 5.0], [6.7, 5.2, 5.0],
                    ], dtype=float),
                    np.array([
                        [1.3, 5.3, 5.0], [1.8, 5.3, 5.0], [0.8, 5.3, 5.0],
                        [4.3, 5.3, 5.0], [4.8, 5.3, 5.0], [3.8, 5.3, 5.0],
                        [7.3, 5.3, 5.0], [7.8, 5.3, 5.0], [6.8, 5.3, 5.0],
                    ], dtype=float),
                ]
                self._forces = [
                    np.array([
                        [1.0, 0.1, 0.0], [0.5, 0.05, 0.0], [0.5, 0.05, 0.0],
                        [2.0, 0.2, 0.0], [1.0, 0.1, 0.0], [1.0, 0.1, 0.0],
                        [3.0, 0.3, 0.0], [1.5, 0.15, 0.0], [1.5, 0.15, 0.0],
                    ], dtype=float),
                    np.array([
                        [1.1, 0.11, 0.0], [0.55, 0.055, 0.0], [0.55, 0.055, 0.0],
                        [2.2, 0.22, 0.0], [1.1, 0.11, 0.0], [1.1, 0.11, 0.0],
                        [3.3, 0.33, 0.0], [1.65, 0.165, 0.0], [1.65, 0.165, 0.0],
                    ], dtype=float),
                    np.array([
                        [1.2, 0.12, 0.0], [0.6, 0.06, 0.0], [0.6, 0.06, 0.0],
                        [2.4, 0.24, 0.0], [1.2, 0.12, 0.0], [1.2, 0.12, 0.0],
                        [3.6, 0.36, 0.0], [1.8, 0.18, 0.0], [1.8, 0.18, 0.0],
                    ], dtype=float),
                    np.array([
                        [1.3, 0.13, 0.0], [0.65, 0.065, 0.0], [0.65, 0.065, 0.0],
                        [2.6, 0.26, 0.0], [1.3, 0.13, 0.0], [1.3, 0.13, 0.0],
                        [3.9, 0.39, 0.0], [1.95, 0.195, 0.0], [1.95, 0.195, 0.0],
                    ], dtype=float),
                ]

            def iter_frames(self, start, stop, period):
                for i in range(start, stop or self.frames, period):
                    yield self._positions[i], self._forces[i]

            def get_frame(self, idx):
                return self._positions[idx], self._forces[idx]

        return IterableMockTrajectoryWithGetFrame()

    def test_compute_density_lambda(self, trajectory_with_get_frame):
        """compute_density with integration='lambda' populates rho_lambda."""
        from revelsMD.density import compute_density

        grid = compute_density(
            trajectory_with_get_frame,
            atom_names='O',
            nbins=5,
            integration='lambda',
            sections=2,
        )

        assert grid.rho_lambda is not None
        assert grid.grid_progress == "Lambda"
        assert grid.rho_lambda.shape == (5, 5, 5)

    def test_compute_density_standard_default(self, trajectory):
        """Default integration='standard' behaves as before."""
        from revelsMD.density import compute_density

        grid = compute_density(trajectory, atom_names='O', nbins=5)

        assert grid.rho_force is not None
        assert grid.rho_lambda is None
        assert grid.grid_progress == "Allocated"

    def test_compute_density_invalid_integration(self, trajectory):
        """Invalid integration raises ValueError."""
        from revelsMD.density import compute_density

        with pytest.raises(ValueError, match="integration"):
            compute_density(trajectory, atom_names='O', nbins=5, integration='invalid')


# ---------------------------------------------------------------------------
# DensityGrid.get_lambda() edge case tests
# ---------------------------------------------------------------------------

class TestDensityGridGetLambdaEdgeCases:
    """Tests for edge case handling in DensityGrid.get_lambda()."""

    def test_get_lambda_produces_finite_output(self):
        """get_lambda produces finite combination and optimal_density values.

        This test verifies the fix for the zero-variance edge case bug where
        division by zero could produce NaN/Inf in the output.
        """
        from revelsMD.density import DensityGrid, Selection

        # Create a minimal trajectory with very few frames
        # This increases the chance of zero-variance voxels
        class MinimalTrajectory:
            def __init__(self):
                self.box_x = self.box_y = self.box_z = 10.0
                self.units = 'real'
                self.frames = 2
                self.beta = 1.0 / (300.0 * 0.0019872041)

            def get_indices(self, atom_name):
                return np.array([0, 1])

            def get_masses(self, atom_name):
                return np.array([1.0, 1.0])

            def get_frame(self, idx):
                # Return identical positions for all frames to create zero variance
                positions = np.array([[2.0, 5.0, 5.0], [8.0, 5.0, 5.0]])
                forces = np.array([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]])
                return positions, forces

            def iter_frames(self, start, stop, period):
                for i in range(start, stop or self.frames, period):
                    yield self.get_frame(i)

        traj = MinimalTrajectory()
        gs = DensityGrid(traj, "number", nbins=3)
        ss = Selection(traj, 'H', centre_location=True, rigid=False, density_type='number')
        gs._selection = ss
        gs.kernel = "triangular"
        gs.to_run = list(range(traj.frames))

        # Manually deposit frames
        for positions, forces in traj.iter_frames(0, traj.frames, 1):
            gs.deposit(
                ss.get_positions(positions),
                ss.get_forces(forces),
                ss.get_weights(),
                kernel="triangular"
            )

        gs.grid_progress = "Allocated"
        gs.get_lambda(traj, sections=2)

        # The key assertion: no NaN or Inf values
        assert np.all(np.isfinite(gs.lambda_weights)), "lambda_weights contains NaN/Inf"
        assert np.all(np.isfinite(gs.rho_lambda)), "rho_lambda contains NaN/Inf"
