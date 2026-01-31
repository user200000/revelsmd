"""Integration tests for make_force_grid with different configurations."""

import pytest
import numpy as np

from revelsMD.density import GridState


class TSMock:
    """Minimal trajectory mock for GridState tests."""

    def __init__(self):
        self.box_x = self.box_y = self.box_z = 10.0
        self.frames = 2
        self.units = "real"
        self.beta = 1.0 / (300.0 * 0.0019872041)  # 1/(kB*T) for T=300K in real units
        self._ids = {}
        self._charges = {}
        self._masses = {}
        self.positions = np.array([
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]],
        ])
        self.forces = np.array([
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            [[0.2, 0.3, 0.4], [0.5, 0.6, 0.7]],
        ])

    def get_indices(self, name):
        return self._ids.get(name, np.array([]))

    def get_charges(self, name):
        return self._charges.get(name, np.array([]))

    def get_masses(self, name):
        return self._masses.get(name, np.array([]))

    def iter_frames(self, start, stop, period):
        stop = stop if stop is not None else self.frames
        for i in range(start, stop, period):
            yield self.positions[i], self.forces[i]

    def get_frame(self, idx):
        return self.positions[idx], self.forces[idx]


class TestMakeForceGridConfigurations:
    """Test make_force_grid with different density/rigid/centre configurations."""

    @pytest.fixture
    def ts_single_species(self):
        """Mock with single species (indistinguishable_set=True)."""
        ts = TSMock()
        ts._ids = {"H": np.array([0, 1])}
        ts._charges = {"H": np.array([0.1, 0.1])}
        ts._masses = {"H": np.array([1.0, 1.0])}
        return ts

    @pytest.fixture
    def ts_multi_species(self):
        """Mock with multiple species (indistinguishable_set=False)."""
        ts = TSMock()
        ts.positions = np.array([
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]],
        ])
        ts.forces = np.array([
            [[0.1, 0.0, 0.0], [0.0, 0.1, 0.0]],
            [[0.1, 0.1, 0.0], [0.0, 0.0, 0.1]],
        ])
        ts._ids = {"H": np.array([0]), "O": np.array([1])}
        ts._charges = {"H": np.array([0.1]), "O": np.array([-0.2])}
        ts._masses = {"H": np.array([1.0]), "O": np.array([16.0])}
        return ts

    # --- Number density tests ---

    def test_number_single_species(self, ts_single_species):
        """Single species number density populates grid correctly."""
        gs = GridState(ts_single_species, "number", nbins=4)
        gs.make_force_grid(ts_single_species, atom_names="H", rigid=False)
        assert gs.grid_progress == "Allocated"
        assert gs.counter.sum() > 0

    def test_number_multi_species_not_rigid(self, ts_multi_species):
        """Multi-species, non-rigid number density populates grid correctly."""
        gs = GridState(ts_multi_species, "number", nbins=4)
        gs.make_force_grid(ts_multi_species, atom_names=["H", "O"], rigid=False)
        assert gs.grid_progress == "Allocated"
        assert gs.counter.sum() > 0

    def test_number_rigid_com(self, ts_multi_species):
        """Rigid number density at COM populates grid correctly."""
        gs = GridState(ts_multi_species, "number", nbins=4)
        gs.make_force_grid(ts_multi_species, atom_names=["H", "O"], rigid=True, centre_location=True)
        assert gs.grid_progress == "Allocated"
        assert gs.counter.sum() > 0

    def test_number_rigid_atom(self, ts_multi_species):
        """Rigid number density at specific atom populates grid correctly."""
        gs = GridState(ts_multi_species, "number", nbins=4)
        gs.make_force_grid(ts_multi_species, atom_names=["H", "O"], rigid=True, centre_location=0)
        assert gs.grid_progress == "Allocated"
        assert gs.counter.sum() > 0

    # --- Charge density tests ---

    def test_charge_single_species(self, ts_single_species):
        """Single species charge density populates grid correctly."""
        gs = GridState(ts_single_species, "charge", nbins=4)
        gs.make_force_grid(ts_single_species, atom_names="H", rigid=False)
        assert gs.grid_progress == "Allocated"
        assert gs.counter.sum() > 0

    def test_charge_multi_species_not_rigid(self, ts_multi_species):
        """Multi-species, non-rigid charge density populates grid correctly."""
        gs = GridState(ts_multi_species, "charge", nbins=4)
        gs.make_force_grid(ts_multi_species, atom_names=["H", "O"], rigid=False)
        assert gs.grid_progress == "Allocated"
        assert np.any(gs.counter != 0)

    def test_charge_rigid_com(self, ts_multi_species):
        """Rigid charge density at COM populates grid correctly."""
        gs = GridState(ts_multi_species, "charge", nbins=4)
        gs.make_force_grid(ts_multi_species, atom_names=["H", "O"], rigid=True, centre_location=True)
        assert gs.grid_progress == "Allocated"
        assert np.any(gs.counter != 0)

    def test_charge_rigid_atom(self, ts_multi_species):
        """Rigid charge density at specific atom populates grid correctly."""
        gs = GridState(ts_multi_species, "charge", nbins=4)
        gs.make_force_grid(ts_multi_species, atom_names=["H", "O"], rigid=True, centre_location=0)
        assert gs.grid_progress == "Allocated"
        assert np.any(gs.counter != 0)

    # --- Polarisation density tests ---

    def test_polarisation_rigid_com(self, ts_multi_species):
        """Rigid polarisation density at COM populates grid correctly."""
        gs = GridState(ts_multi_species, "polarisation", nbins=4)
        gs.make_force_grid(ts_multi_species, atom_names=["H", "O"], rigid=True, centre_location=True, polarisation_axis=0)
        assert gs.grid_progress == "Allocated"
        assert gs.selection_state.polarisation_axis == 0

    def test_polarisation_rigid_atom(self, ts_multi_species):
        """Rigid polarisation density at specific atom populates grid correctly."""
        gs = GridState(ts_multi_species, "polarisation", nbins=4)
        gs.make_force_grid(ts_multi_species, atom_names=["H", "O"], rigid=True, centre_location=0, polarisation_axis=1)
        assert gs.grid_progress == "Allocated"
        assert gs.selection_state.polarisation_axis == 1

    # --- Error cases ---

    def test_polarisation_not_rigid_raises(self, ts_multi_species):
        """Polarisation without rigid=True raises ValueError."""
        gs = GridState(ts_multi_species, "polarisation", nbins=4)
        with pytest.raises(ValueError, match="rigid molecules"):
            gs.make_force_grid(ts_multi_species, atom_names=["H", "O"], rigid=False)

    def test_polarisation_single_species_raises(self, ts_single_species):
        """Polarisation with single species raises ValueError."""
        gs = GridState(ts_single_species, "polarisation", nbins=4)
        with pytest.raises(ValueError, match="single atom"):
            gs.make_force_grid(ts_single_species, atom_names="H", rigid=True, centre_location=True)

    def test_invalid_density_type_raises(self, ts_single_species):
        """Invalid density type raises ValueError."""
        gs = GridState(ts_single_species, "number", nbins=4)
        gs.density_type = "invalid"
        with pytest.raises(ValueError, match="Unknown density_type"):
            gs.make_force_grid(ts_single_species, atom_names="H", rigid=False)

    def test_rigid_invalid_centre_location_raises(self, ts_multi_species):
        """Rigid with invalid centre_location raises ValueError."""
        gs = GridState(ts_multi_species, "number", nbins=4)
        with pytest.raises(ValueError, match="centre_location"):
            gs.make_force_grid(ts_multi_species, atom_names=["H", "O"], rigid=True, centre_location="invalid")
