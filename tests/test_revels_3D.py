"""
Tests for the deprecated revels_3D API.

These tests verify that:
1. Revels3D.GridState and Revels3D.SelectionState still work (smoke tests)
2. Deprecated APIs emit appropriate warnings

The underlying DensityGrid and Selection functionality is tested in test_density.py.
"""

import pytest
import warnings
import numpy as np

from revelsMD.revels_3D import Revels3D
from revelsMD.density import DensityGrid, Selection


class TSMock:
    """Minimal trajectory-state mock for deprecation tests."""
    def __init__(self):
        self.box_x = 10.0
        self.box_y = 10.0
        self.box_z = 10.0
        self.units = "real"
        self.temperature = 300.0
        self.frames = 2

        from revelsMD.trajectories._base import compute_beta
        self.beta = compute_beta(self.units, self.temperature)

        self.positions = np.array([
            [[1, 2, 3], [4, 5, 6]],
            [[2, 3, 4], [5, 6, 7]],
        ])
        self.forces = np.array([
            [[0.1, 0.0, 0.0], [0.0, 0.1, 0.0]],
            [[0.1, 0.1, 0.0], [0.0, 0.0, 0.1]],
        ])

        self.species = ["H", "O"]
        self._ids = {"H": np.array([0]), "O": np.array([1])}
        self._charges = {"H": np.array([0.1]), "O": np.array([-0.1])}
        self._masses = {"H": np.array([1.0]), "O": np.array([16.0])}

    def get_indices(self, atype):
        return self._ids[atype]

    def get_charges(self, atype):
        return self._charges[atype]

    def get_masses(self, atype):
        return self._masses[atype]

    def iter_frames(self, start=0, stop=None, stride=1):
        if stop is None:
            stop = self.frames
        for i in range(start, stop, stride):
            yield self.positions[i], self.forces[i]

    def get_frame(self, index):
        return self.positions[index], self.forces[index]


@pytest.fixture
def ts():
    """Provide a reusable trajectory-state mock."""
    return TSMock()


# -------------------------------
# Deprecation warning tests
# -------------------------------

class TestDeprecationWarnings:
    """Test that deprecated APIs emit appropriate warnings."""

    def test_revels3d_gridstate_emits_warning(self, ts):
        """Revels3D.GridState should emit DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # Access the deprecated attribute (assignment triggers the warning)
            _ = Revels3D.GridState
            # Check warning was emitted
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "Revels3D.GridState is deprecated" in str(w[0].message)

    def test_revels3d_selectionstate_emits_warning(self, ts):
        """Revels3D.SelectionState should emit DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # Access the deprecated attribute (assignment triggers the warning)
            _ = Revels3D.SelectionState
            # Check warning was emitted
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "Revels3D.SelectionState is deprecated" in str(w[0].message)

    def test_revels3d_gridstate_returns_densitygrid(self):
        """Revels3D.GridState should return DensityGrid class."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            assert Revels3D.GridState is DensityGrid

    def test_revels3d_selectionstate_returns_selection(self):
        """Revels3D.SelectionState should return Selection class."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            assert Revels3D.SelectionState is Selection


# -------------------------------
# Smoke tests for deprecated API
# -------------------------------

class TestDeprecatedAPIStillWorks:
    """Verify deprecated Revels3D methods still work."""

    def test_gridstate_via_revels3d_creates_grid(self, ts):
        """Creating DensityGrid via Revels3D.GridState should work."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            GridState = Revels3D.GridState
            gs = GridState(ts, density_type="number", nbins=4)
            assert gs.nbinsx == 4
            assert isinstance(gs, DensityGrid)

    def test_selectionstate_via_revels3d_creates_selection(self, ts):
        """Creating Selection via Revels3D.SelectionState should work."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            SelectionState = Revels3D.SelectionState
            ss = SelectionState(ts, "H", centre_location=True)
            assert ss.single_species
            assert isinstance(ss, Selection)
