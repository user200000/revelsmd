"""
Tests for the deprecated revels_rdf API.

These tests verify that:
1. run_rdf and run_rdf_lambda still work (smoke tests)
2. Deprecated APIs emit appropriate warnings

The underlying RDF class functionality is tested in test_rdf_class.py.
Species validation, edge conditions, and other detailed tests are
covered there and not duplicated here.
"""

import pytest
import warnings
import numpy as np
from revelsMD.revels_rdf import run_rdf, run_rdf_lambda, RevelsRDF


class TSMock:
    """Minimal trajectory-state mock implementing the unified TrajectoryState interface."""
    def __init__(self, temperature: float = 300.0, units: str = "real"):
        self.box_x = 10.0
        self.box_y = 10.0
        self.box_z = 10.0
        self.units = units
        self.temperature = temperature
        self.frames = 3

        # Compute beta from temperature and units
        from revelsMD.trajectories._base import compute_beta
        self.beta = compute_beta(units, temperature)

        # 3 frames x 3 atoms x 3 coordinates
        self._positions = np.array([
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[1.1, 2.1, 3.1], [4.1, 5.1, 6.1], [7.1, 8.1, 9.1]],
            [[0.9, 1.9, 2.9], [3.9, 4.9, 5.9], [6.9, 7.9, 8.9]]
        ], dtype=float)
        self._forces = np.array([
            [[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]],
            [[0.2, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 0.2]],
            [[0.3, 0.0, 0.0], [0.0, 0.3, 0.0], [0.0, 0.0, 0.3]],
        ], dtype=float)

        self.species = ["H", "O", "H"]
        self._ids = {"H": np.array([0, 2]), "O": np.array([1])}
        self._charges = {"H": np.array([0.1, 0.1]), "O": np.array([-0.2])}
        self._masses = {"H": np.array([1.0, 1.0]), "O": np.array([16.0])}

    def get_indices(self, atype):
        return self._ids[atype]

    def get_charges(self, atype):
        return self._charges[atype]

    def get_masses(self, atype):
        return self._masses[atype]

    def iter_frames(self, start=0, stop=None, stride=1):
        """Iterate over frames yielding (positions, forces) tuples."""
        if stop is None:
            stop = self.frames
        elif stop < 0:
            stop = self.frames + stop
        for i in range(start, stop, stride):
            yield self._positions[i], self._forces[i]

    def get_frame(self, index):
        """Return (positions, forces) for a specific frame."""
        return self._positions[index], self._forces[index]


@pytest.fixture
def ts():
    """Provide a reusable trajectory-state mock."""
    return TSMock()


# -------------------------------
# Smoke tests for deprecated functions
# -------------------------------

def test_run_rdf_returns_correct_shape(ts):
    """run_rdf should return a (2, n) array with r and g(r)."""
    result = run_rdf(
        ts,
        atom_a="H",
        atom_b="H",
        delr=1.0,
        start=0,
        stop=2,
        period=1,
        rmax=True,
        from_zero=True,
    )
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == 2
    assert np.all(np.isfinite(result))


def test_run_rdf_lambda_returns_correct_shape(ts):
    """run_rdf_lambda should return an (n, 3) array with r, g(r), and lambda."""
    result = run_rdf_lambda(
        ts,
        atom_a="H",
        atom_b="H",
        delr=1.0,
        start=0,
        stop=2,
        period=1,
        rmax=True,
    )
    assert isinstance(result, np.ndarray)
    assert result.shape[1] == 3
    assert np.all(np.isfinite(result))
    assert np.all((result[:, 2] >= -1) & (result[:, 2] <= 2))  # lambda values roughly bounded


# -------------------------------
# Deprecation warning tests
# -------------------------------

class TestDeprecationWarnings:
    """Test that deprecated APIs emit appropriate warnings."""

    def test_revels_rdf_run_rdf_emits_warning(self, ts):
        """RevelsRDF.run_rdf should emit DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # Access the deprecated method
            func = RevelsRDF.run_rdf
            # Check warning was emitted
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "RevelsRDF.run_rdf is deprecated" in str(w[0].message)

    def test_revels_rdf_run_rdf_lambda_emits_warning(self, ts):
        """RevelsRDF.run_rdf_lambda should emit DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # Access the deprecated method
            func = RevelsRDF.run_rdf_lambda
            # Check warning was emitted
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "RevelsRDF.run_rdf_lambda is deprecated" in str(w[0].message)

    def test_revels_rdf_deprecated_method_still_works(self, ts):
        """Deprecated RevelsRDF methods should still produce valid results."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = RevelsRDF.run_rdf(
                ts,
                atom_a="H",
                atom_b="H",
                delr=1.0,
                start=0,
                stop=2,
            )
            assert isinstance(result, np.ndarray)
            assert result.shape[0] == 2
            assert np.all(np.isfinite(result))
