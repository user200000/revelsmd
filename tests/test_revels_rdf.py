"""
Tests for the backward-compatible revels_rdf API.

These tests verify that:
1. run_rdf and run_rdf_lambda correctly delegate to the RDF class
2. Deprecated APIs emit appropriate warnings
3. Edge cases and validation are handled correctly
"""

import pytest
import warnings
import numpy as np
from revelsMD.revels_rdf import run_rdf, run_rdf_lambda, RevelsRDF
from revelsMD.trajectories import NumpyTrajectory


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
# run_rdf (like pairs)
# -------------------------------

def test_run_rdf_like_pairs(ts):
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
    # Result should be shape (2, n)
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == 2
    assert np.all(np.isfinite(result))


# -------------------------------
# run_rdf (unlike pairs)
# -------------------------------

def test_run_rdf_unlike_pairs(ts):
    result = run_rdf(
        ts,
        atom_a="H",
        atom_b="O",
        delr=1.0,
        start=0,
        stop=2,
        period=1,
        rmax=False,
        from_zero=False,
    )
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == 2
    assert np.all(np.isfinite(result))


# -------------------------------
# run_rdf edge conditions
# -------------------------------

def test_run_rdf_start_exceeds_frames(ts):
    """run_rdf should raise ValueError when start exceeds trajectory frames."""
    with pytest.raises(ValueError, match="First frame index exceeds"):
        run_rdf(ts, "H", "O", start=10)


def test_run_rdf_stop_exceeds_frames(ts):
    """run_rdf should raise ValueError when stop exceeds trajectory frames."""
    with pytest.raises(ValueError, match="Final frame index exceeds"):
        run_rdf(ts, "H", "O", stop=10)


def test_run_rdf_empty_frame_range(ts):
    """run_rdf should raise ValueError when frame range is empty."""
    with pytest.raises(ValueError, match="Final frame occurs before"):
        run_rdf(ts, "H", "O", start=2, stop=1)


# -------------------------------
# run_rdf_lambda
# -------------------------------

def test_run_rdf_lambda_like(ts):
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
    # Three columns: [r, combined RDF, lambda]
    assert result.shape[1] == 3
    assert np.all(np.isfinite(result))
    assert np.all((result[:, 2] >= -1) & (result[:, 2] <= 2))  # lambda values roughly bounded


# -------------------------------
# run_rdf_lambda edge conditions
# -------------------------------

def test_run_rdf_lambda_start_exceeds_frames(ts):
    """run_rdf_lambda should raise ValueError when start exceeds trajectory frames."""
    with pytest.raises(ValueError, match="First frame index exceeds"):
        run_rdf_lambda(ts, "H", "O", start=10)


def test_run_rdf_lambda_stop_exceeds_frames(ts):
    """run_rdf_lambda should raise ValueError when stop exceeds trajectory frames."""
    with pytest.raises(ValueError, match="Final frame index exceeds"):
        run_rdf_lambda(ts, "H", "O", stop=10)


def test_run_rdf_lambda_empty_frame_range(ts):
    """run_rdf_lambda should raise ValueError when frame range is empty."""
    with pytest.raises(ValueError, match="Final frame occurs before"):
        run_rdf_lambda(ts, "H", "O", start=2, stop=1)


# -------------------------------
# Species validation tests
# -------------------------------

class TestSpeciesValidation:
    """Test validation of species atom counts in RDF functions."""

    def test_run_rdf_like_species_one_atom_raises(self):
        """run_rdf with like-species and only 1 atom should raise ValueError."""
        positions = np.array([[[1, 2, 3]]], dtype=float)
        forces = np.array([[[0.1, 0.0, 0.0]]], dtype=float)
        species = ["O"]

        ts = NumpyTrajectory(
            positions, forces, 10.0, 10.0, 10.0, species, temperature=300.0, units="real"
        )

        with pytest.raises(ValueError, match="at least 2 atoms"):
            run_rdf(ts, 'O', 'O')

    def test_run_rdf_unlike_species_empty_raises(self):
        """run_rdf with unlike-species and empty selection should raise ValueError."""
        positions = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=float)
        forces = np.array([[[0.1, 0.0, 0.0], [0.0, 0.1, 0.0]]], dtype=float)
        species = ["H", "H"]

        ts = NumpyTrajectory(
            positions, forces, 10.0, 10.0, 10.0, species, temperature=300.0, units="real"
        )

        with pytest.raises(ValueError, match="No atoms found for species 'O'"):
            run_rdf(ts, 'O', 'H')

    def test_run_rdf_lambda_like_species_one_atom_raises(self):
        """run_rdf_lambda with like-species and only 1 atom should raise ValueError."""
        positions = np.array([[[1, 2, 3]]], dtype=float)
        forces = np.array([[[0.1, 0.0, 0.0]]], dtype=float)
        species = ["O"]

        ts = NumpyTrajectory(
            positions, forces, 10.0, 10.0, 10.0, species, temperature=300.0, units="real"
        )

        with pytest.raises(ValueError, match="at least 2 atoms"):
            run_rdf_lambda(ts, 'O', 'O')

    def test_run_rdf_lambda_unlike_species_empty_raises(self):
        """run_rdf_lambda with unlike-species and empty selection should raise ValueError."""
        positions = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=float)
        forces = np.array([[[0.1, 0.0, 0.0], [0.0, 0.1, 0.0]]], dtype=float)
        species = ["O", "O"]

        ts = NumpyTrajectory(
            positions, forces, 10.0, 10.0, 10.0, species, temperature=300.0, units="real"
        )

        with pytest.raises(ValueError, match="No atoms found for species 'H'"):
            run_rdf_lambda(ts, 'O', 'H')


# -------------------------------
# Tests using real NumpyTrajectory
# -------------------------------

class TestRDFWithNumpyTrajectory:
    """Test RDF functions work with real NumpyTrajectory objects."""

    def test_run_rdf_with_numpy_trajectory_state(self):
        """run_rdf should work with a real NumpyTrajectory."""
        positions = np.array([
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[1.1, 2.1, 3.1], [4.1, 5.1, 6.1], [7.1, 8.1, 9.1]],
            [[0.9, 1.9, 2.9], [3.9, 4.9, 5.9], [6.9, 7.9, 8.9]]
        ], dtype=float)
        forces = np.array([
            [[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]],
            [[0.2, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 0.2]],
            [[0.3, 0.0, 0.0], [0.0, 0.3, 0.0], [0.0, 0.0, 0.3]],
        ], dtype=float)
        species = ["H", "O", "H"]

        ts = NumpyTrajectory(
            positions, forces, 10.0, 10.0, 10.0, species, temperature=300.0, units="real"
        )

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

    def test_run_rdf_lambda_with_numpy_trajectory_state(self):
        """run_rdf_lambda should work with a real NumpyTrajectory."""
        positions = np.array([
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[1.1, 2.1, 3.1], [4.1, 5.1, 6.1], [7.1, 8.1, 9.1]],
            [[0.9, 1.9, 2.9], [3.9, 4.9, 5.9], [6.9, 7.9, 8.9]]
        ], dtype=float)
        forces = np.array([
            [[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]],
            [[0.2, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 0.2]],
            [[0.3, 0.0, 0.0], [0.0, 0.3, 0.0], [0.0, 0.0, 0.3]],
        ], dtype=float)
        species = ["H", "O", "H"]

        ts = NumpyTrajectory(
            positions, forces, 10.0, 10.0, 10.0, species, temperature=300.0, units="real"
        )

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

    def test_run_rdf_stop_none_uses_all_frames(self):
        """stop=None should use all frames (the new default behaviour)."""
        positions = np.array([
            [[1, 2, 3], [4, 5, 6]],
            [[1.1, 2.1, 3.1], [4.1, 5.1, 6.1]],
            [[0.9, 1.9, 2.9], [3.9, 4.9, 5.9]],
            [[1.2, 2.2, 3.2], [4.2, 5.2, 6.2]],
        ], dtype=float)
        forces = np.ones_like(positions) * 0.1
        species = ["H", "H"]

        ts = NumpyTrajectory(
            positions, forces, 10.0, 10.0, 10.0, species, temperature=300.0, units="real"
        )

        # With stop=None, should process all 4 frames
        result = run_rdf(
            ts,
            atom_a="H",
            atom_b="H",
            delr=1.0,
            stop=None,
            rmax=True,
        )

        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 2


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
