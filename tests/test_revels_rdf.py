import pytest
import numpy as np
from revelsMD.rdf import run_rdf, run_rdf_lambda, single_frame_rdf
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

        # 3 frames × 3 atoms × 3 coordinates
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
# single_frame_rdf (like pairs)
# -------------------------------

def test_single_frame_rdf_like(ts):
    bins = np.linspace(0, 5, 10)
    indices_h = ts.get_indices("H")
    positions, forces = ts.get_frame(0)
    result = single_frame_rdf(
        positions,
        forces,
        [indices_h, indices_h],
        ts.box_x,
        ts.box_y,
        ts.box_z,
        bins,
    )
    assert result.shape == bins.shape
    assert np.isfinite(result).all()
    assert np.all(np.isreal(result))


# -------------------------------
# single_frame_rdf (unlike pairs)
# -------------------------------

def test_single_frame_rdf_unlike(ts):
    bins = np.linspace(0, 5, 10)
    indices = [ts.get_indices("H"), ts.get_indices("O")]
    positions, forces = ts.get_frame(0)
    result = single_frame_rdf(
        positions,
        forces,
        indices,
        ts.box_x,
        ts.box_y,
        ts.box_z,
        bins,
    )
    assert result.shape == bins.shape
    assert np.isfinite(result).all()
    assert np.all(np.isreal(result))


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
    # Three columns: [r, combined RDF, λ]
    assert result.shape[1] == 3
    assert np.all(np.isfinite(result))
    assert np.all((result[:, 2] >= -1) & (result[:, 2] <= 2))  # λ values roughly bounded


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
# MIC / distance correctness tests
# -------------------------------

class TestMinimumImageConvention:
    """
    Test that pairwise distances are computed correctly under periodic boundaries.

    These tests use single_frame_rdf with two atoms at known positions.
    The RDF estimator computes F[j]·r_ij/|r|³ for each pair (i,j), where
    r_ij = pos[j] - pos[i]. We verify both that contributions land in the
    correct bin AND have the expected magnitude.

    Sign convention: for pair (i,j), r_ij points from i to j. If F[j] points
    in the opposite direction (toward i), the dot product is negative.
    """

    def test_direct_distance_no_wrapping(self):
        """Two atoms separated by 3.0 in x, well within half-box."""
        box = 20.0
        r = 3.0
        positions = np.array([[1.0, 5.0, 5.0], [4.0, 5.0, 5.0]])
        # Forces pointing toward each other (attractive)
        # Atom 0 at x=1 pushes right (+x), atom 1 at x=4 pushes left (-x)
        forces = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
        indices = np.array([0, 1])

        bins = np.array([r - 0.5, r + 0.5, r + 1.5])

        # Use indices.copy() to ensure value comparison works (not identity)
        result = single_frame_rdf(
            positions, forces, [indices, indices.copy()], box, box, box, bins
        )

        # Upper triangle: only pair (0,1) computed
        # r_01 = (3,0,0), F_diff = F[1] - F[0] = (-2,0,0)
        # dot = -6, contrib = -6/27 = -2/9
        expected = -2.0 / (r * r)
        assert np.isclose(result[0], expected), f"Expected {expected}, got {result[0]}"
        assert result[1] == 0, "No contribution expected in second bin"

    def test_wrapped_distance_across_boundary(self):
        """
        Two atoms at x=1 and x=19 in a box of 20.
        Direct distance is 18, but MIC distance should be 2.
        """
        box = 20.0
        mic_r = 2.0  # After MIC wrapping
        positions = np.array([[1.0, 5.0, 5.0], [19.0, 5.0, 5.0]])
        # After MIC: atom 1 is effectively at x=-1 (or equivalently, r_01 = (-2,0,0))
        # Forces pointing toward each other
        forces = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        indices = np.array([0, 1])

        bins = np.array([mic_r - 0.5, mic_r + 0.5, 15.0, 20.0])

        # Use indices.copy() to ensure value comparison works (not identity)
        result = single_frame_rdf(
            positions, forces, [indices, indices.copy()], box, box, box, bins
        )

        # MIC wraps to r=2. Both pairs contribute -1/r² each.
        expected = -2.0 / (mic_r * mic_r)
        assert np.isclose(result[0], expected), \
            f"MIC distance should be {mic_r}, got contribution {result[0]}"
        assert result[2] == 0, "No contribution at unwrapped distance"

    def test_wrapped_distance_negative_direction(self):
        """
        Two atoms at x=18 and x=2 in a box of 20.
        Direct distance is 16, but MIC distance should be 4.
        """
        box = 20.0
        mic_r = 4.0
        positions = np.array([[18.0, 5.0, 5.0], [2.0, 5.0, 5.0]])
        # After MIC: shortest path is +4 in x (wrapping around)
        forces = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
        indices = np.array([0, 1])

        bins = np.array([mic_r - 0.5, mic_r + 0.5, 15.0, 20.0])

        # Use indices.copy() to ensure value comparison works (not identity)
        result = single_frame_rdf(
            positions, forces, [indices, indices.copy()], box, box, box, bins
        )

        expected = -2.0 / (mic_r * mic_r)
        assert np.isclose(result[0], expected), \
            f"Expected {expected}, got {result[0]}"

    def test_3d_diagonal_wrapping(self):
        """
        Atoms near opposite corners, requiring wrapping in all dimensions.
        Positions: (1,1,1) and (19,19,19) in box of 20.
        Direct distance: sqrt(18² + 18² + 18²) = 31.18
        MIC distance: sqrt(2² + 2² + 2²) ≈ 3.46
        """
        box = 20.0
        mic_r = np.sqrt(12)  # sqrt(2² + 2² + 2²) ≈ 3.46
        positions = np.array([[1.0, 1.0, 1.0], [19.0, 19.0, 19.0]])
        # Unit force along diagonal, pointing toward each other
        f = 1.0 / np.sqrt(3)
        forces = np.array([[-f, -f, -f], [f, f, f]])
        indices = np.array([0, 1])

        bins = np.array([mic_r - 0.5, mic_r + 0.5, 30.0, 35.0])

        # Use indices.copy() to ensure value comparison works (not identity)
        result = single_frame_rdf(
            positions, forces, [indices, indices.copy()], box, box, box, bins
        )

        expected = -2.0 / (mic_r * mic_r)
        assert np.isclose(result[0], expected, rtol=1e-10), \
            f"Expected {expected}, got {result[0]}"
        assert result[2] == 0, "No contribution at unwrapped distance"

    @pytest.mark.parametrize("axis", [0, 1, 2])
    def test_wrapping_each_axis(self, axis):
        """
        Test MIC wrapping works correctly for each axis independently.
        Atoms at 1 and 19 along the test axis, centred on other axes.
        """
        box = 20.0
        mic_r = 2.0

        pos0 = [10.0, 10.0, 10.0]
        pos1 = [10.0, 10.0, 10.0]
        pos0[axis] = 1.0
        pos1[axis] = 19.0
        positions = np.array([pos0, pos1])

        # Forces pointing toward each other along this axis
        f0 = [0.0, 0.0, 0.0]
        f1 = [0.0, 0.0, 0.0]
        f0[axis] = -1.0
        f1[axis] = 1.0
        forces = np.array([f0, f1])

        indices = np.array([0, 1])
        bins = np.array([mic_r - 0.5, mic_r + 0.5, 15.0, 20.0])

        result = single_frame_rdf(
            positions, forces, [indices, indices], box, box, box, bins
        )

        expected = -2.0 / (mic_r * mic_r)
        assert np.isclose(result[0], expected), \
            f"Axis {axis}: expected {expected}, got {result[0]}"
        assert result[2] == 0, f"Axis {axis}: no contribution at unwrapped distance"

    def test_exactly_half_box_boundary(self):
        """
        Edge case: atoms separated by exactly half the box length.
        At r = L/2, the MIC is ambiguous but should still produce a valid result.
        """
        box = 20.0
        half_box = box / 2.0  # r = 10.0

        positions = np.array([[5.0, 10.0, 10.0], [15.0, 10.0, 10.0]])
        forces = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
        indices = np.array([0, 1])

        bins = np.array([half_box - 0.5, half_box + 0.5, half_box + 1.5])

        result = single_frame_rdf(
            positions, forces, [indices, indices], box, box, box, bins
        )

        # At exactly half-box, distance should be 10.0
        expected = -2.0 / (half_box * half_box)
        assert np.isclose(result[0], expected), \
            f"Half-box boundary: expected {expected}, got {result[0]}"

    def test_non_cubic_box(self):
        """
        Test MIC with different box dimensions in each direction.
        """
        box_x, box_y, box_z = 20.0, 30.0, 40.0

        # Atoms requiring wrapping in x (short axis) but not y or z
        positions = np.array([[1.0, 15.0, 20.0], [19.0, 15.0, 20.0]])
        forces = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        indices = np.array([0, 1])

        mic_r = 2.0  # Wrapped distance in x
        bins = np.array([mic_r - 0.5, mic_r + 0.5, 15.0, 20.0])

        result = single_frame_rdf(
            positions, forces, [indices, indices], box_x, box_y, box_z, bins
        )

        expected = -2.0 / (mic_r * mic_r)
        assert np.isclose(result[0], expected), \
            f"Non-cubic box: expected {expected}, got {result[0]}"


class TestForceProjection:
    """
    Test that the force projection (F·r / r³) is computed correctly.

    The RDF estimator uses force-weighted contributions. For pair (i,j):
    contribution = F[j] · (pos[j] - pos[i]) / |r|³
    """

    def test_radial_force_contribution(self):
        """
        Two atoms separated along x-axis with asymmetric forces.

        Atoms at x=0 and x=2. Force on atom 1 = (-1,0,0), force on atom 0 = (0,0,0).
        For pair (0,1): r_01 = (2,0,0), F[1] = (-1,0,0)
            F·r = -2, |r| = 2, F·r/|r|³ = -2/8 = -0.25
        For pair (1,0): r_10 = (-2,0,0), F[0] = (0,0,0)
            F·r = 0
        Total contribution = -0.25
        """
        box = 20.0
        r = 2.0
        positions = np.array([[0.0, 5.0, 5.0], [2.0, 5.0, 5.0]])
        forces = np.array([[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
        indices = np.array([0, 1])

        bins = np.array([r - 0.5, r + 0.5, r + 1.5])

        result = single_frame_rdf(
            positions, forces, [indices, indices], box, box, box, bins
        )

        expected = -0.25
        assert np.isclose(result[0], expected), f"Expected {expected}, got {result[0]}"

    def test_perpendicular_force_no_contribution(self):
        """
        Force perpendicular to separation should give zero dot product.
        Atoms separated along x, force along y.
        """
        box = 20.0
        r = 3.0
        positions = np.array([[0.0, 5.0, 5.0], [3.0, 5.0, 5.0]])
        forces = np.array([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]])
        indices = np.array([0, 1])

        bins = np.array([r - 0.5, r + 0.5, r + 1.5])

        result = single_frame_rdf(
            positions, forces, [indices, indices], box, box, box, bins
        )

        assert result[0] == 0, "Perpendicular force should give zero contribution"
        assert result[1] == 0


class TestUnlikePairRDF:
    """
    Test single_frame_rdf correctness for unlike pairs.

    Unlike pairs use force differences: (F[A] - F[B]) · r_AB / |r|³
    where r_AB = pos[A] - pos[B] (species A position minus species B position).

    Each (A,B) cross-pair contributes once (no double-counting like in like-pairs).
    """

    def test_direct_distance_no_wrapping(self):
        """
        One atom of species A at x=1, one atom of species B at x=4.
        Separation r=3 along x-axis.
        """
        box = 20.0
        r = 3.0
        # Atom 0 is species A, atom 1 is species B
        positions = np.array([[1.0, 5.0, 5.0], [4.0, 5.0, 5.0]])
        # Forces pointing toward each other
        forces = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
        indices = [np.array([0]), np.array([1])]

        bins = np.array([r - 0.5, r + 0.5, r + 1.5])

        result = single_frame_rdf(
            positions, forces, indices, box, box, box, bins
        )

        # r_AB = pos[A] - pos[B] = (1,5,5) - (4,5,5) = (-3, 0, 0)
        # F_diff = F[A] - F[B] = (1,0,0) - (-1,0,0) = (2, 0, 0)
        # dot = 2 * (-3) = -6
        # contribution = -6 / 27 = -2/9
        expected = -2.0 / (r * r)
        assert np.isclose(result[0], expected), f"Expected {expected}, got {result[0]}"

    def test_wrapped_distance_across_boundary(self):
        """
        Species A at x=1, species B at x=19 in box of 20.
        MIC distance should be 2.
        """
        box = 20.0
        mic_r = 2.0
        positions = np.array([[1.0, 5.0, 5.0], [19.0, 5.0, 5.0]])
        # Forces pointing toward each other through the boundary
        forces = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        indices = [np.array([0]), np.array([1])]

        bins = np.array([mic_r - 0.5, mic_r + 0.5, 15.0, 20.0])

        result = single_frame_rdf(
            positions, forces, indices, box, box, box, bins
        )

        # After MIC: r_AB = pos[A] - pos[B] wraps to (+2, 0, 0)
        # F_diff = F[A] - F[B] = (-1,0,0) - (1,0,0) = (-2, 0, 0)
        # dot = (-2)*(+2) = -4
        # contribution = -4 / 8 = -0.5
        expected = -0.5
        assert np.isclose(result[0], expected), \
            f"Expected {expected}, got {result[0]}"

    @pytest.mark.parametrize("axis", [0, 1, 2])
    def test_wrapping_each_axis(self, axis):
        """Test MIC wrapping for unlike pairs works on each axis."""
        box = 20.0
        mic_r = 2.0

        pos_a = [10.0, 10.0, 10.0]
        pos_b = [10.0, 10.0, 10.0]
        pos_a[axis] = 1.0
        pos_b[axis] = 19.0
        positions = np.array([pos_a, pos_b])

        # Forces pointing toward each other through the boundary
        f_a = [0.0, 0.0, 0.0]
        f_b = [0.0, 0.0, 0.0]
        f_a[axis] = -1.0
        f_b[axis] = 1.0
        forces = np.array([f_a, f_b])

        indices = [np.array([0]), np.array([1])]
        bins = np.array([mic_r - 0.5, mic_r + 0.5, 15.0, 20.0])

        result = single_frame_rdf(
            positions, forces, indices, box, box, box, bins
        )

        # After MIC: r_AB[axis] = pos[A] - pos[B] wraps to +2
        # F_diff[axis] = F[A] - F[B] = -1 - 1 = -2
        # dot = (-2)*(+2) = -4
        # contribution = -4 / 8 = -0.5
        expected = -0.5
        assert np.isclose(result[0], expected), \
            f"Axis {axis}: expected {expected}, got {result[0]}"

    def test_multiple_atoms_per_species(self):
        """
        Two atoms of species A, one atom of species B.
        Should sum contributions from both A-B pairs.
        """
        box = 20.0
        r = 2.0
        # A atoms at x=0 and x=4, B atom at x=2
        positions = np.array([
            [0.0, 5.0, 5.0],  # A[0]
            [4.0, 5.0, 5.0],  # A[1]
            [2.0, 5.0, 5.0],  # B[0]
        ])
        # All forces along x-axis pointing inward
        forces = np.array([
            [1.0, 0.0, 0.0],   # A[0] pushes right
            [-1.0, 0.0, 0.0],  # A[1] pushes left
            [0.0, 0.0, 0.0],   # B[0] no force
        ])
        indices = [np.array([0, 1]), np.array([2])]

        bins = np.array([r - 0.5, r + 0.5, r + 1.5])

        result = single_frame_rdf(
            positions, forces, indices, box, box, box, bins
        )

        # Pair (A[0], B[0]): r = (0,5,5) - (2,5,5) = (-2,0,0)
        #   F_diff = (1,0,0) - (0,0,0) = (1,0,0)
        #   dot = 1*(-2) = -2, contrib = -2/8 = -0.25
        # Pair (A[1], B[0]): r = (4,5,5) - (2,5,5) = (2,0,0)
        #   F_diff = (-1,0,0) - (0,0,0) = (-1,0,0)
        #   dot = (-1)*2 = -2, contrib = -2/8 = -0.25
        # Total = -0.5
        expected = -0.5
        assert np.isclose(result[0], expected), f"Expected {expected}, got {result[0]}"

    def test_perpendicular_force_no_contribution(self):
        """Force perpendicular to separation gives zero contribution."""
        box = 20.0
        r = 3.0
        positions = np.array([[0.0, 5.0, 5.0], [3.0, 5.0, 5.0]])
        # Forces along y, separation along x
        forces = np.array([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]])
        indices = [np.array([0]), np.array([1])]

        bins = np.array([r - 0.5, r + 0.5, r + 1.5])

        result = single_frame_rdf(
            positions, forces, indices, box, box, box, bins
        )

        assert result[0] == 0, "Perpendicular force should give zero contribution"

    def test_non_cubic_box(self):
        """Test MIC with different box dimensions for unlike pairs."""
        box_x, box_y, box_z = 20.0, 30.0, 40.0
        mic_r = 2.0

        positions = np.array([[1.0, 15.0, 20.0], [19.0, 15.0, 20.0]])
        forces = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        indices = [np.array([0]), np.array([1])]

        bins = np.array([mic_r - 0.5, mic_r + 0.5, 15.0, 20.0])

        result = single_frame_rdf(
            positions, forces, indices, box_x, box_y, box_z, bins
        )

        # After MIC: r_AB = pos[A] - pos[B] wraps to (+2, 0, 0)
        # F_diff = F[A] - F[B] = (-1,0,0) - (1,0,0) = (-2, 0, 0)
        # dot = (-2)*(+2) = -4, contrib = -4/8 = -0.5
        expected = -0.5
        assert np.isclose(result[0], expected), \
            f"Non-cubic box: expected {expected}, got {result[0]}"
