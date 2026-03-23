"""Tests for frame source functions."""

import numpy as np
import pytest

from revelsMD.frame_sources import contiguous_blocks, interleaved_blocks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_frames(n):
    """Return a list of Frame instances with identifiable data."""
    from revelsMD.frame_sources import Frame
    return [Frame(np.array([[i, 0, 0]], dtype=float), np.array([[i, 0, 0]], dtype=float)) for i in range(n)]


def _frame_ids(blocks):
    """Extract the integer id from each frame across all blocks.

    Returns a list of lists, one per block.
    """
    result = []
    for block in blocks:
        ids = [int(frame.positions[0, 0]) for frame in block]
        result.append(ids)
    return result


class _FakeTrajectory:
    """Minimal trajectory stub supporting get_frame()."""

    def __init__(self, n_frames):
        self._frames = _make_frames(n_frames)

    def get_frame(self, index):
        return self._frames[index]


class _SequentialOnlyTrajectory:
    """Trajectory stub without get_frame()."""
    pass


# ---------------------------------------------------------------------------
# contiguous_blocks
# ---------------------------------------------------------------------------

class TestContiguousBlocks:
    """Tests for contiguous_blocks()."""

    def test_even_split(self):
        frames = _make_frames(6)
        blocks = _frame_ids(contiguous_blocks(iter(frames), block_size=3))
        assert blocks == [[0, 1, 2], [3, 4, 5]]

    def test_remainder_block(self):
        frames = _make_frames(7)
        blocks = _frame_ids(contiguous_blocks(iter(frames), block_size=3))
        assert blocks == [[0, 1, 2], [3, 4, 5], [6]]

    def test_block_size_one(self):
        frames = _make_frames(3)
        blocks = _frame_ids(contiguous_blocks(iter(frames), block_size=1))
        assert blocks == [[0], [1], [2]]

    def test_block_size_equals_total(self):
        frames = _make_frames(5)
        blocks = _frame_ids(contiguous_blocks(iter(frames), block_size=5))
        assert blocks == [[0, 1, 2, 3, 4]]

    def test_block_size_exceeds_total(self):
        frames = _make_frames(3)
        blocks = _frame_ids(contiguous_blocks(iter(frames), block_size=10))
        assert blocks == [[0, 1, 2]]

    def test_empty_iterator(self):
        blocks = list(contiguous_blocks(iter([]), block_size=3))
        assert blocks == []

    def test_single_frame(self):
        frames = _make_frames(1)
        blocks = _frame_ids(contiguous_blocks(iter(frames), block_size=5))
        assert blocks == [[0]]

    def test_invalid_block_size_zero(self):
        with pytest.raises(ValueError, match="block_size must be >= 1"):
            list(contiguous_blocks(iter([]), block_size=0))

    def test_invalid_block_size_negative(self):
        with pytest.raises(ValueError, match="block_size must be >= 1"):
            list(contiguous_blocks(iter([]), block_size=-1))

    def test_yields_single_use_iterators(self):
        """Each block should be a single-use iterator of frames."""
        frames = _make_frames(6)
        block_iter = contiguous_blocks(iter(frames), block_size=3)
        # Get first block but don't consume it
        first_block = next(block_iter)
        # Consuming should give the right frames
        assert [int(frame.positions[0, 0]) for frame in first_block] == [0, 1, 2]


# ---------------------------------------------------------------------------
# interleaved_blocks
# ---------------------------------------------------------------------------

class TestInterleavedBlocks:
    """Tests for interleaved_blocks()."""

    def test_even_interleaving(self):
        traj = _FakeTrajectory(6)
        blocks = _frame_ids(interleaved_blocks(traj, range(6), sections=2))
        assert blocks == [[0, 2, 4], [1, 3, 5]]

    def test_three_sections(self):
        traj = _FakeTrajectory(9)
        blocks = _frame_ids(interleaved_blocks(traj, range(9), sections=3))
        assert blocks == [[0, 3, 6], [1, 4, 7], [2, 5, 8]]

    def test_uneven_sections(self):
        traj = _FakeTrajectory(7)
        blocks = _frame_ids(interleaved_blocks(traj, range(7), sections=3))
        assert blocks == [[0, 3, 6], [1, 4], [2, 5]]

    def test_sections_equals_frames(self):
        """One frame per section."""
        traj = _FakeTrajectory(3)
        blocks = _frame_ids(interleaved_blocks(traj, range(3), sections=3))
        assert blocks == [[0], [1], [2]]

    def test_single_section(self):
        traj = _FakeTrajectory(4)
        blocks = _frame_ids(interleaved_blocks(traj, range(4), sections=1))
        assert blocks == [[0, 1, 2, 3]]

    def test_with_non_zero_start_range(self):
        """frame_indices can be an offset range."""
        traj = _FakeTrajectory(10)
        blocks = _frame_ids(interleaved_blocks(traj, range(2, 8), sections=2))
        assert blocks == [[2, 4, 6], [3, 5, 7]]

    def test_with_strided_range(self):
        traj = _FakeTrajectory(10)
        blocks = _frame_ids(interleaved_blocks(traj, range(0, 10, 2), sections=2))
        # range(0,10,2) = [0,2,4,6,8]; interleaved with 2 sections: [0,4,8], [2,6]
        assert blocks == [[0, 4, 8], [2, 6]]

    def test_raises_without_get_frame(self):
        traj = _SequentialOnlyTrajectory()
        with pytest.raises(ValueError, match="random frame access"):
            list(interleaved_blocks(traj, range(5), sections=2))

    def test_invalid_sections_zero(self):
        traj = _FakeTrajectory(5)
        with pytest.raises(ValueError, match="sections must be >= 1"):
            list(interleaved_blocks(traj, range(5), sections=0))

    def test_sections_exceeding_frames_yields_sparse_blocks(self):
        """More sections than frames: some sections are empty and skipped."""
        traj = _FakeTrajectory(3)
        blocks = _frame_ids(interleaved_blocks(traj, range(3), sections=5))
        assert blocks == [[0], [1], [2]]


# ---------------------------------------------------------------------------
# Frame NamedTuple
# ---------------------------------------------------------------------------

class TestFrame:
    """Tests for the Frame dataclass."""

    def test_named_field_access(self):
        """Frame supports .positions and .forces attribute access."""
        from revelsMD.frame_sources import Frame
        pos = np.array([[1.0, 2.0, 3.0]])
        frc = np.array([[0.1, 0.2, 0.3]])
        frame = Frame(positions=pos, forces=frc)
        np.testing.assert_array_equal(frame.positions, pos)
        np.testing.assert_array_equal(frame.forces, frc)

    def test_mismatched_atom_counts_raises(self):
        """Frame rejects positions and forces with different atom counts."""
        from revelsMD.frame_sources import Frame
        positions = np.array([[1.0, 2.0, 3.0]])
        forces = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        with pytest.raises(ValueError, match="Mismatched atom counts"):
            Frame(positions=positions, forces=forces)

    def test_wrong_shape_raises(self):
        """Frame rejects arrays that are not (n_atoms, 3)."""
        from revelsMD.frame_sources import Frame
        with pytest.raises(ValueError, match="must have shape"):
            Frame(positions=np.array([1.0, 2.0, 3.0]),
                  forces=np.array([0.1, 0.2, 0.3]))

    def test_iter_frames_returns_frame_instances(self, ts):
        """iter_frames should yield Frame instances with named access."""
        from revelsMD.frame_sources import Frame
        for frame in ts.iter_frames():
            assert isinstance(frame, Frame)
            assert frame.positions is not None
            assert frame.forces is not None

    def test_get_frame_returns_frame_instance(self, ts):
        """get_frame should return a Frame instance with named access."""
        from revelsMD.frame_sources import Frame
        frame = ts.get_frame(0)
        assert isinstance(frame, Frame)
        assert frame.positions.shape == (2, 3)
        assert frame.forces.shape == (2, 3)
