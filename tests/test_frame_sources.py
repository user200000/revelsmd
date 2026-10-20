"""Tests for frame source functions."""

import gc
import itertools
import weakref

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


def _numbered_trajectory(n=10):
    """A NumpyTrajectory whose frame i has every position i."""
    from revelsMD.trajectories import NumpyTrajectory
    positions = np.arange(n, dtype=float)[:, None, None] * np.ones((n, 2, 3))
    return NumpyTrajectory(positions, np.ones((n, 2, 3)), 20, 20, 20, ["A", "B"],
                           temperature=300.0)


class _FakeTrajectory:
    """Minimal trajectory stub supporting get_frame()."""

    def __init__(self, n_frames):
        self._frames = _make_frames(n_frames)

    def get_frame(self, index):
        return self._frames[index]


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


    def test_frames_are_read_as_blocks_are_iterated(self):
        pulled = []

        def stream():
            for frame in _make_frames(6):
                pulled.append(int(frame.positions[0, 0]))
                yield frame

        block = next(contiguous_blocks(stream(), block_size=3))
        assert pulled == [0]
        assert int(next(block).positions[0, 0]) == 0
        next(block)
        assert pulled == [0, 1]

    def test_requesting_a_block_before_the_last_is_read_raises(self):
        blocks = contiguous_blocks(iter(_make_frames(6)), block_size=3)
        first = next(blocks)
        next(first)
        with pytest.raises(RuntimeError, match="before the previous block was read to its end"):
            next(blocks)

    def test_an_unread_one_frame_block_is_reported(self):
        blocks = contiguous_blocks(iter(_make_frames(3)), block_size=1)
        next(blocks)
        with pytest.raises(RuntimeError, match="before the previous block was read to its end"):
            next(blocks)

    @pytest.mark.parametrize("read", [
        lambda block: list(itertools.islice(block, 3)),
        lambda block: [f for _, f in zip(range(3), block)],
    ], ids=["islice", "zip"])
    def test_consumers_that_stop_at_the_block_size_do_not_raise(self, read):
        blocks = contiguous_blocks(iter(_make_frames(7)), block_size=3)
        assert [[int(f.positions[0, 0]) for f in read(block)]
                for block in blocks] == [[0, 1, 2], [3, 4, 5], [6]]

    def test_a_full_block_does_not_read_the_next_frame(self):
        pulled = []

        def stream():
            for frame in _make_frames(6):
                pulled.append(int(frame.positions[0, 0]))
                yield frame

        block = next(contiguous_blocks(stream(), block_size=3))
        list(itertools.islice(block, 3))
        assert pulled == [0, 1, 2]

    def test_a_read_frame_is_not_kept(self):
        class Item:
            pass

        blocks = contiguous_blocks((Item() for _ in range(2)), block_size=2)
        block = next(blocks)  # the generator is kept, so its locals are too
        first = weakref.ref(next(block))
        gc.collect()
        assert first() is None


# ---------------------------------------------------------------------------
# interleaved_blocks
# ---------------------------------------------------------------------------

class TestInterleavedBlocks:
    """Tests for interleaved_blocks()."""

    def test_even_interleaving(self):
        traj = _FakeTrajectory(6)
        blocks = _frame_ids(interleaved_blocks(traj, range(6), sections=2))
        assert blocks == [[0, 2, 4], [1, 3, 5]]

    def test_range_sections_of_a_trajectory_are_read_in_order(self, monkeypatch):
        # A file loader then keeps each file open for a section's frames.
        traj = _numbered_trajectory()
        monkeypatch.setattr(type(traj), "get_frame", lambda self, i: pytest.fail("get_frame used"))
        blocks = _frame_ids(interleaved_blocks(traj, range(1, 10), sections=4))
        assert blocks == [[1, 5, 9], [2, 6], [3, 7], [4, 8]]

    @pytest.mark.parametrize("indices, expected", [
        (range(-6, 0), [[4, 6, 8], [5, 7, 9]]),  # negative indices count from the end
        (range(9, -1, -1), [[9, 7, 5, 3, 1], [8, 6, 4, 2, 0]]),  # descending
    ])
    def test_sections_outside_the_iter_frames_path_read_every_frame(self, indices, expected):
        assert _frame_ids(interleaved_blocks(_numbered_trajectory(), indices, sections=2)) == expected

    def test_a_section_past_the_last_frame_raises(self):
        with pytest.raises(IndexError):
            _frame_ids(interleaved_blocks(_numbered_trajectory(), range(5, 12), sections=2))

    def test_a_section_is_read_whole_after_the_blocks_are_dropped(self):
        # A section belongs to whoever reads it, however it was obtained.
        section = next(interleaved_blocks(_numbered_trajectory(), range(10), sections=3))
        gc.collect()
        assert [int(f.positions[0, 0]) for f in section] == [0, 3, 6, 9]
        first, second = itertools.islice(interleaved_blocks(_numbered_trajectory(), range(10), 3), 2)
        next(first)
        gc.collect()
        assert _frame_ids([first, second]) == [[3, 6, 9], [1, 4, 7]]

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
# Frame dataclass
# ---------------------------------------------------------------------------

class TestFrame:
    """Tests for the Frame dataclass."""

    def test_single_precision_arrays_are_stored_in_double_precision(self):
        """MDAnalysis gives float32; a frame holds the same values as float64."""
        from revelsMD.frame_sources import Frame
        positions = np.array([[1.1, 2.2, 3.3]], dtype=np.float32)
        forces = np.array([[0.1, -0.2, 0.3]], dtype=np.float32)
        frame = Frame(positions, forces)
        assert frame.positions.dtype == frame.forces.dtype == np.float64
        np.testing.assert_array_equal(frame.positions, positions.astype(np.float64))
        np.testing.assert_array_equal(frame.forces, forces.astype(np.float64))

    def test_a_frame_is_independent_of_the_arrays_it_was_built_from(self):
        """Overwriting the input arrays after construction leaves the frame unchanged."""
        from revelsMD.frame_sources import Frame
        positions = np.array([[1.0, 2.0, 3.0]])
        forces = np.array([[0.1, 0.2, 0.3]])
        frame = Frame(positions, forces)
        positions[:] = 0.0
        forces[:] = 0.0
        np.testing.assert_array_equal(frame.positions, [[1.0, 2.0, 3.0]])
        np.testing.assert_array_equal(frame.forces, [[0.1, 0.2, 0.3]])

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
        with pytest.raises(ValueError, match="mismatched atom counts"):
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
