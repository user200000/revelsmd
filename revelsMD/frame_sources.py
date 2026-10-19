"""Frame source functions for grouping trajectory frames into blocks.

This module provides the layer between trajectory loaders and analysis code.
Trajectory loaders read frames from disk; frame sources group those frames
into blocks for statistical analysis (e.g. Welford variance estimation).

Both functions yield the same interface: an iterator of blocks, where each
block is an iterator of Frame instances.
"""

from __future__ import annotations

import itertools
from collections.abc import Generator, Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from revelsMD.trajectories._base import Trajectory


@dataclass(frozen=True, slots=True, eq=False)
class Frame:
    """A single trajectory frame with named field access.

    Both arrays must have shape ``(n_atoms, 3)`` with matching ``n_atoms``.

    Parameters
    ----------
    positions : np.ndarray, shape (n_atoms, 3)
        Atomic positions in Cartesian coordinates.
    forces : np.ndarray, shape (n_atoms, 3)
        Atomic forces in Cartesian coordinates.

    Attributes
    ----------
    positions : np.ndarray, shape (n_atoms, 3)
        Atomic positions in Cartesian coordinates.
    forces : np.ndarray, shape (n_atoms, 3)
        Atomic forces in Cartesian coordinates.

    Raises
    ------
    TypeError
        If either field is not a numpy array.
    ValueError
        If either array does not have shape ``(n_atoms, 3)``, or if the
        atom counts differ between ``positions`` and ``forces``.
    """

    positions: np.ndarray
    forces: np.ndarray

    def __post_init__(self):
        for name, arr in (("positions", self.positions), ("forces", self.forces)):
            if not isinstance(arr, np.ndarray):
                raise TypeError(
                    f"{name} must be a numpy array, "
                    f"got {type(arr).__name__}"
                )
            if arr.ndim != 2 or arr.shape[1] != 3:
                raise ValueError(
                    f"{name} must have shape (n_atoms, 3), "
                    f"got {arr.shape}"
                )
        if self.positions.shape[0] != self.forces.shape[0]:
            raise ValueError(
                f"mismatched atom counts: positions has "
                f"{self.positions.shape[0]}, "
                f"forces has {self.forces.shape[0]}"
            )

class Block(Protocol):
    """An iterator of frames that its reader closes when done with it."""

    def __iter__(self) -> Iterator[Frame]: ...

    def __next__(self) -> Frame: ...

    def close(self) -> None: ...


#: A generator of blocks. Whatever reads a block closes it, which releases
#: any file the block holds open.
BlockSource = Generator[Block, None, None]


def contiguous_blocks(
    frame_iterator: Iterator[Frame],
    block_size: int,
) -> BlockSource:
    """Yield contiguous blocks of frames from a sequential stream.

    Each block contains up to ``block_size`` frames drawn from the
    underlying iterator. The final block may contain fewer frames if the
    stream is not evenly divisible. Frames are read as each block is
    iterated, so only one is held at a time. All blocks read the same
    stream, so read each block to its end before requesting the next; the
    blocks cannot be collected first, as ``list(contiguous_blocks(...))``
    would.

    Parameters
    ----------
    frame_iterator : iterator of Frame
        Sequential frame stream, e.g. from ``trajectory.iter_frames()``.
    block_size : int
        Maximum number of frames per block. Must be >= 1.

    Yields
    ------
    iterator of Frame
        One block of frames.

    Raises
    ------
    RuntimeError
        If a block is requested before the previous block was read to its end.
    """
    if block_size < 1:
        raise ValueError("block_size must be >= 1")

    it = iter(frame_iterator)
    for first in it:
        block = _Block(first, it, block_size)
        del first  # the block holds the first frame only until it is read
        yield block
        if not block.read_to_end():
            raise RuntimeError(
                "a block of frames was requested before the previous block was "
                "read to its end"
            )


class _Block:
    """Up to ``size`` frames of a shared stream, starting with ``first``."""

    def __init__(self, first: Frame, stream: Iterator[Frame], size: int) -> None:
        self._first: Frame | None = first
        self._rest = itertools.islice(stream, size - 1)

    def __iter__(self) -> "_Block":
        return self

    def __next__(self) -> Frame:
        if self._first is not None:
            frame, self._first = self._first, None
            return frame
        return next(self._rest)

    def close(self) -> None:
        """Nothing to release: the stream belongs to the caller of contiguous_blocks."""

    def read_to_end(self) -> bool:
        """Whether every frame of the block has been read.

        If a frame is left, this reads it from the stream; the caller then
        raises, so the frame is not needed.
        """
        return self._first is None and next(self._rest, None) is None


def interleaved_blocks(
    trajectory: Trajectory,
    frame_indices: Sequence[int],
    sections: int,
) -> BlockSource:
    """Yield blocks of frames using an interleaved index pattern.

    Section *k* receives frames at indices ``frame_indices[k::sections]``.
    A section that is a range of valid indices is read with
    ``trajectory.iter_frames()``, which keeps each file open while it reads
    the file's frames; other sections are read with ``trajectory.get_frame()``.

    Parameters
    ----------
    trajectory : Trajectory
        The trajectory to read.
    frame_indices : range or sequence of int
        Frame indices to distribute across sections (supports slicing).
    sections : int
        Number of interleaved sections. Must be >= 1.

    Yields
    ------
    iterator of Frame
        One block of frames. Each block is an independent generator;
        blocks do not share iteration state. A part-read block holds its
        file open until it is closed.

    Raises
    ------
    ValueError
        If ``sections`` is less than 1.
    """
    # Imported here: the trajectories package imports this module.
    from revelsMD.trajectories._base import Trajectory

    if sections < 1:
        raise ValueError("sections must be >= 1")

    for k in range(sections):
        section_indices = frame_indices[k::sections]
        if len(section_indices) == 0:
            continue  # defensive: callers should not request sections > frames
        if (
            isinstance(trajectory, Trajectory)
            and isinstance(section_indices, range)
            and section_indices.step > 0
            and 0 <= section_indices[0]
            and section_indices[-1] < trajectory.frames
        ):
            yield trajectory.iter_frames(
                section_indices[0], section_indices[-1] + 1, section_indices.step
            )
        else:
            yield (trajectory.get_frame(i) for i in section_indices)
