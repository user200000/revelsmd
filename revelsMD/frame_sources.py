"""Frame source functions for grouping trajectory frames into blocks.

This module provides the layer between trajectory loaders and analysis code.
Trajectory loaders read frames from disk; frame sources group those frames
into blocks for statistical analysis (e.g. Welford variance estimation).

Both functions yield the same interface: an iterable of blocks, where each
block is an iterable of (positions, forces) tuples.
"""

import itertools
from collections.abc import Iterator, Sequence

import numpy as np


def contiguous_blocks(
    frame_iterator: Iterator[tuple[np.ndarray, np.ndarray]],
    block_size: int,
) -> Iterator[Iterator[tuple[np.ndarray, np.ndarray]]]:
    """Yield contiguous blocks of frames from a sequential stream.

    Each block contains up to ``block_size`` frames drawn from the
    underlying iterator.  The final block may contain fewer frames if the
    stream is not evenly divisible.

    Parameters
    ----------
    frame_iterator : iterator of (positions, forces)
        Sequential frame stream, e.g. from ``trajectory.iter_frames()``.
    block_size : int
        Maximum number of frames per block.  Must be >= 1.

    Yields
    ------
    iterator of (positions, forces)
        One block of frames.
    """
    if block_size < 1:
        raise ValueError("block_size must be >= 1")

    it = iter(frame_iterator)
    while batch := list(itertools.islice(it, block_size)):
        yield iter(batch)


def interleaved_blocks(
    trajectory,
    frame_indices: Sequence[int],
    sections: int,
) -> Iterator[Iterator[tuple[np.ndarray, np.ndarray]]]:
    """Yield blocks of frames using an interleaved index pattern.

    Section *k* receives frames at indices ``frame_indices[k::sections]``.
    This requires random access via ``trajectory.get_frame()``.

    Parameters
    ----------
    trajectory : Trajectory
        Trajectory object with a ``get_frame(index)`` method.
    frame_indices : range or sequence of int
        Frame indices to distribute across sections (supports slicing).
    sections : int
        Number of interleaved sections.  Must be >= 1.

    Yields
    ------
    iterator of (positions, forces)
        One block of frames.  Each block is a single-use generator that
        must be fully consumed before advancing to the next.

    Raises
    ------
    ValueError
        If the trajectory does not support random frame access.
    """
    if sections < 1:
        raise ValueError("sections must be >= 1")

    if not callable(getattr(trajectory, "get_frame", None)):
        raise ValueError(
            "Interleaved blocking requires a trajectory that supports "
            "random frame access (get_frame).  Use blocking='contiguous' "
            "for sequential-only backends."
        )

    for k in range(sections):
        section_indices = frame_indices[k::sections]
        if len(section_indices) == 0:
            continue  # defensive: callers should not request sections > frames
        yield (trajectory.get_frame(i) for i in section_indices)
