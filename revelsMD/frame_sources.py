"""Frame source functions for grouping trajectory frames into blocks.

This module provides the layer between trajectory loaders and analysis code.
Trajectory loaders read frames from disk; frame sources group those frames
into blocks for statistical analysis (e.g. Welford variance estimation).

Both functions yield the same interface: an iterator of blocks, where each
block is an iterator of Frame instances.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

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
                f"Mismatched atom counts: positions has "
                f"{self.positions.shape[0]}, "
                f"forces has {self.forces.shape[0]}"
            )

#: An iterator of blocks, where each block is an iterator of frames.
BlockSource = Iterator[Iterator[Frame]]


def contiguous_blocks(
    frame_iterator: Iterator[Frame],
    block_size: int,
) -> BlockSource:
    """Yield contiguous blocks of frames from a sequential stream.

    Each block contains up to ``block_size`` frames drawn from the
    underlying iterator. The final block may contain fewer frames if the
    stream is not evenly divisible.

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
    """
    if block_size < 1:
        raise ValueError("block_size must be >= 1")

    it = iter(frame_iterator)
    # Materialise each batch so that unconsumed inner iterators
    # cannot silently lose frames from the shared stream.
    while batch := list(itertools.islice(it, block_size)):
        yield iter(batch)


def interleaved_blocks(
    trajectory: Trajectory,
    frame_indices: Sequence[int],
    sections: int,
) -> BlockSource:
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
        Number of interleaved sections. Must be >= 1.

    Yields
    ------
    iterator of Frame
        One block of frames. Each block is an independent generator;
        blocks do not share iteration state.

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
            "random frame access (get_frame). Use blocking='contiguous' "
            "for sequential-only backends."
        )

    for k in range(sections):
        section_indices = frame_indices[k::sections]
        if len(section_indices) == 0:
            continue  # defensive: callers should not request sections > frames
        yield (trajectory.get_frame(i) for i in section_indices)
