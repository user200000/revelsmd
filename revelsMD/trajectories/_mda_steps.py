"""
Frame order of MDAnalysis trajectory files.

Each file's frames are ordered by integer step where the format records
one (TRR, H5MD, TNG), and otherwise by time (AMBER NetCDF). The keys of
every frame are read when the trajectory is constructed: from the frame
headers for TRR, the step and time datasets for H5MD and the time variable
for NetCDF, and by reading each frame for other formats. The files are then
joined by :func:`revelsMD.trajectories._base.join_restarts`, as LAMMPS dumps are.
"""

import os
import struct
from dataclasses import dataclass

import numpy as np
from MDAnalysis.coordinates.H5MD import H5MDReader  # type: ignore[import-untyped]
from MDAnalysis.coordinates.TRJ import NCDFReader  # type: ignore[import-untyped]
from MDAnalysis.coordinates.TRR import TRRReader  # type: ignore[import-untyped]


_TRR_MAGIC = 1993
# GROMACS writes TRR steps as signed 32-bit integers, which wrap.
_WRAP = 2**32


@dataclass(frozen=True)
class FrameKeys:
    """The steps and times of one file's frames, where the format records them.

    ``complete`` is the number of frames that are whole; a job killed while
    writing may leave the last frame of a TRR cut short. ``wraps`` is True
    for formats whose steps are 32-bit and wrap.
    """

    steps: np.ndarray | None
    times: np.ndarray | None
    complete: int
    wraps: bool = False


def frame_keys(reader) -> FrameKeys:
    """Read the steps and times of every frame of one MDAnalysis reader."""
    if isinstance(reader, TRRReader):
        return _trr_keys(reader.filename, reader._xdr.offsets)
    if isinstance(reader, H5MDReader):
        particles = reader._particle_group
        group = next(particles[name] for name in ("position", "force", "velocity") if name in particles)
        steps = np.asarray(group["step"][()], dtype=np.int64)
        times = np.asarray(group["time"][()], dtype=np.float64) if "time" in group else None
        return FrameKeys(steps, times, len(reader))
    if isinstance(reader, NCDFReader):
        times = (
            np.asarray(reader.trjfile.variables["time"][:], dtype=np.float64)
            if reader.has_time else None
        )
        return FrameKeys(None, times, len(reader))
    # Other formats: read every frame.
    read_steps: list[int | None] = []
    read_times: list[float] = []
    for ts in reader:
        read_steps.append(ts.data.get("step"))
        read_times.append(ts.time)
    reader.rewind()
    has_steps = all(step is not None for step in read_steps)
    return FrameKeys(
        np.array(read_steps, dtype=np.int64) if has_steps else None,
        np.array(read_times, dtype=np.float64),
        len(reader),
    )


def _trr_keys(path: str, offsets: np.ndarray) -> FrameKeys:
    """Steps and times from the header of each TRR frame, and whether the last is whole."""
    size = os.path.getsize(path)
    steps = np.empty(len(offsets), dtype=np.int64)
    times = np.empty(len(offsets), dtype=np.float64)
    complete = len(offsets)
    with open(path, "rb") as f:
        for k, offset in enumerate(offsets):
            steps[k], times[k], end = _trr_header(f, int(offset), path, k)
            if end > size:
                complete = k
    # A drop of more than half the 32-bit range is a wrap, not a step back.
    wrapped = np.concatenate([[0], np.cumsum(np.diff(steps) < -_WRAP // 2)])
    return FrameKeys(steps + wrapped * _WRAP, times, complete, wraps=True)


def _trr_header(f, offset: int, path: str, frame: int) -> tuple[int, float, int]:
    """The step, time and end offset of the TRR frame at ``offset``."""
    f.seek(offset)
    magic, _, length = struct.unpack(">3i", f.read(12))
    if magic != _TRR_MAGIC:
        raise ValueError(f"{path}, frame {frame}: not a TRR frame header at byte {offset}")
    f.seek(offset + 12 + (length + 3) // 4 * 4)
    sizes = struct.unpack(">13i", f.read(52))
    box, x, v, force, n_atoms, step = sizes[2], sizes[7], sizes[8], sizes[9], sizes[10], sizes[11]
    data = max(x, v, force)
    double = box == 9 * 8 if box else bool(n_atoms) and data == n_atoms * 3 * 8
    real = 8 if double else 4
    (time,) = struct.unpack(">d" if double else ">f", f.read(real))
    header = 12 + (length + 3) // 4 * 4 + 52 + 2 * real
    # The sizes are those of ir, e, box, vir, pres, top, sym, x, v and f.
    end = offset + header + sum(sizes[:10])
    return step, time, end


def order_keys(keys: list[FrameKeys], paths: list[str]) -> tuple[str, list[np.ndarray]]:
    """Choose steps, or times if any file records no steps, and return each file's keys.

    32-bit steps that wrap between files are unwrapped. Each file's keys
    cover its complete frames only.

    Raises
    ------
    ValueError
        If several files are given and one records neither steps nor times.
    """
    if all(k.steps is not None for k in keys):
        unit = "step"
        arrays = [k.steps[:k.complete] for k in keys]  # type: ignore[index]
        offset = 0
        for i in range(1, len(arrays)):
            if keys[i].wraps and keys[i - 1].wraps and len(arrays[i]) and len(arrays[i - 1]):
                if arrays[i][0] + offset < arrays[i - 1][-1] - _WRAP // 2:
                    offset += _WRAP
                arrays[i] = arrays[i] + offset
        return unit, arrays
    if len(keys) > 1:
        for k, path in zip(keys, paths):
            if k.times is None:
                raise ValueError(
                    f"{path}: the file records neither steps nor times, so it cannot "
                    "be joined with other files; load it as a separate trajectory"
                )
    arrays = [(k.times if k.times is not None else np.arange(k.complete))[:k.complete] for k in keys]
    return "time", arrays
