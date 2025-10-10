"""
LAMMPS trajectory parser utilities for RevelsMD.

Provides helper functions to read, interpret, and step through
LAMMPS custom dump files. The parser assumes each frame has a
consistent header structure.

Notes
-----
- Supports ASCII text LAMMPS dump format (``ITEM:`` headers).
- Box dimensions are assumed orthorhombic (``pp pp pp`` periodicity).
- The parser reads one or multiple frames sequentially for analysis.
"""

import numpy as np


# -----------------------------------------------------------------------------
# Header Parsing
# -----------------------------------------------------------------------------
def first_read(dumpFile: str):
    """
    Perform an initial read of a LAMMPS custom dump file to extract header metadata.

    Parameters
    ----------
    dumpFile : str
        Path to the LAMMPS dump file.

    Returns
    -------
    frames : int
        Number of trajectory frames.
    num_ats : int
        Number of atoms per frame.
    dic : list of str
        Column headers extracted from the ``ITEM: ATOMS`` line.
    header_length : int
        Number of header lines per frame (before atom coordinates start).
    dimgrid : numpy.ndarray of shape (3, 2)
        Box boundaries (x, y, z) for the first frame.

    Notes
    -----
    This function assumes:
    - Box bounds are printed as ``ITEM: BOX BOUNDS pp pp pp``.
    - The number of atoms and header structure remain constant across all frames.
    - If the total line count does not divide evenly into frames, a warning is printed.
    """
    header_length = 0
    dimgrid = np.zeros((3, 2))
    closer = 0
    num_ats = 0

    with open(dumpFile, "r") as f:
        while closer == 0:
            currentString = f.readline()
            if not currentString:
                raise ValueError("Unexpected EOF while parsing header.")
            if currentString.startswith("ITEM: ATOMS"):
                dic = currentString.split()
                closer = 1
            if currentString.strip() == "ITEM: NUMBER OF ATOMS":
                header_length += 1
                currentString = f.readline()
                num_ats = int(currentString)
            header_length += 1
            if currentString.strip().startswith("ITEM: BOX BOUNDS"):
                header_length += 3
                dimgrid[0, :] = np.array(f.readline().split(), dtype=float)
                dimgrid[1, :] = np.array(f.readline().split(), dtype=float)
                dimgrid[2, :] = np.array(f.readline().split(), dtype=float)

    numLines = sum(1 for _ in open(dumpFile))
    frames = numLines / float(num_ats + header_length)
    if frames % 1 != 0:
        print("WARNING: Non-integer frame count — incomplete file or inconsistent headers.")
    return int(frames), num_ats, dic, header_length, dimgrid


# -----------------------------------------------------------------------------
# Frame Extraction
# -----------------------------------------------------------------------------
def get_a_frame(f, num_ats: int, header_length: int, strngdex: list[int]) -> np.ndarray:
    """
    Extract a single frame of atomic data from an open LAMMPS dump file.

    Parameters
    ----------
    f : file object
        Open trajectory file (text mode).
    num_ats : int
        Number of atoms per frame.
    header_length : int
        Number of header lines preceding atomic data.
    strngdex : list of int
        Column indices to extract (relative to the start of atom data).

    Returns
    -------
    numpy.ndarray
        Array of shape ``(num_ats, len(strngdex))`` containing the requested
        columns (e.g. coordinates, forces, velocities).
    """
    vars_trest = np.zeros((num_ats, len(strngdex)))
    for _ in range(header_length):
        f.readline()
    for i in range(num_ats):
        currentString = f.readline().split()
        for j, k in enumerate(strngdex):
            vars_trest[i, j] = float(currentString[k])
    return vars_trest


# -----------------------------------------------------------------------------
# Column Index Mapping
# -----------------------------------------------------------------------------
def define_strngdex(our_string: list[str], dic: list[str]) -> list[int]:
    """
    Map requested LAMMPS quantities to their corresponding column indices.

    Parameters
    ----------
    our_string : list of str
        Names of quantities to extract (e.g. ``['x', 'y', 'z', 'fx', 'fy', 'fz']``).
    dic : list of str
        Column header list from the LAMMPS dump (``ITEM: ATOMS ...`` line).

    Returns
    -------
    list of int
        Column indices corresponding to the requested quantities.

    Examples
    --------
    >>> dic = ['ITEM:', 'ATOMS', 'id', 'type', 'x', 'y', 'z', 'fx', 'fy', 'fz']
    >>> define_strngdex(['x', 'z'], dic)
    [3, 5]
    """
    return [int(dic.index(ele) - 2) for ele in our_string]


# -----------------------------------------------------------------------------
# Frame Skipping
# -----------------------------------------------------------------------------
def frame_skip(f, num_ats: int, num_skip: int, header_length: int):
    """
    Skip a specified number of frames in an open trajectory file.

    Parameters
    ----------
    f : file object
        Open trajectory file in text mode.
    num_ats : int
        Number of atoms per frame.
    num_skip : int
        Number of frames to skip.
    header_length : int
        Number of header lines per frame.

    Notes
    -----
    This function simply advances the file pointer by reading and discarding
    the appropriate number of lines, based on the known frame structure.
    """
    for _ in range(num_skip * (num_ats + header_length)):
        f.readline()

