"""
VASP trajectory backend for RevelsMD.

This module provides the VaspTrajectory class for reading VASP vasprun.xml files,
and the helpers that index and read them.
"""

import mmap
import os
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import BinaryIO, Iterator

import numpy as np
from lxml import etree  # type: ignore[import-untyped]

from revelsMD.frame_sources import Frame
from ._base import Trajectory, as_paths, check_fixed_cell, read_frames_one_file_at_a_time


# -----------------------------------------------------------------------------
# XML Parsing Helpers
# -----------------------------------------------------------------------------

def parse_varray(varray: etree.Element) -> list[list[float]] | list[list[int]] | list[list[bool]]:
    """
    Parse a ``<varray>`` XML element from ``vasprun.xml``.

    Parameters
    ----------
    varray : etree.Element
        The ``<varray>`` XML element containing numerical data.

    Returns
    -------
    list of list of float or int or bool
        Nested list of parsed values, with types determined by the ``type``
        attribute of the ``<varray>`` tag (``float``, ``int``, or ``logical``).
    """
    varray_type = varray.get("type", None)
    v_list = [v.text.split() for v in varray.findall("v")]

    if varray_type == "int":
        return [[int(num) for num in v] for v in v_list]
    elif varray_type == "logical":
        return [[val == "T" for val in v] for v in v_list]
    else:
        return [[float(num) for num in v] for v in v_list]


_CALCULATION_OPEN = b"<calculation>"
_CALCULATION_CLOSE = b"</calculation>"
_STRUCTURE = b"<structure>"
_FORCES = b'<varray name="forces"'
_BASIS = b'<varray name="basis"'
_VARRAY_CLOSE = b"</varray>"


@dataclass(frozen=True)
class VaspFrame:
    """Where one ionic step lies in its vasprun.xml."""

    path: str
    calculation: int
    start: int
    end: int
    basis: bytes


@dataclass(frozen=True)
class VasprunIndex:
    """The atoms and the complete ionic steps of one vasprun.xml."""

    path: str
    atom_names: list[str]
    frames: list[VaspFrame]


def index_vasprun(path: str) -> VasprunIndex:
    """Index the ionic steps of one vasprun.xml without parsing them.

    A final ``<calculation>`` with no closing tag, as left by a run killed
    while writing, is dropped with a warning.

    Raises
    ------
    ValueError
        If the file has no ``<atominfo>`` or no complete ionic step, if a
        calculation other than the last is not closed, or if a calculation
        lacks a structure, a lattice basis or forces.
    """
    with open(path, "rb") as f:
        if os.fstat(f.fileno()).st_size == 0:
            raise ValueError(f"{path}: the file is empty")
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
            atom_names = _read_atom_names(m, path)
            frames: list[VaspFrame] = []
            start = m.find(_CALCULATION_OPEN)
            while start != -1:
                close = m.find(_CALCULATION_CLOSE, start)
                following = m.find(_CALCULATION_OPEN, start + len(_CALCULATION_OPEN))
                if close == -1 or (following != -1 and following < close):
                    if following == -1:
                        warnings.warn(
                            f"{path}: dropping the incomplete final ionic step "
                            f"(calculation {len(frames)}); the file appears to have "
                            f"been cut short, so its {len(frames)} complete steps are used",
                            UserWarning,
                            stacklevel=2,
                        )
                        break
                    raise ValueError(
                        f"{path}, calculation {len(frames)}: no closing "
                        "</calculation> before the next ionic step"
                    )
                end = close + len(_CALCULATION_CLOSE)
                frames.append(_index_calculation(m, path, len(frames), start, end))
                start = following
    if not frames:
        raise ValueError(f"{path}: no complete ionic steps")
    return VasprunIndex(path, atom_names, frames)


def _index_calculation(m: mmap.mmap, path: str, k: int, start: int, end: int) -> VaspFrame:
    """Check one calculation's contents and record where it lies."""
    structure = m.find(_STRUCTURE, start, end)
    missing = []
    if structure == -1:
        missing.append("structure")
    if m.find(_FORCES, start, end) == -1:
        missing.append("forces")
    if missing:
        raise ValueError(
            f"{path}, calculation {k}: no {' and no '.join(missing)}; every "
            "ionic step needs a structure and forces"
        )
    basis_start = m.find(_BASIS, structure, end)
    basis_end = m.find(_VARRAY_CLOSE, basis_start, end) if basis_start != -1 else -1
    if basis_end == -1:
        raise ValueError(f"{path}, calculation {k}: the structure has no lattice basis")
    return VaspFrame(path, k, start, end, bytes(m[basis_start:basis_end + len(_VARRAY_CLOSE)]))


def _read_atom_names(m: mmap.mmap, path: str) -> list[str]:
    """Element names of the atoms, from the ``<atominfo>`` section."""
    start = m.find(b"<atominfo>")
    end = m.find(b"</atominfo>", start) if start != -1 else -1
    if end == -1:
        raise ValueError(f"{path}: no <atominfo> section")
    atominfo = etree.fromstring(bytes(m[start:end + len(b"</atominfo>")]))
    for array in atominfo.findall("array"):
        if array.get("name") == "atoms":
            names = [rc.find("c").text.strip() for rc in array.find("set")]
            if names:
                return names
    raise ValueError(f"{path}: no atom list in <atominfo>")


def parse_basis(element: bytes) -> np.ndarray:
    """The 3x3 lattice basis (rows are lattice vectors) from a basis varray."""
    return np.array(parse_varray(etree.fromstring(element)), dtype=np.float64)


def read_vasp_frame(f: BinaryIO, frame: VaspFrame, n_atoms: int) -> tuple[np.ndarray, np.ndarray]:
    """Read the Cartesian positions and the forces of one indexed ionic step.

    Raises
    ------
    ValueError
        If the step's basis, positions or forces are missing or do not have
        one row per atom.
    """
    f.seek(frame.start)
    calculation = etree.fromstring(f.read(frame.end - frame.start))
    basis = calculation.find("structure/crystal/varray[@name='basis']")
    fractional = calculation.find("structure/varray[@name='positions']")
    forces = calculation.find("varray[@name='forces']")
    if basis is None or fractional is None or forces is None:
        raise ValueError("the ionic step lacks a basis, positions or forces")
    lattice = np.array(parse_varray(basis), dtype=np.float64)
    positions = np.array(parse_varray(fractional), dtype=np.float64) @ lattice
    force_array = np.array(parse_varray(forces), dtype=np.float64)
    if positions.shape != (n_atoms, 3) or force_array.shape != (n_atoms, 3):
        raise ValueError(
            f"expected positions and forces for {n_atoms} atoms, got arrays of "
            f"shape {positions.shape} and {force_array.shape}"
        )
    return positions, force_array




# -----------------------------------------------------------------------------
# VASP Trajectory Class
# -----------------------------------------------------------------------------

class VaspTrajectory(Trajectory):
    """
    Represents a molecular dynamics trajectory obtained from VASP ``vasprun.xml`` output.

    The files are indexed when the trajectory is constructed and each ionic
    step is read from disk as it is needed, so memory use does not grow
    with the length of the run.

    Parameters
    ----------
    trajectory_file : str, path, or sequence of these
        ``vasprun.xml`` file(s) from consecutive runs, joined in the order
        given.
    temperature : float
        Simulation temperature in Kelvin.

    Attributes
    ----------
    frames : int
        Number of complete ionic steps across all files.
    cell_matrix : np.ndarray
        Cell matrix with rows = lattice vectors, shape ``(3, 3)``.
    temperature : float
        Simulation temperature in Kelvin.
    beta : float
        Inverse thermal energy 1/(kB*T) in eV.

    Raises
    ------
    ValueError
        If a file has no atom list or no complete ionic step, if an ionic
        step lacks a structure or forces, if the files list different atoms,
        or if the lattice changes between steps.

    Notes
    -----
    An incomplete final ionic step, as left by a run killed while writing,
    is dropped with a warning.
    """

    def __init__(
        self,
        trajectory_file: "str | os.PathLike[str] | Sequence[str | os.PathLike[str]]",
        *,
        temperature: float,
    ):
        super().__init__(units='metal', temperature=temperature)

        self.trajectory_file = trajectory_file
        paths = as_paths(trajectory_file)
        if not paths:
            raise ValueError("At least one vasprun.xml file is required.")

        indices = [index_vasprun(path) for path in paths]
        self._atom_names = indices[0].atom_names
        for index in indices[1:]:
            if index.atom_names != self._atom_names:
                raise ValueError(
                    f"{index.path}: the atoms differ from those in {indices[0].path}"
                )
        self._index = [frame for index in indices for frame in index.frames]

        first = self._index[0]
        self.cell_matrix = parse_basis(first.basis)
        self._validate_cell_matrix(self.cell_matrix)
        for frame in self._index[1:]:
            if frame.basis != first.basis:
                check_fixed_cell(
                    parse_basis(frame.basis), self.cell_matrix,
                    f"{frame.path}, calculation {frame.calculation}",
                )
        self.frames = len(self._index)

    def get_indices(self, atype: str) -> np.ndarray:
        """
        Return indices of atoms corresponding to a given species.

        Parameters
        ----------
        atype : str
            Element symbol (e.g., `'O'`, `'Si'`, `'Fe'`).

        Returns
        -------
        np.ndarray
            Array of atom indices matching the requested species.
        """
        return np.flatnonzero(np.asarray(self._atom_names) == atype)

    # get_charges and get_masses are inherited from Trajectory
    # and raise DataUnavailableError since VASP doesn't provide this data

    def _read(self, f: BinaryIO, index: int) -> Frame:
        """Read frame ``index`` from its open vasprun.xml."""
        frame = self._index[index]
        try:
            positions, forces = read_vasp_frame(f, frame, len(self._atom_names))
        except (ValueError, etree.XMLSyntaxError) as e:
            raise ValueError(f"{frame.path}, calculation {frame.calculation}: {e}") from e
        return Frame(positions, forces)

    def _iter_frames_impl(
        self,
        start: int,
        stop: int,
        stride: int
    ) -> Iterator[Frame]:
        """Read the indexed ionic steps in order, with one file open at a time."""
        return read_frames_one_file_at_a_time(
            range(start, stop, stride), lambda i: self._index[i].path, self._read
        )

    def get_frame(self, index: int) -> Frame:
        """Return positions and forces for a specific frame by index.

        The frame is read from disk by seeking to it; nothing else is read
        or kept in memory.

        Parameters
        ----------
        index : int
            Frame index to retrieve. Negative indices count from the end.

        Returns
        -------
        Frame
            A dataclass with ``positions`` and ``forces`` arrays,
            each of shape ``(n_atoms, 3)``.
        """
        with open(self._index[index].path, "rb") as f:
            return self._read(f, index)
