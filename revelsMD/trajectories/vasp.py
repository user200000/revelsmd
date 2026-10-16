"""
VASP trajectory backend for RevelsMD.

This module provides the VaspTrajectory class for reading VASP vasprun.xml files,
and the helpers that index and read them.
"""

import array
import mmap
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import BinaryIO

import numpy as np
from lxml import etree  # type: ignore[import-untyped]

from revelsMD.cell import fractional_to_cartesian
from ._base import (
    FileFrames, IndexedFileTrajectory, as_paths, check_fixed_cell, warn_user,
)


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
    v_list = [(v.text or "").split() for v in varray.findall("v")]

    if varray_type == "int":
        return [[int(num) for num in v] for v in v_list]
    elif varray_type == "logical":
        return [[val == "T" for val in v] for v in v_list]
    else:
        return [[float(num) for num in v] for v in v_list]


_CALCULATION_OPEN = b"<calculation>"
_CALCULATION_CLOSE = b"</calculation>"
_STRUCTURE = b"<structure>"
_STRUCTURE_CLOSE = b"</structure>"
_POSITIONS = b'<varray name="positions"'
_FORCES = b'<varray name="forces"'
_BASIS = b'<varray name="basis"'
_VARRAY_CLOSE = b"</varray>"


@dataclass(frozen=True, eq=False)
class VasprunIndex:
    """The atoms and the complete ionic steps of one vasprun.xml.

    Ionic step ``k`` is the file's ``k``-th ``<calculation>`` element, which
    occupies bytes ``[starts[k], ends[k])``. ``basis`` is the first step's
    ``<varray name="basis">`` element and ``lattice`` its 3x3 matrix; every
    step has the same lattice.
    """

    path: str
    atom_names: list[str]
    basis: bytes
    lattice: np.ndarray
    starts: np.ndarray
    ends: np.ndarray


def index_vasprun(path: str, reference: "VasprunIndex | None" = None) -> VasprunIndex:
    """Index the ionic steps of one vasprun.xml without parsing them.

    A final ``<calculation>`` with no closing tag, as left by a run killed
    while writing, is dropped, with a warning if earlier steps are complete.

    Parameters
    ----------
    path : str
        The vasprun.xml file.
    reference : VasprunIndex, optional
        The index of the trajectory's first file, when this file is not the
        first. The atoms must be the same, and every step's lattice is
        compared with the first file's, so the first step that differs is
        named. By default steps are compared with the file's first step.

    Raises
    ------
    ValueError
        If the file is empty, holds more than one vasprun.xml document, or
        has no ``<atominfo>`` atom list or no complete ionic step; if a
        calculation other than the last is not closed; if a calculation
        lacks a structure, a lattice basis, positions or forces; if the
        atoms differ from ``reference``'s; or if the lattice changes
        between steps or differs from ``reference``'s.
    """
    with open(path, "rb") as f:
        if os.fstat(f.fileno()).st_size == 0:
            raise ValueError(f"{path}: the file is empty")
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
            atom_names = _read_atom_names(m, path)
            _check_single_document(m, path)
            if reference is not None and atom_names != reference.atom_names:
                raise ValueError(f"{path}: the atoms differ from those in {reference.path}")
            starts = array.array("q")
            ends = array.array("q")
            first_basis, first_lattice = b"", np.empty((0, 0))
            reference_basis, reference_lattice = (
                (reference.basis, reference.lattice) if reference is not None else (b"", None)
            )
            start = m.find(_CALCULATION_OPEN)
            while start != -1:
                k = len(starts)
                close = m.find(_CALCULATION_CLOSE, start)
                following = m.find(_CALCULATION_OPEN, start + len(_CALCULATION_OPEN))
                if close == -1 and following == -1:
                    if k:
                        warn_user(
                            f"{path}: dropping the incomplete final ionic step "
                            f"(calculation {k}); the file appears to have "
                            f"been cut short, so its {k} complete steps are used"
                        )
                    break
                if close == -1 or -1 < following < close:
                    raise ValueError(
                        f"{path}, calculation {k}: no closing "
                        "</calculation> before the next ionic step"
                    )
                end = close + len(_CALCULATION_CLOSE)
                where = f"{path}, calculation {k}"
                basis = _calculation_basis(m, where, start, end)
                if k == 0:
                    first_basis, first_lattice = basis, _parse_basis_at(basis, where)
                    if reference_lattice is None:
                        reference_basis, reference_lattice = first_basis, first_lattice
                if basis != reference_basis:
                    assert reference_lattice is not None
                    check_fixed_cell(_parse_basis_at(basis, where), reference_lattice, where)
                starts.append(start)
                ends.append(end)
                start = following
    if not starts:
        raise ValueError(
            f"{path}: no complete ionic steps (a run killed during its first "
            "step leaves none)"
        )
    return VasprunIndex(
        path, atom_names, first_basis, first_lattice,
        np.array(starts, dtype=np.int64), np.array(ends, dtype=np.int64),
    )


def _calculation_basis(m: mmap.mmap, where: str, start: int, end: int) -> bytes:
    """Check one calculation's contents and return its lattice basis element."""
    structure = m.find(_STRUCTURE, start, end)
    missing = []
    if structure == -1:
        missing.append("structure")
    if m.find(_FORCES, start, end) == -1:
        missing.append("forces")
    if missing:
        raise ValueError(
            f"{where}: no {' and no '.join(missing)}; every "
            "ionic step needs a structure and forces"
        )
    structure_end = m.find(_STRUCTURE_CLOSE, structure, end)
    if structure_end == -1:
        structure_end = end
    if m.find(_POSITIONS, structure, structure_end) == -1:
        raise ValueError(f"{where}: the structure has no positions")
    basis_start = m.find(_BASIS, structure, structure_end)
    basis_end = m.find(_VARRAY_CLOSE, basis_start, end) if basis_start != -1 else -1
    if basis_end == -1:
        raise ValueError(f"{where}: the structure has no lattice basis")
    return bytes(m[basis_start:basis_end + len(_VARRAY_CLOSE)])


def _parse_basis_at(element: bytes, where: str) -> np.ndarray:
    """parse_basis, naming the file and calculation if the basis is malformed."""
    try:
        return parse_basis(element)
    except (ValueError, etree.XMLSyntaxError) as e:
        raise ValueError(f"{where}: malformed lattice basis: {e}") from e


def _check_single_document(m: mmap.mmap, path: str) -> None:
    """Raise if the file holds more than one vasprun.xml, as ``cat a b > c`` makes.

    A second XML declaration or ``<atominfo>`` marks a second document.
    """
    for marker in (b"<?xml", b"<atominfo>"):
        first = m.find(marker)
        second = m.find(marker, first + 1) if first != -1 else -1
        if second != -1:
            raise ValueError(
                f"{path}: a second {marker.decode()} at byte {second}; the file "
                "holds more than one vasprun.xml, so pass them as separate files"
            )


def _read_atom_names(m: mmap.mmap, path: str) -> list[str]:
    """Element names of the atoms, from the ``<atominfo>`` section."""
    start = m.find(b"<atominfo>")
    end = m.find(b"</atominfo>", start) if start != -1 else -1
    if end == -1:
        raise ValueError(f"{path}: no <atominfo> section")
    try:
        atominfo = etree.fromstring(bytes(m[start:end + len(b"</atominfo>")]))
        for element in atominfo.findall("array"):
            if element.get("name") == "atoms":
                names = [rc.find("c").text.strip() for rc in element.find("set")]
                if names:
                    return names
    except (etree.XMLSyntaxError, AttributeError, TypeError) as e:
        raise ValueError(f"{path}: malformed <atominfo>: {e}") from e
    raise ValueError(f"{path}: no atom list in <atominfo>")


def _float_varray(varray: etree.Element) -> np.ndarray:
    """A float ``<varray>`` element as an array with one row per ``<v>``."""
    return np.array(parse_varray(varray), dtype=np.float64)


def parse_basis(element: bytes) -> np.ndarray:
    """The 3x3 lattice basis (rows are lattice vectors) from a basis varray."""
    return _float_varray(etree.fromstring(element))


def read_vasp_frame(
    f: BinaryIO, start: int, end: int, n_atoms: int, cell: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Read the Cartesian positions and the forces of the ionic step in bytes [start, end).

    The positions are those of the calculation's ``<structure>`` element
    with no attributes, which is the one VASP writes for each ionic step.

    Parameters
    ----------
    f : binary file object
        The vasprun.xml, opened in binary mode.
    start, end : int
        The byte range of the ``<calculation>`` element.
    n_atoms : int
        The number of atoms.
    cell : np.ndarray, optional
        The trajectory's cell matrix. If given, the step's lattice must
        equal it exactly.

    Raises
    ------
    ValueError
        If the byte range is not well-formed XML, if the step's basis,
        positions or forces are missing or malformed, if they do not have
        three values for each atom, or if the lattice differs from ``cell``.
    """
    f.seek(start)
    try:
        calculation = etree.fromstring(f.read(end - start))
    except etree.XMLSyntaxError as e:
        raise ValueError(f"malformed XML: {e}") from e
    structure = next((e for e in calculation.findall("structure") if not e.attrib), None)
    basis = fractional = None
    if structure is not None:
        basis = structure.find("crystal/varray[@name='basis']")
        fractional = structure.find("varray[@name='positions']")
    forces = calculation.find("varray[@name='forces']")
    if basis is None or fractional is None or forces is None:
        raise ValueError("the ionic step lacks a basis, positions or forces")
    lattice = _float_varray(basis)
    if cell is not None and not np.array_equal(lattice, cell):
        raise ValueError("the lattice differs from the first ionic step's")
    positions = fractional_to_cartesian(_float_varray(fractional), lattice)
    force_array = _float_varray(forces)
    if positions.shape != (n_atoms, 3) or force_array.shape != (n_atoms, 3):
        raise ValueError(
            f"expected positions and forces for {n_atoms} atoms, got arrays of "
            f"shape {positions.shape} and {force_array.shape}"
        )
    return positions, force_array


# -----------------------------------------------------------------------------
# VASP Trajectory Class
# -----------------------------------------------------------------------------

class VaspTrajectory(IndexedFileTrajectory):
    """
    Represents a molecular dynamics trajectory obtained from VASP ``vasprun.xml`` output.

    The files are indexed when the trajectory is constructed and each ionic
    step is read from disk as it is needed; the index keeps a few tens of
    bytes per step.

    Parameters
    ----------
    trajectory_file : str, path, or sequence of these
        ``vasprun.xml`` file(s) from consecutive runs, joined in the order
        given. vasprun.xml records no timesteps, so the order, and repeated
        or missing steps between files, cannot be checked.
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
        Inverse thermal energy 1/(kB*T) in 1/eV.

    Raises
    ------
    ValueError
        If a file is empty, holds more than one vasprun.xml document, or
        has no atom list or no complete ionic step; if a calculation other
        than the last in a file is not closed; if an ionic step lacks a
        structure, a lattice basis, positions or forces; if the files list
        different atoms; if the lattice changes between steps; if the same
        file is given twice; or if the cell is invalid.
    TypeError
        If ``trajectory_file`` is an unordered collection such as a set.

    Notes
    -----
    An incomplete final ionic step, as left by a run killed while writing,
    is dropped with a warning; a file with no complete step raises.
    Reading a step raises ValueError if its
    positions or forces are malformed, do not match the atom count or
    contain NaN or infinity, or if its lattice differs from the first
    step's.
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

        # Later files are checked against the first file's atoms and lattice,
        # so a lattice error names the first ionic step that differs.
        first = index_vasprun(paths[0])
        indices = [first] + [index_vasprun(path, first) for path in paths[1:]]
        self._atom_names = first.atom_names

        self.cell_matrix = first.lattice.copy()
        self._validate_cell_matrix(self.cell_matrix, f"{first.path}, calculation 0")

        # Frame i is ionic step _calculation[i] of its file.
        self._frames = FileFrames.join(
            [index.path for index in indices],
            [index.starts for index in indices],
            [index.ends for index in indices],
        )
        self._calculation = np.concatenate(
            [np.arange(len(index.starts), dtype=np.int32) for index in indices]
        )
        self.frames = len(self._frames)

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

        Raises
        ------
        ValueError
            If no atom is of that species.
        """
        indices = np.flatnonzero(np.asarray(self._atom_names) == atype)
        if len(indices) == 0:
            species = ", ".join(sorted(set(self._atom_names)))
            raise ValueError(f"no atoms of species '{atype}'; the species are {species}")
        return indices

    # get_charges and get_masses are inherited from Trajectory and raise
    # DataUnavailableError: charges are not in vasprun.xml, and masses
    # (in <atominfo>) are not read.

    def _where(self, index: int) -> str:
        """Name the file and calculation of frame ``index``."""
        return f"{self._frames.path(index)}, calculation {int(self._calculation[index])}"

    def _read_arrays(self, f: BinaryIO, index: int) -> tuple[np.ndarray, np.ndarray]:
        frames = self._frames
        return read_vasp_frame(
            f, int(frames.starts[index]), int(frames.ends[index]), len(self._atom_names),
            self.cell_matrix,
        )
