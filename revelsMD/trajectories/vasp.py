"""VASP backend: VaspTrajectory and the helpers that index and read vasprun.xml."""

import array
import mmap
import os
from dataclasses import dataclass
from typing import BinaryIO

import numpy as np
from lxml import etree  # type: ignore[import-untyped]

from revelsMD.cell import fractional_to_cartesian
from ._base import (
    TrajectoryFiles,
    FileFrames, IndexedFileTrajectory, as_paths, check_fixed_cell, warn_cut_short,
)


# -----------------------------------------------------------------------------
# XML Parsing Helpers
# -----------------------------------------------------------------------------

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

    Step ``k`` is the ``k``-th ``<calculation>``, in bytes ``[starts[k],
    ends[k])``. ``basis`` is the first step's basis ``<varray>`` and
    ``lattice`` its matrix, the same for every step. ``cut`` is the number
    of an incomplete final step, which is not indexed. A file with no
    complete step has no atom names, basis or lattice.
    """

    path: str
    atom_names: list[str]
    basis: bytes
    lattice: np.ndarray
    starts: np.ndarray
    ends: np.ndarray
    cut: int | None = None

    @classmethod
    def without_steps(cls, path: str) -> "VasprunIndex":
        """The index of a file cut short before its first step was complete."""
        empty = np.empty(0, dtype=np.int64)
        return cls(path, [], b"", np.empty((0, 0)), empty, empty, cut=0)


def index_vasprun(path: str, reference: "VasprunIndex | None" = None) -> VasprunIndex:
    """Index the ionic steps of one vasprun.xml without parsing them.

    An unclosed final ``<calculation>`` is recorded as ``cut``, as is step 0
    of a file with no ``<calculation>``. Atoms and every step's lattice are
    compared with ``reference`` (the first file with a step) if given, else
    with the file's first step. Raises ValueError if the atom list is
    missing, a calculation other than the last is unclosed or incomplete,
    or the atoms or lattice differ.
    """
    with open(path, "rb") as f:
        if os.fstat(f.fileno()).st_size == 0:
            return VasprunIndex.without_steps(path)
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
            if m.find(_CALCULATION_OPEN) == -1:
                return VasprunIndex.without_steps(path)
            atom_names = _read_atom_names(m, path)
            if reference is not None and atom_names != reference.atom_names:
                raise ValueError(f"{path}: the atoms differ from those in {reference.path}")
            starts = array.array("q")
            ends = array.array("q")
            first_basis, first_lattice = b"", np.empty((0, 0))
            cut = None
            start = m.find(_CALCULATION_OPEN)
            while start != -1:
                k = len(starts)
                where = f"{path}, calculation {k}"
                close = m.find(_CALCULATION_CLOSE, start)
                following = m.find(_CALCULATION_OPEN, start + len(_CALCULATION_OPEN))
                if close == -1 and following == -1:
                    cut = k
                    break
                if close == -1 or -1 < following < close:
                    raise ValueError(
                        f"{where}: no closing </calculation> before the next ionic step"
                    )
                end = close + len(_CALCULATION_CLOSE)
                basis = _calculation_basis(m, where, start, end)
                if k == 0:
                    first_basis, first_lattice = basis, _parse_basis_at(basis, where)
                    reference_basis, reference_lattice = (
                        (reference.basis, reference.lattice) if reference is not None
                        else (first_basis, first_lattice)
                    )
                if basis != reference_basis:
                    check_fixed_cell(_parse_basis_at(basis, where), reference_lattice, where)
                starts.append(start)
                ends.append(end)
                start = following
    if not starts:
        return VasprunIndex.without_steps(path)
    return VasprunIndex(
        path, atom_names, first_basis, first_lattice,
        np.array(starts, dtype=np.int64), np.array(ends, dtype=np.int64), cut,
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
    """A float ``<varray>`` as an array with one row per ``<v>``; ragged rows raise ValueError."""
    return np.array([(v.text or "").split() for v in varray.findall("v")], dtype=np.float64)


def parse_basis(element: bytes) -> np.ndarray:
    """The 3x3 lattice basis (rows are lattice vectors) from a basis varray."""
    return _float_varray(etree.fromstring(element))


def read_vasp_frame(
    f: BinaryIO, start: int, end: int, n_atoms: int, cell: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Read the Cartesian positions and the forces of the ionic step in bytes [start, end).

    Positions come from the calculation's ``<structure>`` with no
    attributes, the one VASP writes for each ionic step. Raises ValueError
    if the XML is malformed, the basis, positions or forces are missing or
    not ``(n_atoms, 3)``, or the lattice differs from ``cell`` when given.
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
        raise ValueError(
            "the lattice of the step's own <structure> differs from the trajectory's "
            "cell; the calculation may hold another <structure> element that was "
            "indexed in its place, or the file may have changed since it was indexed"
        )
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
    """A trajectory read from VASP ``vasprun.xml`` files, in the ``'metal'`` unit system.

    The files are indexed on construction and each ionic step is read from
    disk as needed. Several files are concatenated in the order given;
    since vasprun.xml records no timesteps, their order and any repeated or
    missing steps cannot be checked.

    Parameters
    ----------
    trajectory_file : str, path, or sequence of these
        ``vasprun.xml`` file(s).
    temperature : float
        Temperature in Kelvin.

    Raises
    ------
    ValueError
        If a file is malformed or lacks an ionic step's structure or
        forces, the files list different atoms, the lattice changes or is
        invalid, or no file has a complete ionic step.

    Notes
    -----
    See :doc:`/tutorials/trajectory-formats` for how incomplete files are
    handled.
    """

    def __init__(
        self,
        trajectory_file: TrajectoryFiles,
        *,
        temperature: float,
    ):
        super().__init__(units='metal', temperature=temperature)

        self.trajectory_file = trajectory_file
        paths = as_paths(trajectory_file)

        # Later files are checked against the atoms and lattice of the first
        # file with a complete step, so a lattice error names the first ionic
        # step that differs.
        indices: list[VasprunIndex] = []
        first: VasprunIndex | None = None
        for path in paths:
            indices.append(index_vasprun(path, first))
            if first is None and len(indices[-1].starts):
                first = indices[-1]
        if first is None:
            raise ValueError(
                "no complete ionic steps in " + ", ".join(paths)
                + " (a run killed during its first step leaves none)"
            )
        for index in indices:
            if index.cut is not None:
                warn_cut_short(index.path, f"ionic step (calculation {index.cut})")
        self._atom_names = first.atom_names

        # Every ionic step of every file is kept, so a frame's position in
        # its file is its calculation number.
        self._frames = FileFrames.join(
            [index.path for index in indices],
            [index.starts for index in indices],
            [index.ends for index in indices],
        )
        self.frames = len(self._frames)
        self.cell_matrix = first.lattice.copy()
        self._validate_cell_matrix(self.cell_matrix, self._where(0))

    def get_indices(self, atype: str) -> np.ndarray:
        """Return the indices of the atoms of an element.

        Parameters
        ----------
        atype : str
            Element symbol (e.g. ``'O'``).

        Returns
        -------
        np.ndarray
            Row positions in the frames, whose atoms are in ``<atominfo>`` order.

        Raises
        ------
        ValueError
            If no atom is of that element.
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
        k, calculation = self._frames.locate(index)
        return f"{self._frames.paths[k]}, calculation {calculation}"

    def _read_arrays(self, f: BinaryIO, index: int) -> tuple[np.ndarray, np.ndarray]:
        frames = self._frames
        return read_vasp_frame(
            f, int(frames.starts[index]), int(frames.ends[index]), len(self._atom_names),
            self.cell_matrix,
        )
