"""
VASP trajectory backend for RevelsMD.

This module provides the VaspTrajectory class for reading VASP vasprun.xml files,
along with the Vasprun parser class.
"""

from __future__ import annotations

import mmap
import os
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, BinaryIO, Iterator

import numpy as np
from lxml import etree  # type: ignore[import-untyped]

from revelsMD.frame_sources import Frame
from ._base import Trajectory

if TYPE_CHECKING:
    from pymatgen.core import Structure


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


def parse_structure(structure: etree.Element) -> dict[str, Any]:
    """
    Parse a ``<structure>`` XML element into dictionary form.

    Parameters
    ----------
    structure : etree.Element
        The ``<structure>`` XML node from ``vasprun.xml``.

    Returns
    -------
    dict
        Dictionary with the following keys:

        ``lattice`` : list of list of float
            3x3 lattice matrix.

        ``frac_coords`` : list of list of float
            Fractional coordinates.

        ``selective_dynamics`` : list of list of bool or None
            Selective dynamics flags, if present.
    """
    crystal = structure.find("crystal")
    if crystal is None or crystal.find("varray") is None:
        raise ValueError("No lattice data found in structure node.")

    latt = parse_varray(crystal.find("varray"))

    pos = parse_varray(structure.find("varray"))
    sdyn_elem = structure.find("varray[@name='selective']")
    sdyn = parse_varray(sdyn_elem) if sdyn_elem is not None else None

    return {
        "lattice": latt,
        "frac_coords": pos,
        "selective_dynamics": sdyn,
    }


def structure_from_structure_data(
    lattice: list[list[float]],
    atom_names: list[str],
    frac_coords: list[list[float]],
) -> Structure:
    """
    Create a :class:`pymatgen.core.Structure` from parsed structure data.

    Parameters
    ----------
    lattice : list of list of float
        3x3 lattice matrix.
    atom_names : list of str
        Atom species names.
    frac_coords : list of list of float
        Fractional atomic coordinates.

    Returns
    -------
    pymatgen.core.Structure
        Constructed :class:`Structure` object.
    """
    try:
        from pymatgen.core import Structure
    except ImportError as exc:
        raise ImportError(
            "pymatgen is required for VASP trajectory parsing. "
            "Install it with: pip install revelsMD[vasp]"
        ) from exc

    return Structure(lattice=lattice, species=atom_names, coords=frac_coords, coords_are_cartesian=False)


# -----------------------------------------------------------------------------
# Vasprun Parser Class
# -----------------------------------------------------------------------------

class Vasprun:
    """
    Lightweight parser for ``vasprun.xml`` files.

    Parameters
    ----------
    filename : str
        Path to the ``vasprun.xml`` file.

    Attributes
    ----------
    atom_names : list of str
        Atom name strings.
    structures : list of pymatgen.core.Structure
        Structures parsed from each ``<calculation>`` node.
    frac_coords : numpy.ndarray
        Fractional coordinates of shape ``(frames, atoms, 3)``.
    cart_coords : numpy.ndarray
        Cartesian coordinates of shape ``(frames, atoms, 3)``.
    forces : numpy.ndarray or None
        Cartesian forces of shape ``(frames, atoms, 3)``, if present.

    Raises
    ------
    ValueError
        If required XML sections are missing or malformed.
    """

    def __init__(self, filename: str) -> None:
        self.doc = etree.parse(filename).getroot()
        self._atom_names: list[str] | None = None
        self._structures: list[Structure] | None = None

    # -------------------------------------------------------------------------
    # Cached properties
    # -------------------------------------------------------------------------
    @property
    def atom_names(self) -> list[str]:
        """List of atomic species parsed from the ``<atominfo>`` section."""
        if self._atom_names is None:
            self._atom_names = self._parse_atom_names()
        return self._atom_names

    @property
    def structures(self) -> list[Structure]:
        """List of :class:`pymatgen.core.Structure` objects, parsed and cached."""
        if self._structures is None:
            self._structures = self._parse_structures()
        return self._structures

    # -------------------------------------------------------------------------
    # XML Parsing Internals
    # -------------------------------------------------------------------------
    def _parse_atom_names(self) -> list[str]:
        """Extract atom names from the ``<atominfo>`` block."""
        atominfo = self.doc.find("atominfo")
        if atominfo is None:
            raise ValueError("Missing <atominfo> in vasprun.xml")

        for array in atominfo.findall("array"):
            if array.get("name") == "atoms":
                names = [c.find("c").text.strip() for c in array.find("set")]
                if not names:
                    raise ValueError("Empty atom list in <atominfo>")
                return names
        raise ValueError("No atom array named 'atoms' in <atominfo>")

    def _parse_structures(self) -> list[Structure]:
        """Extract all ``<structure>`` elements and convert to :class:`Structure` objects."""
        structures = []
        for calc in self.doc.iterfind("calculation"):
            elem = calc.find("structure")
            if elem is None:
                continue
            sdata = parse_structure(elem)
            structures.append(
                structure_from_structure_data(
                    lattice=sdata["lattice"],
                    atom_names=self.atom_names,
                    frac_coords=sdata["frac_coords"],
                )
            )
        if not structures:
            raise ValueError("No structures found in vasprun.xml")
        return structures

    # -------------------------------------------------------------------------
    # Coordinate + Force Extraction
    # -------------------------------------------------------------------------
    @property
    def frac_coords(self) -> np.ndarray:
        """Fractional coordinates for each structure."""
        return np.array([s.frac_coords for s in self.structures])

    @property
    def cart_coords(self) -> np.ndarray:
        """Cartesian coordinates for each structure."""
        return np.array([s.cart_coords for s in self.structures])

    @property
    def forces(self) -> np.ndarray | None:
        """
        Forces extracted from ``<varray name="forces">`` if present.

        Returns
        -------
        numpy.ndarray or None
            Force array of shape ``(frames, atoms, 3)``, or ``None`` if absent.
        """
        all_forces = []
        for calc in self.doc.iterfind("calculation"):
            elem = calc.find("varray[@name='forces']")
            if elem is not None:
                all_forces.append(parse_varray(elem))

        if not all_forces:
            raise ValueError("No forces found in vasprun.xml")
        return np.array(all_forces)


# -----------------------------------------------------------------------------
# VASP Trajectory Class
# -----------------------------------------------------------------------------

class VaspTrajectory(Trajectory):
    """
    Represents a molecular dynamics trajectory obtained from VASP ``vasprun.xml`` output.

    Supports parsing one or multiple sequential ``vasprun.xml`` files and
    providing cartesian coordinates and forces as NumPy arrays.

    Parameters
    ----------
    trajectory_file : str or list of str
        Path or list of paths to ``vasprun.xml`` file(s).
    temperature : float
        Simulation temperature in Kelvin.

    Attributes
    ----------
    frames : int
        Total number of time steps across all vasprun files.
    cell_matrix : np.ndarray
        Cell matrix with rows = lattice vectors, shape ``(3, 3)``.
    positions : np.ndarray
        Cartesian coordinates array of shape ``(frames, atoms, 3)``.
    forces : np.ndarray
        Cartesian forces array of shape ``(frames, atoms, 3)``.
    temperature : float
        Simulation temperature in Kelvin.
    beta : float
        Inverse thermal energy 1/(kB*T) in eV.

    Raises
    ------
    ValueError
        If no forces are found or the cell matrix is invalid.

    Notes
    -----
    - The parser enforces presence of forces for physical completeness.
    - For NVT/NVE MD, ensure `IBRION=-1` and `NSW > 0` during VASP runs.
    """

    def __init__(self, trajectory_file: str | list[str], *, temperature: float):
        super().__init__(units='metal', temperature=temperature)

        self.trajectory_file: str | list[str] = trajectory_file

        if isinstance(trajectory_file, list):
            self.Vasprun = Vasprun(trajectory_file[0])
            self._start_structure = self.Vasprun.structures[0]
            self.frames = len(self.Vasprun.structures)

            self.cell_matrix = np.array(self._start_structure.lattice.matrix, dtype=np.float64)
            self._validate_cell_matrix(self.cell_matrix)
            if self.Vasprun.forces is None:
                raise ValueError(f"No forces found in {trajectory_file[0]}")
            self.positions: np.ndarray = self.Vasprun.cart_coords
            self.forces: np.ndarray = self.Vasprun.forces

            for item in trajectory_file[1:]:
                next_run = Vasprun(item)
                if next_run.forces is None:
                    raise ValueError(f"No forces found in {item}")
                self.frames += len(next_run.structures)
                self.positions = np.append(self.positions, next_run.cart_coords, axis=0)
                self.forces = np.append(self.forces, next_run.forces, axis=0)

        else:
            self.Vasprun = Vasprun(trajectory_file)
            self._start_structure = self.Vasprun.structures[0]
            self.frames = len(self.Vasprun.structures)

            self.cell_matrix = np.array(self._start_structure.lattice.matrix, dtype=np.float64)
            self._validate_cell_matrix(self.cell_matrix)
            if self.Vasprun.forces is None:
                raise ValueError(f"No forces found in {trajectory_file}")
            self.positions = self.Vasprun.cart_coords
            self.forces = self.Vasprun.forces

        # Backwards compatibility: expose start structure via Vasprun.start
        self.Vasprun.start = self._start_structure  # type: ignore[attr-defined]

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
        return np.array(self._start_structure.indices_from_symbol(atype))

    # get_charges and get_masses are inherited from Trajectory
    # and raise DataUnavailableError since VASP doesn't provide this data

    def _iter_frames_impl(
        self,
        start: int,
        stop: int,
        stride: int
    ) -> Iterator[Frame]:
        """Iterate over in-memory position/force arrays."""
        for i in range(start, stop, stride):
            yield Frame(self.positions[i], self.forces[i])

    def get_frame(self, index: int) -> Frame:
        """Return positions and forces for a specific frame by index.

        Parameters
        ----------
        index : int
            Frame index to retrieve.

        Returns
        -------
        Frame
            A dataclass with ``positions`` and ``forces`` arrays,
            each of shape ``(n_atoms, 3)``.
        """
        return Frame(self.positions[index], self.forces[index])
