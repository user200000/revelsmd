"""
VASP trajectory backend for RevelsMD.

This module provides the VaspTrajectory class for reading VASP vasprun.xml files,
along with the Vasprun parser class.
"""

from typing import Any, Iterator

import numpy as np
from lxml import etree  # type: ignore[import-untyped]
from pymatgen.core import Structure, Lattice

from ._base import Trajectory


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

    Supports parsing one or multiple sequential ``vasprun.xml`` files, validating
    orthorhombicity, and providing cartesian coordinates and forces as NumPy arrays.

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
    box_x, box_y, box_z : float
        Simulation box lengths along each Cartesian axis.
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
        If no forces are found or if the lattice is non-orthorhombic.

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

            self._validate_cell(self._start_structure.lattice)
            diag = np.diag(self._start_structure.lattice.matrix)
            self.cell_matrix = self._cell_matrix_from_dimensions(*diag)
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

            self._validate_cell(self._start_structure.lattice)
            diag = np.diag(self._start_structure.lattice.matrix)
            self.cell_matrix = self._cell_matrix_from_dimensions(*diag)
            if self.Vasprun.forces is None:
                raise ValueError(f"No forces found in {trajectory_file}")
            self.positions = self.Vasprun.cart_coords
            self.forces = self.Vasprun.forces

        # Backwards compatibility: expose start structure via Vasprun.start
        self.Vasprun.start = self._start_structure  # type: ignore[attr-defined]

    @staticmethod
    def _validate_cell(lattice: Lattice):
        """Validate that the VASP lattice is orthorhombic."""
        Trajectory._validate_orthorhombic(list(lattice.angles))

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
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Iterate over in-memory position/force arrays."""
        for i in range(start, stop, stride):
            yield self.positions[i], self.forces[i]

    def get_frame(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Return positions and forces for a specific frame by index."""
        return self.positions[index], self.forces[index]
