"""
VASP XML trajectory parser for RevelsMD.

This module provides lightweight parsing of ``vasprun.xml`` files to extract
structures, coordinates, and forces. It is designed for compatibility with
RevelsMD trajectory states and :class:`pymatgen.core.Structure` objects.

Notes
-----
- Only orthorhombic (or cubic) cells are fully supported downstream.
- The parser reads structures and, if present, forces from each ``<calculation>`` tag.
"""

from lxml import etree  # type: ignore
from typing import List, Union, Optional, Any, Dict
from pymatgen.core import Structure
import numpy as np


# -----------------------------------------------------------------------------
# XML Parsing Helpers
# -----------------------------------------------------------------------------
def parse_varray(varray: etree.Element) -> Union[List[List[float]], List[List[int]], List[List[bool]]]:
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


def parse_structure(structure: etree.Element) -> Dict[str, Any]:
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
            3×3 lattice matrix.

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
    lattice: List[List[float]],
    atom_names: List[str],
    frac_coords: List[List[float]],
) -> Structure:
    """
    Create a :class:`pymatgen.core.Structure` from parsed structure data.

    Parameters
    ----------
    lattice : list of list of float
        3×3 lattice matrix.
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
# Main Parser Class
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
        self._atom_names: Optional[List[str]] = None
        self._structures: Optional[List[Structure]] = None

    # -------------------------------------------------------------------------
    # Cached properties
    # -------------------------------------------------------------------------
    @property
    def atom_names(self) -> List[str]:
        """List of atomic species parsed from the ``<atominfo>`` section."""
        if self._atom_names is None:
            self._atom_names = self._parse_atom_names()
        return self._atom_names

    @property
    def structures(self) -> List[Structure]:
        """List of :class:`pymatgen.core.Structure` objects, parsed and cached."""
        if self._structures is None:
            self._structures = self._parse_structures()
        return self._structures

    # -------------------------------------------------------------------------
    # XML Parsing Internals
    # -------------------------------------------------------------------------
    def _parse_atom_names(self) -> List[str]:
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

    def _parse_structures(self) -> List[Structure]:
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
    def forces(self) -> Optional[np.ndarray]:
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

