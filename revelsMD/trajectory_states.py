import numpy as np
from tqdm import tqdm
import MDAnalysis as MD
from lxml import etree  # type: ignore
from typing import List, Union, Optional
from pymatgen.core import Lattice
from pymatgen.io.ase import AseAtomsAdaptor
from ase.io.cube import write_cube
from revelsMD.revels_tools.lammps_parser import first_read
from revelsMD.revels_tools.vasp_parser import Vasprun


class MDATrajectoryState:
    """
    Represents a molecular dynamics trajectory handled by **MDAnalysis**.

    This class acts as a unified interface for reading, validating, and accessing
    data from MDAnalysis-compatible trajectory and topology files.

    Parameters
    ----------
    trajectory_file : str
        Path to the trajectory file (e.g., `.xtc`, `.trr`, `.dcd`, `.lammpstrj`).
    topology_file : str
        Path to the topology file (e.g., `.pdb`, `.gro`, `.data`).

    Attributes
    ----------
    variety : str
        Identifier for trajectory type (`'mda'`).
    frames : int
        Number of trajectory frames.
    box_x, box_y, box_z : float
        Orthorhombic simulation box dimensions in each Cartesian direction.
    units : str
        Unit system identifier (`'mda'`).
    charge_and_mass : bool
        Indicates whether charge and mass data are accessible.

    Raises
    ------
    ValueError
        If no topology file is provided or the box is non-orthorhombic.
    RuntimeError
        If MDAnalysis fails to load the trajectory or topology file.

    Notes
    -----
    - Only orthorhombic or cubic cells are supported (α = β = γ = 90°).
    - For triclinic boxes, preprocessing to orthorhombic form is required.
    """

    def __init__(self, trajectory_file: str, topology_file: str):
        if not topology_file:
            raise ValueError("A topology file is required for MDAnalysis trajectories.")

        self.variety = 'mda'
        self.trajectory_file = trajectory_file
        self.topology_file = topology_file

        try:
            mdanalysis_universe = MD.Universe(topology_file, trajectory_file)
        except Exception as e:
            raise RuntimeError(f"Failed to load MDAnalysis Universe: {e}")

        self.mdanalysis_universe = mdanalysis_universe
        self.frames = len(mdanalysis_universe.trajectory)
        self.charge_and_mass = True
        self.units = 'mda'

        dims = mdanalysis_universe.dimensions
        if len(dims) < 3:
            raise ValueError(f"Invalid simulation box dimensions: {dims}")

        # Safe unpack for older trajectories lacking angular information
        lx, ly, lz = dims[:3]
        alpha, beta, gamma = dims[3:6] if len(dims) >= 6 else (90.0, 90.0, 90.0)

        if not np.allclose([alpha, beta, gamma], 90.0, atol=1e-3):
            raise ValueError("Only orthorhombic or cubic cells are supported.")

        self.box_x, self.box_y, self.box_z = lx, ly, lz

    def get_indices(self, atype: str) -> np.ndarray:
        """
        Return indices of atoms matching a given atom name.

        Parameters
        ----------
        atype : str
            Atom name to select (e.g., `'O'`, `'H'`, `'C'`).

        Returns
        -------
        np.ndarray
            Array of atom indices corresponding to the given atom name.
        """
        return np.array(self.mdanalysis_universe.select_atoms(f'name {atype}').ids)

    get_indicies = get_indices  # backward compatibility alias

    def get_charges(self, atype: str) -> np.ndarray:
        """
        Return atomic charges for atoms of a given name.

        Parameters
        ----------
        atype : str
            Atom name to select.

        Returns
        -------
        np.ndarray
            Array of atomic charges.
        """
        return np.array(self.mdanalysis_universe.select_atoms(f'name {atype}').charges)

    def get_masses(self, atype: str) -> np.ndarray:
        """
        Return atomic masses for atoms of a given name.

        Parameters
        ----------
        atype : str
            Atom name to select.

        Returns
        -------
        np.ndarray
            Array of atomic masses.
        """
        return np.array(self.mdanalysis_universe.select_atoms(f'name {atype}').masses)


class NumpyTrajectoryState:
    """
    Represents a trajectory stored directly as NumPy arrays.

    Designed for simulation data already resident in memory, or for synthetic
    or analytical trajectories generated numerically.

    Parameters
    ----------
    positions : np.ndarray
        Atomic positions of shape ``(frames, atoms, 3)``.
    forces : np.ndarray
        Atomic forces of shape ``(frames, atoms, 3)``.
    box_x, box_y, box_z : float
        Simulation box lengths in each Cartesian direction.
    species_list : list of str
        Atom names corresponding to each atom index.
    units : str, optional
        Unit system string (default: `'real'`).
    charge_list : np.ndarray, optional
        Atomic charge array (optional).
    mass_list : np.ndarray, optional
        Atomic mass array (optional).

    Raises
    ------
    ValueError
        If positions and forces are inconsistent, or if box dimensions are invalid.

    Notes
    -----
    This class provides a simple in-memory structure compatible with the
    ``Revels3D`` and ``RevelsRDF`` interfaces.
    """

    def __init__(
        self,
        positions: np.ndarray,
        forces: np.ndarray,
        box_x: float,
        box_y: float,
        box_z: float,
        species_list: List[str],
        units: str = 'real',
        charge_list: Optional[np.ndarray] = None,
        mass_list: Optional[np.ndarray] = None,
    ):
        if positions.shape != forces.shape:
            raise ValueError("Force and position arrays are incommensurate.")

        if positions.shape[1] != len(species_list):
            raise ValueError("Species list and trajectory arrays are incommensurate.")

        if not all(val > 0 for val in (box_x, box_y, box_z)):
            raise ValueError("Box dimensions must all be positive values.")

        self.variety = 'numpy'
        self.positions = positions
        self.forces = forces
        self.species_string = species_list
        self.box_x = box_x
        self.box_y = box_y
        self.box_z = box_z
        self.units = units
        self.frames = positions.shape[0]
        self.charge_and_mass = bool(charge_list is not None and mass_list is not None)

        if charge_list is not None:
            self.charge_list = charge_list
        if mass_list is not None:
            self.mass_list = mass_list

    def get_indices(self, atype: str) -> np.ndarray:
        """
        Return atom indices for a given species.

        Parameters
        ----------
        atype : str
            Atom species name to select (e.g., `'O'`, `'H'`, `'C'`).

        Returns
        -------
        np.ndarray
            Indices of selected atoms.

        Raises
        ------
        ValueError
            If the species name is not present in the provided species list.
        """
        inds = np.where(np.array(self.species_string) == atype)[0]
        if len(inds) == 0:
            raise ValueError(f"Species '{atype}' not found in species list.")
        return inds

    get_indicies = get_indices


class LammpsTrajectoryState:
    """
    Represents a molecular dynamics trajectory obtained from LAMMPS output.

    Parses the LAMMPS trajectory file to obtain metadata (frames, atoms, box size),
    and loads coordinates via MDAnalysis for compatibility with the rest of RevelsMD.

    Parameters
    ----------
    trajectory_file : str or list of str
        Path(s) to LAMMPS dump files.
    topology_file : str
        Path to corresponding LAMMPS data or topology file.
    units : str, optional
        LAMMPS unit system (default: `'real'`).
    atom_style : str, optional
        LAMMPS atom style (default: `'full'`).
    charge_and_mass : bool, optional
        Whether charge/mass data are accessible (default: ``True``).

    Raises
    ------
    ValueError
        If the cell is not orthorhombic or box dimensions are invalid.
    RuntimeError
        If the trajectory cannot be parsed by MDAnalysis.
    """

    def __init__(
        self,
        trajectory_file: Union[str, List[str]],
        topology_file: Optional[str] = None,
        units: str = 'real',
        atom_style: str = 'full',
        charge_and_mass: bool = True,
    ):
        self.variety = 'lammps'
        self.trajectory_file = trajectory_file
        self.topology_file = topology_file
        self.units = units
        self.charge_and_mass = charge_and_mass

        if topology_file is None:
            raise ValueError("A topology file is required for LAMMPS trajectories.")

        if isinstance(trajectory_file, list):
            first_traj = trajectory_file[0]
            all_trajs = trajectory_file
        else:
            first_traj = trajectory_file
            all_trajs = [trajectory_file]

        try:
            self.frames, self.num_ats, self.dic, self.header_length, self.dimgrid = first_read(first_traj)
        except Exception as e:
            raise RuntimeError(f"Failed to parse LAMMPS trajectory header: {e}")

        try:
            mdanalysis_universe = MD.Universe(topology_file, *all_trajs, atom_style=atom_style,format="LAMMPSDump")
        except Exception as e:
            raise RuntimeError(f"Failed to load LAMMPS trajectory with MDAnalysis: {e}")

        self.mdanalysis_universe = mdanalysis_universe
        self.frames = len(mdanalysis_universe.trajectory)

        dims = mdanalysis_universe.dimensions
        if len(dims) < 6:
            raise ValueError(f"Invalid LAMMPS box dimensions: {dims}")

        lx, ly, lz, alpha, beta, gamma = dims[:6]

        if not np.allclose([alpha, beta, gamma], 90.0, atol=1e-3):
            raise ValueError("Only orthorhombic or cubic boxes are supported.")

        if not all(np.isfinite([lx, ly, lz])) or not all(val > 0 for val in (lx, ly, lz)):
            raise ValueError(f"Invalid box dimensions: ({lx}, {ly}, {lz})")

        self.box_x, self.box_y, self.box_z = lx, ly, lz

    def get_indices(self, atype: Union[int, str]) -> np.ndarray:
        """Return atom indices for a given LAMMPS atom type."""
        return self.mdanalysis_universe.select_atoms(f'type {atype}').ids - 1

    get_indicies = get_indices

    def get_charges(self, atype: Union[int, str]) -> np.ndarray:
        """Return atomic charges for a given LAMMPS atom type."""
        return self.mdanalysis_universe.select_atoms(f'type {atype}').charges

    def get_masses(self, atype: Union[int, str]) -> np.ndarray:
        """Return atomic masses for a given LAMMPS atom type."""
        return self.mdanalysis_universe.select_atoms(f'type {atype}').masses


class VaspTrajectoryState:
    """
    Represents a molecular dynamics trajectory obtained from VASP ``vasprun.xml`` output.

    Supports parsing one or multiple sequential ``vasprun.xml`` files, validating
    orthorhombicity, and providing cartesian coordinates and forces as NumPy arrays.

    Parameters
    ----------
    trajectory_file : str or list of str
        Path or list of paths to ``vasprun.xml`` file(s).

    Attributes
    ----------
    variety : str
        Identifier (`'vasp'`).
    frames : int
        Total number of time steps across all vasprun files.
    box_x, box_y, box_z : float
        Simulation box lengths along each Cartesian axis.
    positions : np.ndarray
        Cartesian coordinates array of shape ``(frames, atoms, 3)``.
    forces : np.ndarray
        Cartesian forces array of shape ``(frames, atoms, 3)``.

    Raises
    ------
    ValueError
        If no forces are found or if the lattice is non-orthorhombic.

    Notes
    -----
    - The parser enforces presence of forces for physical completeness.
    - For NVT/NVE MD, ensure `IBRION=-1` and `NSW > 0` during VASP runs.
    """

    def __init__(self, trajectory_file: Union[str, List[str]]):
        self.variety = 'vasp'
        self.units = 'metal'
        self.charge_and_mass = False

        if isinstance(trajectory_file, list):
            self.trajectory_file = trajectory_file
            self.Vasprun = Vasprun(trajectory_file[0])
            self.Vasprun.start = self.Vasprun.structures[0]
            self.frames = len(self.Vasprun.structures)
            start = self.Vasprun.structures[0]

            self._validate_cell(self.Vasprun.start.lattice)
            self.box_x, self.box_y, self.box_z = np.diag(self.Vasprun.start.lattice.matrix)
            self.positions = self.Vasprun.cart_coords
            self.forces = self.Vasprun.forces
            if self.forces is None:
                raise ValueError(f"No forces found in {trajectory_file[0]}")

            for item in trajectory_file[1:]:
                next_run = Vasprun(item)
                if next_run.forces is None:
                    raise ValueError(f"No forces found in {item}")
                self.frames += len(next_run.structures)
                self.positions = np.append(self.positions, next_run.cart_coords, axis=0)
                self.forces = np.append(self.forces, next_run.forces, axis=0)

            self.Vasprun.start = start

        else:
            self.trajectory_file = trajectory_file
            self.Vasprun = Vasprun(trajectory_file)
            self.Vasprun.start = self.Vasprun.structures[0]
            self.frames = len(self.Vasprun.structures)

            self._validate_cell(self.Vasprun.start.lattice)
            self.box_x, self.box_y, self.box_z = np.diag(self.Vasprun.start.lattice.matrix)
            self.positions = self.Vasprun.cart_coords
            self.forces = self.Vasprun.forces
            if self.forces is None:
                raise ValueError(f"No forces found in {trajectory_file}")

    @staticmethod
    def _validate_cell(lattice: Lattice):
        """Validate that the VASP lattice is orthorhombic."""
        if not np.allclose(lattice.angles, 90.0, atol=1e-3):
            raise ValueError(
                "Non-orthorhombic or non-cubic cell detected. "
                "Only orthorhombic/cubic cells are supported."
            )

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
        return self.Vasprun.start.indices_from_symbol(atype)

