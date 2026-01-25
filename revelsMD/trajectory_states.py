import numpy as np
from tqdm import tqdm
import MDAnalysis as MD
from lxml import etree  # type: ignore
from abc import ABC, abstractmethod
from typing import List, Union, Optional, Iterator, Tuple
from pymatgen.core import Lattice
from pymatgen.io.ase import AseAtomsAdaptor
from ase.io.cube import write_cube
from revelsMD.revels_tools.lammps_parser import first_read, get_a_frame, define_strngdex, frame_skip
from revelsMD.revels_tools.vasp_parser import Vasprun


class TrajectoryState(ABC):
    """
    Abstract base class defining the interface for trajectory state objects.

    All trajectory backends must implement this interface to ensure consistent
    access patterns across different file formats and data sources.

    Required Attributes
    -------------------
    frames : int
        Number of frames in the trajectory.
    box_x, box_y, box_z : float
        Simulation box dimensions in each Cartesian direction.
    units : str
        Unit system identifier (e.g., 'real', 'metal', 'mda').
    charge_and_mass : bool
        Whether charge and mass data are available for this trajectory.
    """

    # Required attributes - subclasses must set these
    frames: int
    box_x: float
    box_y: float
    box_z: float
    units: str
    charge_and_mass: bool

    def _normalize_bounds(
        self, start: int, stop: Optional[int], stride: int
    ) -> Tuple[int, int, int]:
        """
        Normalize start/stop bounds to handle negative indices Pythonically.

        Parameters
        ----------
        start : int
            Start index (can be negative).
        stop : int or None
            Stop index (can be negative or None for end of trajectory).
        stride : int
            Step between frames.

        Returns
        -------
        tuple of (int, int, int)
            Normalized (start, stop, stride) suitable for use with range().

        Notes
        -----
        Follows Python slice semantics:
        - Negative indices count from the end (e.g., -1 is the last frame)
        - None for stop means iterate to the end
        - Out-of-bounds indices are clamped to valid range
        """
        n = self.frames

        # Handle None stop
        if stop is None:
            stop = n

        # Handle negative start
        if start < 0:
            start = max(0, n + start)

        # Handle negative stop
        if stop < 0:
            stop = max(0, n + stop)

        # Clamp to valid range
        start = min(start, n)
        stop = min(stop, n)

        return start, stop, stride

    @abstractmethod
    def get_indices(self, atype: Union[str, int]) -> np.ndarray:
        """Return atom indices for a given species or type."""
        ...

    @abstractmethod
    def get_charges(self, atype: Union[str, int]) -> np.ndarray:
        """Return atomic charges for atoms of a given species or type."""
        ...

    @abstractmethod
    def get_masses(self, atype: Union[str, int]) -> np.ndarray:
        """Return atomic masses for atoms of a given species or type."""
        ...

    def iter_frames(
        self,
        start: int = 0,
        stop: Optional[int] = None,
        stride: int = 1
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Iterate over trajectory frames, yielding positions and forces.

        Parameters
        ----------
        start : int, optional
            First frame index (default: 0). Negative indices count from end.
        stop : int, optional
            Stop iteration before this frame (default: None, meaning all frames).
            Negative indices count from end.
        stride : int, optional
            Step between frames (default: 1).

        Yields
        ------
        positions : np.ndarray
            Atomic positions for the current frame, shape (n_atoms, 3).
        forces : np.ndarray
            Atomic forces for the current frame, shape (n_atoms, 3).
        """
        start, stop, stride = self._normalize_bounds(start, stop, stride)
        return self._iter_frames_impl(start, stop, stride)

    @abstractmethod
    def _iter_frames_impl(
        self,
        start: int,
        stop: int,
        stride: int
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Internal implementation of frame iteration.

        Subclasses implement this with normalized (non-negative) bounds.
        """
        ...

    @abstractmethod
    def get_frame(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return positions and forces for a specific frame by index.

        Parameters
        ----------
        index : int
            Frame index to retrieve.

        Returns
        -------
        positions : np.ndarray
            Atomic positions for the frame, shape (n_atoms, 3).
        forces : np.ndarray
            Atomic forces for the frame, shape (n_atoms, 3).
        """
        ...


class MDATrajectoryState(TrajectoryState):
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

    def _iter_frames_impl(
        self,
        start: int,
        stop: int,
        stride: int
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Iterate using MDAnalysis trajectory slicing."""
        for ts in self.mdanalysis_universe.trajectory[start:stop:stride]:
            yield ts.positions.copy(), ts.forces.copy()

    def get_frame(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return positions and forces for a specific frame by index."""
        ts = self.mdanalysis_universe.trajectory[index]
        return ts.positions.copy(), ts.forces.copy()


class NumpyTrajectoryState(TrajectoryState):
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

    def get_charges(self, atype: str) -> np.ndarray:
        """Return atomic charges for atoms of a given species."""
        if not self.charge_and_mass:
            raise ValueError("Charge data not available for this trajectory.")
        indices = self.get_indices(atype)
        return self.charge_list[indices]

    def get_masses(self, atype: str) -> np.ndarray:
        """Return atomic masses for atoms of a given species."""
        if not self.charge_and_mass:
            raise ValueError("Mass data not available for this trajectory.")
        indices = self.get_indices(atype)
        return self.mass_list[indices]

    def _iter_frames_impl(
        self,
        start: int,
        stop: int,
        stride: int
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Iterate over in-memory position/force arrays."""
        for i in range(start, stop, stride):
            yield self.positions[i], self.forces[i]

    def get_frame(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return positions and forces for a specific frame by index."""
        return self.positions[index], self.forces[index]


class LammpsTrajectoryState(TrajectoryState):
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

    def _iter_frames_impl(
        self,
        start: int,
        stop: int,
        stride: int
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Parse LAMMPS dump file sequentially for positions and forces."""
        needed_quantities = ["x", "y", "z", "fx", "fy", "fz"]
        strngdex = define_strngdex(needed_quantities, self.dic)

        traj_file = self.trajectory_file
        if isinstance(traj_file, list):
            traj_file = traj_file[0]

        with open(traj_file) as f:
            # Skip to start frame
            if start > 0:
                frame_skip(f, self.num_ats, start, self.header_length)

            frame_idx = start
            while frame_idx < stop:
                data = get_a_frame(f, self.num_ats, self.header_length, strngdex)
                positions = data[:, :3]
                forces = data[:, 3:]
                yield positions, forces

                # Skip (stride - 1) frames before the next read
                frames_to_skip = stride - 1
                frame_idx += 1
                if frame_idx < stop and frames_to_skip > 0:
                    frame_skip(f, self.num_ats, frames_to_skip, self.header_length)
                    frame_idx += frames_to_skip

    def get_frame(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return positions and forces for a specific frame by index.

        LAMMPS dump files are sequential, so random access requires caching
        all frames in memory on first call. Subsequent calls use the cache.
        """
        # Lazy-load cache on first random access
        if not hasattr(self, '_frame_cache'):
            self._frame_cache = list(self.iter_frames())

        positions, forces = self._frame_cache[index]
        return positions, forces


class VaspTrajectoryState(TrajectoryState):
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

    def get_charges(self, atype: str) -> np.ndarray:
        """Return atomic charges for atoms of a given species.

        VASP does not provide charge data in vasprun.xml, so this raises an error.
        """
        raise ValueError("Charge data not available for VASP trajectories.")

    def get_masses(self, atype: str) -> np.ndarray:
        """Return atomic masses for atoms of a given species.

        VASP does not provide mass data in vasprun.xml, so this raises an error.
        """
        raise ValueError("Mass data not available for VASP trajectories.")

    def _iter_frames_impl(
        self,
        start: int,
        stop: int,
        stride: int
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Iterate over in-memory position/force arrays."""
        for i in range(start, stop, stride):
            yield self.positions[i], self.forces[i]

    def get_frame(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return positions and forces for a specific frame by index."""
        return self.positions[index], self.forces[index]
