"""MDAnalysis backend: MDATrajectory."""

import os
import warnings
from collections.abc import Generator, Mapping

from MDAnalysis.coordinates.H5MD import H5MDReader  # type: ignore[import-untyped]
from MDAnalysis.coordinates.LAMMPS import DumpReader  # type: ignore[import-untyped]
from MDAnalysis.coordinates.chain import ChainReader  # type: ignore[import-untyped]
from MDAnalysis.lib.mdamath import triclinic_vectors  # type: ignore[import-untyped]
from MDAnalysis.lib.util import NamedStream  # type: ignore[import-untyped]
import numpy as np

from revelsMD.frame_sources import Frame
from ._base import (
    TrajectoryFiles, TrajectoryPath, Trajectory, as_paths, check_fixed_cell, open_universe,
    NOT_PARSE_ERRORS, atom_values, given_values, locate, reading_with_mdanalysis, topology_values,
    warn_cut_short,
)


class MDATrajectory(Trajectory):
    """A trajectory read through MDAnalysis, in Angstrom and kJ/(mol Angstrom).

    The format must record its length and force units (GROMACS TRR and TNG,
    AMBER NetCDF, H5MD with units), and every frame must hold positions,
    forces and the first frame's box. Several files are concatenated in
    the order given; frames they share are read from each.

    Parameters
    ----------
    trajectory_file : str, path, or sequence of these
        Trajectory file(s).
    topology_file : str, bytes or path
        Topology file (e.g. ``.tpr``, ``.gro``), or an MDAnalysis stream.
    temperature : float
        Temperature in Kelvin.
    masses, charges : mapping, optional
        Masses or charges by atom name (e.g. ``{'Ow': 15.999}``), replacing
        the topology's for those names. Neither is guessed.

    Raises
    ------
    ValueError
        If no file is given, a given mass or charge is not a finite number
        or names no atom, the format records no length or force units,
        an H5MD file saves positions and forces at different steps, or the
        first frame has no box, positions or forces, or an invalid cell.
    RuntimeError
        If MDAnalysis cannot read a file (e.g. H5MD without ``h5py``).
    OSError
        If a file cannot be opened.
    ImportError
        If a TNG file is read without ``pytng``.

    Notes
    -----
    ``mdanalysis_universe.trajectory`` also holds a final frame that cannot
    be read, which ``frames`` does not count. See
    :doc:`/tutorials/trajectory-formats` for the formats and settings that
    work.
    """

    def __init__(
        self,
        trajectory_file: TrajectoryFiles,
        topology_file: "TrajectoryPath | object",
        *,
        temperature: float,
        masses: Mapping[str, float] | None = None,
        charges: Mapping[str, float] | None = None,
    ):
        if not topology_file:
            raise ValueError("a topology file is required for MDAnalysis trajectories")
        paths = as_paths(trajectory_file)

        super().__init__(units='mda', temperature=temperature)

        self.trajectory_file = trajectory_file
        self.topology_file = topology_file

        # MDAnalysis streams are passed on as they are.
        topology = (
            os.fsdecode(topology_file)
            if isinstance(topology_file, (str, bytes, os.PathLike))
            and not isinstance(topology_file, NamedStream)
            else topology_file
        )
        self._topology = str(topology)
        mdanalysis_universe = open_universe(topology, paths)

        self.mdanalysis_universe = mdanalysis_universe
        def names() -> list[str]:
            return list(mdanalysis_universe.atoms.names)
        self._masses = given_values(masses, names, "masses", "atom names")
        self._charges = given_values(charges, names, "charges", "atom names")
        trajectory = mdanalysis_universe.trajectory
        readers = _readers(trajectory)
        self._paths = paths

        dims = mdanalysis_universe.dimensions
        if dims is None:
            raise ValueError(
                f"{paths[0]}: the trajectory has no simulation box; "
                "a periodic box is required"
            )
        dims = np.array(dims, dtype=np.float64)  # a copy: reading frames changes it
        if not trajectory.ts.has_positions:
            raise ValueError(
                f"{paths[0]}: the trajectory has no positions; "
                "positions are needed in every frame"
            )
        if not trajectory.ts.has_forces:
            raise ValueError(
                f"{paths[0]}: the trajectory has no forces; force "
                "sampling needs forces in every frame"
            )
        complete = []
        for reader, path in zip(readers, paths, strict=True):
            _check_reader(reader)
            n = len(reader)
            if n and not _readable(reader, n - 1):
                n -= 1
                warn_cut_short(path, f"frame (frame {n})")
            complete.append(n)
        trajectory.rewind()
        # Frame i is frame i - _first[k] of file k, which is MDAnalysis frame
        # i - _first[k] + _mdanalysis_first[k]; a cut final frame is skipped.
        self._first = np.concatenate([[0], np.cumsum(complete)])
        self._mdanalysis_first = np.concatenate([[0], np.cumsum([len(r) for r in readers])])
        self.frames = int(self._first[-1])
        self._dimensions = dims
        self.cell_matrix = np.array(triclinic_vectors(dims), dtype=np.float64)
        self._validate_cell_matrix(self.cell_matrix, paths[0])

    def get_indices(self, atype: str) -> np.ndarray:
        """Return the indices of the atoms with a given name.

        Parameters
        ----------
        atype : str
            Atom name (e.g. ``'Ow'``).

        Returns
        -------
        np.ndarray
            Row positions in the frames, independent of the topology's atom
            numbering.

        Raises
        ------
        ValueError
            If no atom has that name.
        """
        indices = np.array(self.mdanalysis_universe.select_atoms(f'name {atype}').ix)
        if len(indices) == 0:
            raise ValueError(f"{self._topology}: no atoms named '{atype}'")
        return indices

    def get_charges(self, atype: str) -> np.ndarray:
        """Return the charges of the atoms with a given name.

        Parameters
        ----------
        atype : str
            Atom name.

        Raises
        ------
        DataUnavailableError
            If neither ``charges`` nor the topology gives them.
        """
        atoms = self.mdanalysis_universe.select_atoms(f'name {atype}')
        return atom_values(atoms.names, "charges", self._charges,
                           lambda: topology_values(atoms, "charges"), f"{self._topology}: the topology")

    def get_masses(self, atype: str) -> np.ndarray:
        """Return the masses of the atoms with a given name.

        Parameters
        ----------
        atype : str
            Atom name.

        Raises
        ------
        DataUnavailableError
            If neither ``masses`` nor the topology gives them.
        """
        atoms = self.mdanalysis_universe.select_atoms(f'name {atype}')
        return atom_values(atoms.names, "masses", self._masses,
                           lambda: topology_values(atoms, "masses"), f"{self._topology}: the topology")

    def _where(self, index: int) -> str:
        k, local = locate(self._first, index)
        return f"{self._paths[k]}, frame {local}"

    def _get_frame(self, index: int) -> Frame:
        """Read frame ``index`` with MDAnalysis and check it."""
        where = self._where(index)
        k, local = locate(self._first, index)
        with reading_with_mdanalysis(where, ValueError):
            ts = self.mdanalysis_universe.trajectory[int(self._mdanalysis_first[k]) + local]
        if not ts.has_positions:
            raise ValueError(f"{where}: no positions")
        if not ts.has_forces:
            raise ValueError(f"{where}: no forces; force sampling needs forces in every frame")
        if ts.dimensions is None:
            raise ValueError(f"{where}: no simulation box")
        check_fixed_cell(np.array(ts.dimensions, dtype=np.float64), self._dimensions, where)
        return Frame(ts.positions, ts.forces)

    def _iter_frames_impl(
        self,
        start: int,
        stop: int,
        stride: int
    ) -> Generator[Frame, None, None]:
        """Read the selected frames in order."""
        for index in range(start, stop, stride):
            yield self._get_frame(index)


def _readable(reader, frame: int) -> bool:
    """Whether MDAnalysis can read ``frame`` of ``reader``, with its warnings silenced."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reader[frame]
    except NOT_PARSE_ERRORS:
        raise
    except Exception:
        return False
    return True


def _check_reader(reader) -> None:
    """Raise unless MDAnalysis converts a reader's positions and forces to its units.

    MDAnalysis returns the file's numbers unchanged for formats that do not
    record their units. The H5MD reader pairs positions and forces by their
    order in the file, so their steps must agree (RuntimeError if the steps
    cannot be read).
    """
    missing = [quantity for quantity in ("length", "force") if reader.units.get(quantity) is None]
    if missing:
        hint = "; read LAMMPS dumps with LammpsTrajectory" if isinstance(reader, DumpReader) else ""
        raise ValueError(
            f"{reader.filename}: MDAnalysis knows no {' or '.join(missing)} units "
            f"for this format ({type(reader).__name__}), which may store no "
            f"forces at all, so forces in kJ/(mol Angstrom) cannot be read{hint}"
        )
    if not isinstance(reader, H5MDReader):
        return
    cannot_check = (
        f"{reader.filename}: cannot check that this H5MD file saves positions "
        "and forces at the same steps"
    )
    particles = getattr(reader, "_particle_group", None)
    if particles is None:
        raise RuntimeError(f"{cannot_check} with this version of MDAnalysis")
    if "position" in particles and "force" in particles:
        try:
            position_steps = particles["position/step"][()]
            force_steps = particles["force/step"][()]
        except (KeyError, OSError, TypeError) as e:
            raise RuntimeError(f"{cannot_check}: {e}") from e
        if not np.array_equal(position_steps, force_steps):
            raise ValueError(
                f"{reader.filename}: the forces are saved at different steps "
                "from the positions; every frame needs both"
            )


def _readers(trajectory) -> list:
    """The readers of an MDAnalysis trajectory, one per file."""
    if isinstance(trajectory, ChainReader):
        return list(trajectory.readers)
    return [trajectory]

