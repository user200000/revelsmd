"""GridState class for accumulating 3D force fields and converting to densities."""

from __future__ import annotations

import copy

import numpy as np
from tqdm import tqdm
from ase import Atoms
from ase.io.cube import write_cube
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from revelsMD.trajectories._base import Trajectory
from revelsMD.utils import generate_boltzmann
from revelsMD.density.selection_state import SelectionState
from revelsMD.density.estimators import Estimators


class GridState:
    """
    State for accumulating 3D force fields and converting to densities.

    Parameters
    ----------
    trajectory : Trajectory
        Trajectory-state object providing `box_x`, `box_y`, `box_z`, and `units`.
    density_type : {'number', 'charge', 'polarisation'}
        Type of density to be constructed (controls the estimator weighting).
    temperature : float
        System temperature (K) used for force->density conversion in k-space.
    nbins : int, optional
        Default number of voxels per box dimension (overridden by `nbinsx/y/z`).
    nbinsx, nbinsy, nbinsz : int or bool, optional
        Explicit voxel counts per dimension. If `False`, fall back to `nbins`.

    Attributes
    ----------
    forceX, forceY, forceZ : np.ndarray
        Accumulators for voxelized force components (shape: nbinsx x nbinsy x nbinsz).
    counter : np.ndarray
        Accumulator for counting-based density (same shape as force accumulators).
    binsx, binsy, binsz : np.ndarray
        Bin edges per dimension (inclusive right edge by construction here).
    lx, ly, lz : float
        Voxel sizes in x/y/z.
    voxel_volume : float
        Volume of a single voxel.
    count : int
        Number of processed frames (for normalization).
    grid_progress : {'Generated', 'Allocated', 'Lambda'}
        Simple state flag used by getters and lambda estimator.
    """

    def __init__(
        self,
        trajectory: Trajectory,
        density_type: str,
        temperature: float,
        nbins: int = 100,
        nbinsx: int | bool = False,
        nbinsy: int | bool = False,
        nbinsz: int | bool = False,
    ):
        # Resolve per-dimension bin counts
        nbinsx = nbins if nbinsx is False else int(nbinsx)
        nbinsy = nbins if nbinsy is False else int(nbinsy)
        nbinsz = nbins if nbinsz is False else int(nbinsz)
        if min(nbinsx, nbinsy, nbinsz) <= 0:
            raise ValueError("nbinsx, nbinsy, nbinsz must be positive integers.")

        # Voxel sizes
        lx = trajectory.box_x / nbinsx
        ly = trajectory.box_y / nbinsy
        lz = trajectory.box_z / nbinsz
        if min(lx, ly, lz) <= 0:
            raise ValueError("Box lengths must be positive to define voxel sizes.")

        # Box and bins
        self.box_x = trajectory.box_x
        self.box_y = trajectory.box_y
        self.box_z = trajectory.box_z
        self.box_array = np.array([trajectory.box_x, trajectory.box_y, trajectory.box_z])
        self.binsx = np.arange(0, trajectory.box_x + lx, lx)
        self.binsy = np.arange(0, trajectory.box_y + ly, ly)
        self.binsz = np.arange(0, trajectory.box_z + lz, lz)

        # Bookkeeping
        self.voxel_volume = float(np.prod(self.box_array) / (nbinsx * nbinsy * nbinsz))
        self.temperature = float(temperature)
        self.lx, self.ly, self.lz = float(lx), float(ly), float(lz)
        self.count = 0
        self.units = trajectory.units
        self.nbinsx, self.nbinsy, self.nbinsz = nbinsx, nbinsy, nbinsz

        # Accumulators
        self.forceX = np.zeros((nbinsx, nbinsy, nbinsz), dtype=float)
        self.forceY = np.zeros((nbinsx, nbinsy, nbinsz), dtype=float)
        self.forceZ = np.zeros((nbinsx, nbinsy, nbinsz), dtype=float)
        self.counter = np.zeros((nbinsx, nbinsy, nbinsz), dtype=float)

        # Density selection
        dty = density_type.lower().strip()
        if dty not in {"number", "charge", "polarisation"}:
            raise ValueError("density_type must be one of {'number','charge','polarisation'}.")
        self.density_type = dty

        # Progress flag
        self.grid_progress = "Generated"

    def make_force_grid(
        self,
        trajectory: Trajectory,
        atom_names: str | list[str],
        rigid: bool = False,
        centre_location: bool | int = True,
        kernel: str = "triangular",
        polarisation_axis: int = 0,
        start: int = 0,
        stop: int | None = None,
        period: int = 1,
    ) -> None:
        """
        Accumulate per-frame force contributions on the voxel grid.

        Parameters
        ----------
        trajectory : Trajectory
            Trajectory-state object with positions/forces.
        atom_names : str or list of str
            Atom name(s) used for selection (single species or multi-species molecule).
        rigid : bool, optional
            Treat listed species as a rigid group and sum forces (default: False).
            For a species to be treated as rigid each atom in the molecule must have a different name.

        centre_location : bool or int, optional
            If `True`, use center-of-mass; if `int`, use that atom index within the
            rigid set as the reference position (default: True).
        kernel : {'triangular','box'}, optional
            Deposition kernel for grid assignment (default: 'triangular').
        polarisation_axis : int, optional
            Axis index for polarisation projection if `density_type='polarisation'`.
        start : int, optional
            Start frame index (default: 0). Negative indices count from end.
        stop : int or None, optional
            Stop frame index (default: None, meaning all frames).
            Negative indices count from end (e.g., -1 = all but last).
        period : int, optional
            Frame stride (default: 1).

        Raises
        ------
        ValueError
            On invalid frame selection, unsupported density/kernel combinations,
            or malformed inputs for rigid/centre options.
        """
        # --- Validate atom_names ---
        if isinstance(atom_names, str):
            atom_list = atom_names.replace(',', ' ').split()
            if len(atom_list) != len(set(atom_list)):
                raise ValueError(f"Duplicate atom names detected in input string: {atom_names!r}")
        elif isinstance(atom_names, list):
            if len(atom_names) != len(set(atom_names)):
                raise ValueError(f"Duplicate atom names detected in list: {atom_names!r}")
        else:
            raise ValueError("`atom_names` must be a string or list of strings.")

        # Validate frame bounds
        if start > trajectory.frames:
            raise ValueError("First frame index exceeds frames in trajectory.")
        self.start = start

        if stop is not None and stop > trajectory.frames:
            raise ValueError("Final frame index exceeds frames in trajectory.")
        self.stop = stop

        # Calculate to_run for progress bar - normalize bounds for range()
        norm_start = start % trajectory.frames if start >= 0 else max(0, trajectory.frames + start)
        if stop is None:
            norm_stop = trajectory.frames
        elif stop < 0:
            norm_stop = max(0, trajectory.frames + stop)
        else:
            norm_stop = stop
        to_run = range(int(norm_start), int(norm_stop), period)
        if len(to_run) == 0:
            raise ValueError("Final frame occurs before first frame in trajectory.")
        self.period = period
        self.kernel = kernel
        self.to_run = to_run

        # Build selection wrapper (keeps original attribute spellings)
        self.SS = SelectionState(trajectory, atom_names=atom_names, centre_location=centre_location, rigid=rigid)

        # Choose estimator based on density type and rigid settings
        if self.density_type == "number":
            if not self.SS.indistinguishable_set:
                if rigid:
                    if centre_location is True:
                        self.single_frame_function = Estimators.single_frame_rigid_number_com_grid
                    elif isinstance(centre_location, int):
                        self.single_frame_function = Estimators.single_frame_rigid_number_atom_grid
                    else:
                        raise ValueError("centre_location must be True (COM) or int (specific atom index).")
                else:
                    self.single_frame_function = Estimators.single_frame_number_many_grid
            else:
                self.single_frame_function = Estimators.single_frame_number_single_grid

        elif self.density_type == "charge":
            if not self.SS.indistinguishable_set:
                if rigid:
                    if centre_location is True:
                        self.single_frame_function = Estimators.single_frame_rigid_charge_com_grid
                    elif isinstance(centre_location, int):
                        self.single_frame_function = Estimators.single_frame_rigid_charge_atom_grid
                    else:
                        raise ValueError("centre_location must be True (COM) or int (specific atom index).")
                else:
                    self.single_frame_function = Estimators.single_frame_charge_many_grid
            else:
                self.single_frame_function = Estimators.single_frame_number_single_grid

        elif self.density_type == "polarisation":
            if not self.SS.indistinguishable_set:
                if rigid:
                    if centre_location is True:
                        self.single_frame_function = Estimators.single_frame_rigid_polarisation_com_grid
                        self.SS.polarisation_axis = polarisation_axis
                    elif isinstance(centre_location, int):
                        self.single_frame_function = Estimators.single_frame_rigid_polarisation_atom_grid
                        self.SS.polarisation_axis = polarisation_axis
                    else:
                        raise ValueError("centre_location must be True (COM) or an integer atom index.")
                else:
                    raise ValueError("Polarisation densities are only implemented for rigid molecules.")
            else:
                raise ValueError("A single atom does not have a polarisation density; specify a rigid molecule.")
        else:
            raise ValueError("Supported densities: 'number', 'polarisation', 'charge'.")

        # Unified frame iteration using iter_frames
        for positions, forces in tqdm(trajectory.iter_frames(start, stop, period), total=len(self.to_run)):
            self.single_frame_function(
                positions, forces, trajectory, self, self.SS, kernel=self.kernel
            )

        self.frames_processed = self.to_run
        self.grid_progress = "Allocated"

    def get_real_density(self) -> None:
        """
        Convert accumulated force field to real-space density via FFT.

        Notes
        -----
        Implements (Borgis et al., *Mol. Phys.* **111**, 3486-3492 (2013)):
        delta_rho(k) = i / (k_B T k^2) * k . F(k), with delta_rho(k=0) := 0,
        then rho(r) = <rho_count(r)> + F^-1[delta_rho(k)].

        Side Effects
        ------------
        Sets `self.del_rho_k`, `self.del_rho_n`, `self.particle_density`, `self.rho`.
        """
        if self.grid_progress == "Generated":
            raise RuntimeError("Run make_force_grid before computing densities.")

        # Normalize by number of frames and voxel volume before FFT
        with np.errstate(divide="ignore", invalid="ignore"):
            forceX = np.fft.fftn(self.forceX / self.count / self.voxel_volume)
            forceY = np.fft.fftn(self.forceY / self.count / self.voxel_volume)
            forceZ = np.fft.fftn(self.forceZ / self.count / self.voxel_volume)

        # k-vectors per dimension
        xrep, yrep, zrep = self.get_kvectors()

        # Multiply by k components (component-wise dot in spectral space)
        for n in range(len(xrep)):
            forceX[n, :, :] = xrep[n] * forceX[n, :, :]
        for m in range(len(yrep)):
            forceY[:, m, :] = yrep[m] * forceY[:, m, :]
        for l in range(len(zrep)):
            forceZ[:, :, l] = zrep[l] * forceZ[:, :, l]

        # delta_rho(k): i/(k_B T k^2) * (F.k); handle k^2=0 via errstate; enforce delta_rho(0)=0
        with np.errstate(divide="ignore", invalid="ignore"):
            self.del_rho_k = (
                complex(0, 1)
                / (self.temperature * generate_boltzmann(self.units) * self.get_ksquared())
                * (forceX + forceY + forceZ)
            )
        self.del_rho_k[0, 0, 0] = 0.0

        # Back to real space (density perturbation), add mean counting density
        del_rho_n = np.fft.ifftn(self.del_rho_k)
        self.del_rho_n = -1.0 * np.real(del_rho_n)

        # Conventional counting density (averaged)
        self.get_particle_density()
        self.rho = self.del_rho_n + np.mean(self.particle_density)

    def get_particle_density(self) -> None:
        """
        Compute conventional counting-based density from `self.counter`.

        Side Effects
        ------------
        Sets `self.particle_density`.
        """
        if self.grid_progress == "Generated":
            raise RuntimeError("Run make_force_grid before computing densities.")
        with np.errstate(divide="ignore", invalid="ignore"):
            self.particle_density = self.counter / self.voxel_volume / self.count

    def get_kvectors(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return the k-vectors (2*pi * FFT frequencies) per dimension.

        Returns
        -------
        (kx, ky, kz) : tuple of np.ndarray
            Frequency vectors for x, y, z (units: 1/length).
        """
        xrep = 2 * np.pi * np.fft.fftfreq(self.nbinsx, d=self.lx)
        yrep = 2 * np.pi * np.fft.fftfreq(self.nbinsy, d=self.ly)
        zrep = 2 * np.pi * np.fft.fftfreq(self.nbinsz, d=self.lz)
        return xrep, yrep, zrep

    def get_ksquared(self) -> np.ndarray:
        """
        Return k^2 on the 3D grid (broadcasted from 1D k-vectors).

        Returns
        -------
        np.ndarray
            k^2 array with shape (nbinsx, nbinsy, nbinsz).
        """
        xrep, yrep, zrep = self.get_kvectors()

        # Broadcast 1D k-vectors to a 3D grid
        xrep_3d = np.repeat(np.repeat(xrep[:, None, None], self.nbinsy, axis=1), self.nbinsz, axis=2)
        yrep_3d = np.repeat(np.repeat(yrep[None, :, None], self.nbinsx, axis=0), self.nbinsz, axis=2)
        zrep_3d = np.repeat(np.repeat(zrep[None, None, :], self.nbinsx, axis=0), self.nbinsy, axis=1)

        return xrep_3d * xrep_3d + yrep_3d * yrep_3d + zrep_3d * zrep_3d

    def write_to_cube(
        self,
        structure: Structure | Atoms,
        grid: np.ndarray,
        filename: str,
    ) -> None:
        """
        Write a 3D density grid to a Gaussian `.cube` file.

        Parameters
        ----------
        structure : pymatgen.Structure or ase.Atoms
            Input structure used to define the cell geometry. Pymatgen structures
            are automatically converted to ASE Atoms.
        grid : np.ndarray
            3D grid data to write (shape: nbinsx x nbinsy x nbinsz).
        filename : str
            Output filename.

        Notes
        -----
        - Atom deletion occurs **after** conversion from pymatgen to ASE,
          preserving your original intent.
        """
        # Convert from pymatgen if needed; otherwise assume ASE Atoms
        if isinstance(structure, Structure):
            atoms = AseAtomsAdaptor.get_atoms(structure)
        else:
            atoms = structure

        # Optional removal (e.g., to drop solute atoms from density writeout)
        # Keep attribute name `indices` for backward compatibility
        if hasattr(self, "SS") and hasattr(self.SS, "indices") and self.SS.indices is not None:
            try:
                del atoms[np.array(self.SS.indices)]
            except Exception:
                # If selection is a list of arrays (multi-species), do nothing silently
                pass

        with open(filename, "w") as f:
            write_cube(f, atoms, data=grid)

    def get_lambda(self, trajectory: Trajectory, sections: int | None = None) -> "GridState":
        """
        Compute optimal lambda(r) to combine counting and force densities.

        Implements the linear-combination approach of:
        J. Chem. Phys. 154, 191101 (2021).

        Parameters
        ----------
        trajectory : Trajectory
            Trajectory-state providing per-frame positions and forces.
        sections : int, optional
            Number of interleaved frame-subsets used to accumulate covariance
            buffers. If None, defaults to `trajectory.frames`.

        Returns
        -------
        GridState
            A deep-copied `GridState` instance with
            `expected_rho`, `expected_particle_density`, `delta`,
            covariance/variance buffers, `combination` (=1-cov_F/var),
            and `optimal_density` populated.

        Raises
        ------
        RuntimeError
            If called before `make_force_grid`.
        ValueError
            If called on a grid already produced by `get_lambda`.
        """
        if self.grid_progress == "Generated":
            raise RuntimeError("Run make_force_grid before estimating lambda.")
        if self.grid_progress == "Lambda":
            raise ValueError("This grid was already produced by get_lambda; re-run upstream to refresh.")

        GS_Lambda = copy.deepcopy(self)
        if sections is None:
            sections = trajectory.frames

        # Baseline expectation from full accumulation
        GS_Lambda.get_real_density()
        GS_Lambda.expected_rho = np.copy(GS_Lambda.rho)
        GS_Lambda.expected_particle_density = np.copy(GS_Lambda.particle_density)
        GS_Lambda.delta = GS_Lambda.expected_rho - GS_Lambda.expected_particle_density

        # Covariance/variance accumulators
        GS_Lambda.cov_buffer_particle = np.zeros((GS_Lambda.nbinsx, GS_Lambda.nbinsy, GS_Lambda.nbinsz))
        GS_Lambda.cov_buffer_force = np.zeros((GS_Lambda.nbinsx, GS_Lambda.nbinsy, GS_Lambda.nbinsz))
        GS_Lambda.var_buffer = np.zeros((GS_Lambda.nbinsx, GS_Lambda.nbinsy, GS_Lambda.nbinsz))

        # Interleaved accumulation across sections
        for k in tqdm(range(sections)):
            # Reset accumulators for this section
            GS_Lambda.forceX *= 0
            GS_Lambda.forceY *= 0
            GS_Lambda.forceZ *= 0
            GS_Lambda.particle_density *= 0
            GS_Lambda.counter *= 0
            GS_Lambda.del_rho_k *= 0
            GS_Lambda.del_rho_n *= 0
            GS_Lambda.rho *= 0
            GS_Lambda.count *= 0

            # Frame indices for this section (interleaved sampling)
            frame_indices = np.array(GS_Lambda.to_run)[
                np.arange(k, sections * (len(GS_Lambda.to_run) // sections), sections)
            ]
            for frame_idx in frame_indices:
                positions, forces = trajectory.get_frame(frame_idx)
                GS_Lambda.single_frame_function(
                    positions, forces, trajectory, GS_Lambda, GS_Lambda.SS, kernel=GS_Lambda.kernel
                )

            # Compute densities for this section and accumulate statistics
            GS_Lambda.get_real_density()
            delta_cur = GS_Lambda.rho - GS_Lambda.particle_density
            GS_Lambda.var_buffer += (delta_cur - GS_Lambda.delta) ** 2
            GS_Lambda.cov_buffer_force += (delta_cur - GS_Lambda.delta) * (GS_Lambda.rho - GS_Lambda.expected_rho)
            GS_Lambda.cov_buffer_particle += (delta_cur - GS_Lambda.delta) * (
                GS_Lambda.particle_density - GS_Lambda.expected_particle_density
            )

        # lambda = 1 - cov(force)/var(delta); optimal density = (1-lambda)*count + lambda*force
        GS_Lambda.combination = 1.0 - (GS_Lambda.cov_buffer_force / GS_Lambda.var_buffer)
        GS_Lambda.optimal_density = (1.0 - GS_Lambda.combination) * GS_Lambda.expected_particle_density + \
            GS_Lambda.combination * GS_Lambda.expected_rho
        GS_Lambda.grid_progress = "Lambda"
        return GS_Lambda
