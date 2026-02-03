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
from revelsMD.density.selection_state import SelectionState
from revelsMD.density.grid_helpers import get_backend_functions as _get_grid_backend_functions

# Module-level backend functions (loaded once at import)
_triangular_allocation, _box_allocation = _get_grid_backend_functions()


class GridState:
    """
    State for accumulating 3D force fields and converting to densities.

    Parameters
    ----------
    trajectory : Trajectory
        Trajectory-state object providing `box_x`, `box_y`, `box_z`, and `units`.
    density_type : {'number', 'charge', 'polarisation'}
        Type of density to be constructed (controls the estimator weighting).
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
        self.beta = trajectory.beta
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
        if dty not in SelectionState.VALID_DENSITY_TYPES:
            raise ValueError(
                f"density_type must be one of {SelectionState.VALID_DENSITY_TYPES}, got {density_type!r}"
            )
        self.density_type = dty

        # Progress flag
        self.grid_progress = "Generated"

    def _process_frame(
        self,
        positions: np.ndarray,
        forces: np.ndarray,
        weight: float | np.ndarray = 1.0,
        kernel: str = "triangular",
    ) -> None:
        """
        Deposit a single set of positions/forces to the grid.

        Parameters
        ----------
        positions : (N, 3) np.ndarray
            Positions in Cartesian coordinates.
        forces : (N, 3) np.ndarray
            Forces corresponding to `positions`.
        weight : float or np.ndarray, optional
            Scalar or per-particle weight (number/charge/polarisation projection).
        kernel : {'triangular', 'box'}
            Assignment kernel.
        """
        self.count += 1

        # Bring positions to the primary image (periodic remainder)
        homeX = np.remainder(positions[:, 0], self.box_x)
        homeY = np.remainder(positions[:, 1], self.box_y)
        homeZ = np.remainder(positions[:, 2], self.box_z)

        # Component forces
        fox = forces[:, 0]
        foy = forces[:, 1]
        foz = forces[:, 2]

        # Map to voxel indices (np.digitize returns 1..len(bins)-1)
        x = np.digitize(homeX, self.binsx)
        y = np.digitize(homeY, self.binsy)
        z = np.digitize(homeZ, self.binsz)

        if kernel.lower() == "triangular":
            _triangular_allocation(
                self.forceX, self.forceY, self.forceZ, self.counter,
                x, y, z, homeX, homeY, homeZ,
                fox, foy, foz, weight,
                self.lx, self.ly, self.lz,
                self.nbinsx, self.nbinsy, self.nbinsz,
            )
        elif kernel.lower() == "box":
            _box_allocation(
                self.forceX, self.forceY, self.forceZ, self.counter,
                x - 1, y - 1, z - 1,
                fox, foy, foz, weight,
            )
        else:
            raise ValueError(f"Unsupported kernel: {kernel!r}")

    def deposit_to_grid(
        self,
        positions: np.ndarray | list[np.ndarray],
        forces: np.ndarray | list[np.ndarray],
        weights: float | np.ndarray | list[float | np.ndarray],
        kernel: str = "triangular",
    ) -> None:
        """
        Deposit positions/forces to the grid using weights.

        Parameters
        ----------
        positions : np.ndarray or list of np.ndarray
            Deposit positions. Single array for single species or rigid COM.
            List of arrays for multi-species non-rigid.
        forces : np.ndarray or list of np.ndarray
            Deposit forces. Same structure as positions.
        weights : float, np.ndarray, or list of np.ndarray
            Weight factor. 1.0 for number density, charges for charge density,
            dipole projection for polarisation.
        kernel : {'triangular', 'box'}
            Deposition kernel (default: 'triangular').
        """
        if isinstance(positions, list):
            if not isinstance(weights, list):
                weights = [weights] * len(positions)
            for pos, frc, wgt in zip(positions, forces, weights):
                self._process_frame(pos, frc, weight=wgt, kernel=kernel)  # type: ignore[arg-type]
        else:
            self._process_frame(positions, forces, weight=weights, kernel=kernel)  # type: ignore[arg-type]

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

        # Validate centre_location
        if not isinstance(centre_location, (bool, int)):
            raise ValueError("centre_location must be True (COM) or int (specific atom index).")

        # Validate polarisation constraints
        if self.density_type == "polarisation":
            if isinstance(atom_names, str) and len(atom_names.replace(',', ' ').split()) == 1:
                raise ValueError("A single atom does not have a polarisation density; specify a rigid molecule.")
            if not rigid:
                raise ValueError("Polarisation densities are only implemented for rigid molecules.")

        # Build selection wrapper with density configuration
        self.selection_state = SelectionState(
            trajectory,
            atom_names=atom_names,
            centre_location=centre_location,
            rigid=rigid,
            density_type=self.density_type,
            polarisation_axis=polarisation_axis,
        )

        # Process frames using unified approach
        for positions, forces in tqdm(trajectory.iter_frames(start, stop, period), total=len(self.to_run)):
            deposit_positions = self.selection_state.get_positions(positions)
            deposit_forces = self.selection_state.get_forces(forces)
            weights = self.selection_state.get_weights(positions)
            self.deposit_to_grid(deposit_positions, deposit_forces, weights, kernel=self.kernel)  # type: ignore[arg-type]

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

        # delta_rho(k): i * beta / k^2 * (F.k); handle k^2=0 via errstate; enforce delta_rho(0)=0
        with np.errstate(divide="ignore", invalid="ignore"):
            self.del_rho_k = (
                complex(0, 1)
                * self.beta / self.get_ksquared()
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
        if hasattr(self, "selection_state") and hasattr(self.selection_state, "indices") and self.selection_state.indices is not None:
            try:
                del atoms[np.array(self.selection_state.indices)]
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

        # TODO: Future refactor - integrate uncertainty/statistics accumulation into
        # base GridState, eliminating need for dynamic attributes here.
        grid_state_lambda = copy.deepcopy(self)
        if sections is None:
            sections = trajectory.frames

        # Baseline expectation from full accumulation
        grid_state_lambda.get_real_density()
        grid_state_lambda.expected_rho = np.copy(grid_state_lambda.rho)  # type: ignore[attr-defined]
        grid_state_lambda.expected_particle_density = np.copy(grid_state_lambda.particle_density)  # type: ignore[attr-defined]
        grid_state_lambda.delta = grid_state_lambda.expected_rho - grid_state_lambda.expected_particle_density  # type: ignore[attr-defined]

        # Covariance/variance accumulators
        grid_state_lambda.cov_buffer_particle = np.zeros((grid_state_lambda.nbinsx, grid_state_lambda.nbinsy, grid_state_lambda.nbinsz))  # type: ignore[attr-defined]
        grid_state_lambda.cov_buffer_force = np.zeros((grid_state_lambda.nbinsx, grid_state_lambda.nbinsy, grid_state_lambda.nbinsz))  # type: ignore[attr-defined]
        grid_state_lambda.var_buffer = np.zeros((grid_state_lambda.nbinsx, grid_state_lambda.nbinsy, grid_state_lambda.nbinsz))  # type: ignore[attr-defined]

        # Interleaved accumulation across sections
        for k in tqdm(range(sections)):
            # Reset accumulators for this section
            grid_state_lambda.forceX *= 0
            grid_state_lambda.forceY *= 0
            grid_state_lambda.forceZ *= 0
            grid_state_lambda.particle_density *= 0
            grid_state_lambda.counter *= 0
            grid_state_lambda.del_rho_k *= 0
            grid_state_lambda.del_rho_n *= 0
            grid_state_lambda.rho *= 0
            grid_state_lambda.count *= 0

            # Frame indices for this section (interleaved sampling)
            frame_indices = np.array(grid_state_lambda.to_run)[
                np.arange(k, sections * (len(grid_state_lambda.to_run) // sections), sections)
            ]
            for frame_idx in frame_indices:
                positions, forces = trajectory.get_frame(frame_idx)
                deposit_positions = grid_state_lambda.selection_state.get_positions(positions)
                deposit_forces = grid_state_lambda.selection_state.get_forces(forces)
                weights = grid_state_lambda.selection_state.get_weights(positions)
                grid_state_lambda.deposit_to_grid(deposit_positions, deposit_forces, weights, kernel=grid_state_lambda.kernel)  # type: ignore[arg-type]

            # Compute densities for this section and accumulate statistics
            grid_state_lambda.get_real_density()
            delta_cur = grid_state_lambda.rho - grid_state_lambda.particle_density
            grid_state_lambda.var_buffer += (delta_cur - grid_state_lambda.delta) ** 2  # type: ignore[attr-defined]
            grid_state_lambda.cov_buffer_force += (delta_cur - grid_state_lambda.delta) * (grid_state_lambda.rho - grid_state_lambda.expected_rho)  # type: ignore[attr-defined]
            grid_state_lambda.cov_buffer_particle += (delta_cur - grid_state_lambda.delta) * (  # type: ignore[attr-defined]
                grid_state_lambda.particle_density - grid_state_lambda.expected_particle_density  # type: ignore[attr-defined]
            )

        # lambda = 1 - cov(force)/var(delta); optimal density = (1-lambda)*count + lambda*force
        grid_state_lambda.combination = 1.0 - (grid_state_lambda.cov_buffer_force / grid_state_lambda.var_buffer)  # type: ignore[attr-defined]
        grid_state_lambda.optimal_density = (  # type: ignore[attr-defined]
            (1.0 - grid_state_lambda.combination) * grid_state_lambda.expected_particle_density +  # type: ignore[attr-defined]
            grid_state_lambda.combination * grid_state_lambda.expected_rho  # type: ignore[attr-defined]
        )
        grid_state_lambda.grid_progress = "Lambda"
        return grid_state_lambda
