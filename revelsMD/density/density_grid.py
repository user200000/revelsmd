"""DensityGrid class for accumulating 3D force fields and converting to densities."""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
from tqdm import tqdm
from ase import Atoms
from ase.io.cube import write_cube
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from revelsMD.trajectories._base import Trajectory
from revelsMD.density.constants import validate_density_type
from revelsMD.density.selection import Selection
from revelsMD.density.grid_helpers import get_backend_functions as _get_grid_backend_functions
from revelsMD.statistics import compute_lambda_weights, combine_estimators

# Module-level backend functions (loaded once at import)
_triangular_allocation, _box_allocation = _get_grid_backend_functions()


class DensityGrid:
    """
    State for accumulating 3D force fields and converting to densities.

    Parameters
    ----------
    trajectory : Trajectory
        Trajectory-state object providing `box_x`, `box_y`, `box_z`, and `units`.
    density_type : {'number', 'charge', 'polarisation'}
        Type of density to be constructed (controls the estimator weighting).
    nbins : int or tuple of int
        Number of voxels per dimension. Either a single int for uniform binning
        or a tuple (nbinsx, nbinsy, nbinsz) for per-axis bin counts.

    Attributes
    ----------
    force_x, force_y, force_z : np.ndarray
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
    progress : {'Generated', 'Allocated', 'Lambda'}
        Simple state flag used by getters and lambda estimator.
    """

    def __init__(
        self,
        trajectory: Trajectory,
        density_type: str,
        nbins: int | tuple[int, int, int] = 100,
    ):
        # Resolve per-dimension bin counts
        if isinstance(nbins, tuple):
            nbinsx, nbinsy, nbinsz = nbins
        else:
            nbinsx = nbinsy = nbinsz = nbins
        if min(nbinsx, nbinsy, nbinsz) <= 0:
            raise ValueError("nbins values must be positive integers.")

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
        self.force_x = np.zeros((nbinsx, nbinsy, nbinsz), dtype=float)
        self.force_y = np.zeros((nbinsx, nbinsy, nbinsz), dtype=float)
        self.force_z = np.zeros((nbinsx, nbinsy, nbinsz), dtype=float)
        self.counter = np.zeros((nbinsx, nbinsy, nbinsz), dtype=float)

        # Density selection
        self.density_type = validate_density_type(density_type)

        # Progress flag
        self.progress = "Generated"

        # Density results (populated by get_real_density or get_lambda)
        self._rho_count = np.zeros((nbinsx, nbinsy, nbinsz), dtype=float)
        self._rho_force = np.zeros((nbinsx, nbinsy, nbinsz), dtype=float)
        self._rho_lambda: np.ndarray | None = None
        self._lambda_weights: np.ndarray | None = None

    @property
    def rho_count(self) -> np.ndarray:
        """Counting-based density (zeros until get_real_density or get_lambda is called)."""
        return self._rho_count

    @property
    def rho_force(self) -> np.ndarray:
        """Force-based density via FFT (zeros until get_real_density or get_lambda is called)."""
        return self._rho_force

    @property
    def rho_lambda(self) -> np.ndarray | None:
        """Variance-minimised density (available after get_lambda)."""
        return self._rho_lambda

    @property
    def lambda_weights(self) -> np.ndarray | None:
        """Per-voxel lambda weights (available after get_lambda)."""
        return self._lambda_weights

    @property
    def optimal_density(self) -> np.ndarray | None:
        """Deprecated: use rho_lambda instead."""
        warnings.warn(
            "optimal_density is deprecated, use rho_lambda instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._rho_lambda

    @property
    def combination(self) -> np.ndarray | None:
        """Deprecated: use lambda_weights instead."""
        warnings.warn(
            "combination is deprecated, use lambda_weights instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._lambda_weights

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
                self.force_x, self.force_y, self.force_z, self.counter,
                x, y, z, homeX, homeY, homeZ,
                fox, foy, foz, weight,
                self.lx, self.ly, self.lz,
                self.nbinsx, self.nbinsy, self.nbinsz,
            )
        elif kernel.lower() == "box":
            _box_allocation(
                self.force_x, self.force_y, self.force_z, self.counter,
                x - 1, y - 1, z - 1,
                fox, foy, foz, weight,
            )
        else:
            raise ValueError(f"Unsupported kernel: {kernel!r}")

    def deposit(
        self,
        positions: np.ndarray | list[np.ndarray],
        forces: np.ndarray | list[np.ndarray],
        weights: float | np.ndarray | list[np.ndarray],
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
            if not isinstance(forces, list):
                raise TypeError("positions and forces must both be lists or both be arrays")
            weight_seq: Sequence[float | np.ndarray]
            if isinstance(weights, list):
                weight_seq = weights
            else:
                weight_seq = [weights] * len(positions)
            for pos, frc, wgt in zip(positions, forces, weight_seq):
                self._process_frame(pos, frc, weight=wgt, kernel=kernel)
        else:
            if isinstance(forces, list):
                raise TypeError("positions and forces must both be lists or both be arrays")
            if isinstance(weights, list):
                raise TypeError("weights cannot be a list when positions is a single array")
            self._process_frame(positions, forces, weight=weights, kernel=kernel)

    def accumulate(
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
        self._selection = Selection(
            trajectory,
            atom_names=atom_names,
            centre_location=centre_location,
            rigid=rigid,
            density_type=self.density_type,
            polarisation_axis=polarisation_axis,
        )

        # Process frames using unified approach
        for positions, forces in tqdm(trajectory.iter_frames(start, stop, period), total=len(self.to_run)):
            deposit_positions = self._selection.get_positions(positions)
            deposit_forces = self._selection.get_forces(forces)
            weights = self._selection.get_weights(positions)
            self.deposit(deposit_positions, deposit_forces, weights, kernel=self.kernel)

        self.frames_processed = self.to_run
        self.progress = "Allocated"

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
        Sets `self.del_rho_k`, `self.del_rho_n`, `self.rho_count`, `self.rho_force`.
        """
        if self.progress == "Generated":
            raise RuntimeError("Run accumulate() before computing densities.")

        # Normalize by number of frames and voxel volume before FFT
        with np.errstate(divide="ignore", invalid="ignore"):
            force_x = np.fft.fftn(self.force_x / self.count / self.voxel_volume)
            force_y = np.fft.fftn(self.force_y / self.count / self.voxel_volume)
            force_z = np.fft.fftn(self.force_z / self.count / self.voxel_volume)

        # k-vectors per dimension
        xrep, yrep, zrep = self.get_kvectors()

        # Multiply by k components (component-wise dot in spectral space)
        for n in range(len(xrep)):
            force_x[n, :, :] = xrep[n] * force_x[n, :, :]
        for m in range(len(yrep)):
            force_y[:, m, :] = yrep[m] * force_y[:, m, :]
        for l in range(len(zrep)):
            force_z[:, :, l] = zrep[l] * force_z[:, :, l]

        # delta_rho(k): i * beta / k^2 * (F.k); handle k^2=0 via errstate; enforce delta_rho(0)=0
        with np.errstate(divide="ignore", invalid="ignore"):
            self.del_rho_k = (
                complex(0, 1)
                * self.beta / self.get_ksquared()
                * (force_x + force_y + force_z)
            )
        self.del_rho_k[0, 0, 0] = 0.0

        # Back to real space (density perturbation), add mean counting density
        del_rho_n = np.fft.ifftn(self.del_rho_k)
        self.del_rho_n = -1.0 * np.real(del_rho_n)

        # Conventional counting density (averaged)
        self._compute_rho_count()
        self._rho_force = self.del_rho_n + np.mean(self._rho_count)

    def _compute_rho_count(self) -> None:
        """
        Compute conventional counting-based density from `self.counter`.

        Side Effects
        ------------
        Sets `self._rho_count`.
        """
        if self.progress == "Generated":
            raise RuntimeError("Run accumulate() before computing densities.")
        with np.errstate(divide="ignore", invalid="ignore"):
            self._rho_count = self.counter / self.voxel_volume / self.count

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
        if hasattr(self, "_selection") and hasattr(self._selection, "indices") and self._selection.indices is not None:
            try:
                del atoms[np.array(self._selection.indices)]
            except Exception:
                # If selection is a list of arrays (multi-species), do nothing silently
                pass

        with open(filename, "w") as f:
            write_cube(f, atoms, data=grid)

    def get_lambda(self, trajectory: Trajectory, sections: int | None = None) -> None:
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
        None
            Results stored as attributes:
            - rho_lambda: variance-minimised density
            - lambda_weights: per-voxel combination weights

        Raises
        ------
        RuntimeError
            If called before `accumulate`.
        ValueError
            If called on a grid already produced by `get_lambda`.

        Notes
        -----
        After this method completes, the internal accumulators (force_x/y/z, counter)
        will contain only the last section's data, not the full trajectory. This means
        rho_force and rho_count will no longer reflect the full accumulation — only
        rho_lambda should be used after calling this method.
        """
        if self.progress == "Generated":
            raise RuntimeError("Run accumulate() before estimating lambda.")
        if self.progress == "Lambda":
            raise ValueError("This grid was already produced by get_lambda; re-run upstream to refresh.")

        if sections is None:
            sections = trajectory.frames

        # Baseline expectation from full accumulation
        self.get_real_density()
        expected_rho_force = np.copy(self._rho_force)
        expected_rho_count = np.copy(self._rho_count)
        delta = expected_rho_force - expected_rho_count

        # Covariance/variance accumulators (local, not stored)
        cov_buffer_force = np.zeros((self.nbinsx, self.nbinsy, self.nbinsz))
        var_buffer = np.zeros((self.nbinsx, self.nbinsy, self.nbinsz))

        # Interleaved accumulation across sections
        for k in tqdm(range(sections)):
            # Reset accumulators for this section
            self.force_x.fill(0)
            self.force_y.fill(0)
            self.force_z.fill(0)
            self._rho_count.fill(0)
            self.counter.fill(0)
            self.del_rho_k.fill(0)
            self.del_rho_n.fill(0)
            self._rho_force.fill(0)
            self.count = 0

            # Frame indices for this section (interleaved sampling)
            frame_indices = np.array(self.to_run)[
                np.arange(k, sections * (len(self.to_run) // sections), sections)
            ]
            for frame_idx in frame_indices:
                positions, forces = trajectory.get_frame(frame_idx)
                deposit_positions = self._selection.get_positions(positions)
                deposit_forces = self._selection.get_forces(forces)
                weights = self._selection.get_weights(positions)
                self.deposit(deposit_positions, deposit_forces, weights, kernel=self.kernel)

            # Compute densities for this section and accumulate statistics
            self.get_real_density()
            delta_cur = self._rho_force - self._rho_count
            var_buffer += (delta_cur - delta) ** 2
            cov_buffer_force += (delta_cur - delta) * (self._rho_force - expected_rho_force)

        # lambda = 1 - cov(force)/var(delta); optimal density = (1-lambda)*count + lambda*force
        lambda_raw = compute_lambda_weights(var_buffer, cov_buffer_force)
        self._lambda_weights = 1.0 - lambda_raw
        self._rho_lambda = combine_estimators(
            expected_rho_count,
            expected_rho_force,
            self._lambda_weights,
        )
        self.progress = "Lambda"


def compute_density(
    trajectory: Trajectory,
    atom_names: str | list[str],
    rigid: bool = False,
    centre_location: bool | int = True,
    density_type: str = "number",
    nbins: int | tuple[int, int, int] = 100,
    kernel: str = "triangular",
    polarisation_axis: int = 0,
    start: int = 0,
    stop: int | None = None,
    period: int = 1,
    integration: str = "standard",
    sections: int | None = None,
) -> DensityGrid:
    """
    Compute density from trajectory with a single function call.

    This is a convenience wrapper that creates a DensityGrid, accumulates
    force data from the trajectory, and computes the real-space density.

    Parameters
    ----------
    trajectory : Trajectory
        Trajectory-state object providing positions, forces, and box dimensions.
    atom_names : str or list of str
        Atom type(s) to include in the density calculation.
    rigid : bool, optional
        Whether to treat multi-atom selections as rigid molecules (default: False).
    centre_location : bool or int, optional
        For rigid molecules, where to locate the density:
        True = centre of mass, int = index of atom in atom_names list.
    density_type : {'number', 'charge', 'polarisation'}, optional
        Type of density to compute (default: 'number').
    nbins : int or tuple of int, optional
        Number of grid voxels. Either a single int for uniform binning
        or a tuple (nbinsx, nbinsy, nbinsz) for per-axis control (default: 100).
    kernel : {'triangular', 'box'}, optional
        Deposition kernel for force allocation (default: 'triangular').
    polarisation_axis : int, optional
        Axis for polarisation density projection, 0=x, 1=y, 2=z (default: 0).
    start : int, optional
        First frame to process (default: 0).
    stop : int or None, optional
        Last frame to process, None for all frames (default: None).
    period : int, optional
        Frame stride (default: 1).
    integration : {'standard', 'lambda'}, optional
        Integration method (default: 'standard'). Use 'lambda' for
        variance-minimised combination of counting and force densities.
    sections : int or None, optional
        Number of interleaved frame-subsets for lambda estimation.
        Only used when integration='lambda'. If None, defaults to the
        number of frames in the trajectory.

    Returns
    -------
    DensityGrid
        Grid with computed density. For integration='standard', the result
        is available as `rho_force`. For integration='lambda', the
        variance-minimised result is available as `rho_lambda`.

    Examples
    --------
    >>> from revelsMD.density import compute_density
    >>> grid = compute_density(trajectory, 'O', nbins=50)
    >>> density = grid.rho_force

    >>> # With variance-minimised lambda integration
    >>> grid = compute_density(trajectory, 'O', integration='lambda', sections=10)
    >>> density = grid.rho_lambda
    """
    grid = DensityGrid(trajectory, density_type, nbins=nbins)
    grid.accumulate(
        trajectory,
        atom_names=atom_names,
        rigid=rigid,
        centre_location=centre_location,
        kernel=kernel,
        polarisation_axis=polarisation_axis,
        start=start,
        stop=stop,
        period=period,
    )
    grid.get_real_density()

    if integration == "lambda":
        grid.get_lambda(trajectory, sections=sections)
    elif integration != "standard":
        raise ValueError(f"integration must be 'standard' or 'lambda', got '{integration}'")

    return grid
