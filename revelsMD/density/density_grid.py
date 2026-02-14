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
from revelsMD.cell import (
    cartesian_to_fractional,
    cells_are_compatible,
    wrap_fractional,
)
from revelsMD.density.constants import validate_density_type
from revelsMD.density.selection import Selection
from revelsMD.density.grid_helpers import get_backend_functions as _get_grid_backend_functions
from revelsMD.statistics import (
    WelfordAccumulator3D,
    combine_estimators,
    compute_lambda_weights,
)

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

        # Cell geometry
        self.cell_matrix = np.array(trajectory.cell_matrix, dtype=np.float64)
        self.cell_inverse = np.linalg.inv(self.cell_matrix)

        # Voxel volume from cell determinant
        self.voxel_volume = float(
            abs(np.linalg.det(self.cell_matrix)) / (nbinsx * nbinsy * nbinsz)
        )
        self.nbinsx, self.nbinsy, self.nbinsz = nbinsx, nbinsy, nbinsz

        # Fractional bin edges and voxel sizes (work for any cell geometry)
        self.lx = 1.0 / nbinsx
        self.ly = 1.0 / nbinsy
        self.lz = 1.0 / nbinsz

        self.binsx = np.linspace(0, 1, nbinsx + 1)
        self.binsy = np.linspace(0, 1, nbinsy + 1)
        self.binsz = np.linspace(0, 1, nbinsz + 1)

        # Precompute full 3D k-vectors
        self._k_vectors, self._ksquared = self._build_kvectors_3d()

        self.beta = trajectory.beta
        self.count = 0
        self.units = trajectory.units

        # Accumulators
        self.force_x = np.zeros((nbinsx, nbinsy, nbinsz), dtype=float)
        self.force_y = np.zeros((nbinsx, nbinsy, nbinsz), dtype=float)
        self.force_z = np.zeros((nbinsx, nbinsy, nbinsz), dtype=float)
        self.counter = np.zeros((nbinsx, nbinsy, nbinsz), dtype=float)

        # Density selection
        self.density_type = validate_density_type(density_type)

        # Bookkeeping for multi-trajectory accumulation
        self.frames_processed = 0

        # Density results (computed on demand when properties are accessed)
        self._rho_count: np.ndarray | None = None
        self._rho_force: np.ndarray | None = None
        self._rho_lambda: np.ndarray | None = None
        self._lambda_weights: np.ndarray | None = None

        # Lambda statistics accumulator (populated if compute_lambda=True in accumulate)
        self._welford: WelfordAccumulator3D | None = None

    @property
    def rho_count(self) -> np.ndarray | None:
        """Counting-based density. Computed on first access after accumulate()."""
        if self.count == 0:
            return None  # No data accumulated yet
        if self._rho_count is None:
            self._compute_real_densities()
        return self._rho_count

    @property
    def rho_force(self) -> np.ndarray | None:
        """Force-based density via FFT. Computed on first access after accumulate()."""
        if self.count == 0:
            return None  # No data accumulated yet
        if self._rho_force is None:
            self._compute_real_densities()
        return self._rho_force

    @property
    def rho_lambda(self) -> np.ndarray | None:
        """Variance-minimised density (available after get_lambda or compute_lambda)."""
        if self._rho_lambda is None and self._welford is not None:
            self._finalise_lambda()
        return self._rho_lambda

    @property
    def lambda_weights(self) -> np.ndarray | None:
        """Per-voxel lambda weights (available after get_lambda or compute_lambda)."""
        if self._lambda_weights is None and self._welford is not None:
            self._finalise_lambda()
        return self._lambda_weights

    @property
    def optimal_density(self) -> np.ndarray | None:
        """Deprecated: use rho_lambda instead."""
        warnings.warn(
            "optimal_density is deprecated, use rho_lambda instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.rho_lambda

    @property
    def combination(self) -> np.ndarray | None:
        """Deprecated: use lambda_weights instead."""
        warnings.warn(
            "combination is deprecated, use lambda_weights instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.lambda_weights

    def rho_hybrid(self, threshold: float) -> np.ndarray:
        """Threshold-switched density combining force and count estimators.

        Uses the force-based density where the counting density is at or
        above the threshold, and the counting density elsewhere. This
        eliminates spurious negative density artefacts from the force
        estimator in poorly sampled regions while preserving the higher
        resolution of the force estimator in well-sampled regions.

        Parameters
        ----------
        threshold : float
            Density threshold (in the same units as ``rho_count``).
            Voxels with ``rho_count >= threshold`` use the force estimate;
            voxels below the threshold use the counting estimate.

        Returns
        -------
        np.ndarray
            Hybrid density field, same shape as ``rho_count``.

        Raises
        ------
        RuntimeError
            If no data has been accumulated yet.
        ValueError
            If threshold is negative.
        """
        if threshold < 0:
            raise ValueError(
                f"threshold must be non-negative, got {threshold}"
            )

        rho_c = self.rho_count
        rho_f = self.rho_force

        if rho_c is None or rho_f is None:
            raise RuntimeError(
                "No density data available. Run accumulate() first."
            )

        return np.where(rho_c >= threshold, rho_f, rho_c)

    def _wrap_to_grid(self, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Wrap Cartesian positions to fractional grid coordinates in [0, 1).

        Returns (homeX, homeY, homeZ) as fractional coordinates.
        The bin edges and voxel sizes (lx/ly/lz, binsx/y/z) are in the same
        fractional coordinate system.
        """
        frac = cartesian_to_fractional(positions, self.cell_inverse)
        frac = wrap_fractional(frac)
        homeX, homeY, homeZ = frac[:, 0], frac[:, 1], frac[:, 2]
        return homeX, homeY, homeZ

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

        # Bring positions to the primary image
        homeX, homeY, homeZ = self._wrap_to_grid(positions)

        # Component forces (always Cartesian)
        fox = forces[:, 0]
        foy = forces[:, 1]
        foz = forces[:, 2]

        # Map to voxel indices (np.digitize returns 1..len(bins)-1)
        x = np.clip(np.digitize(homeX, self.binsx), 1, self.nbinsx)
        y = np.clip(np.digitize(homeY, self.binsy), 1, self.nbinsy)
        z = np.clip(np.digitize(homeZ, self.binsz), 1, self.nbinsz)

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
        compute_lambda: bool = False,
        sections: int | None = None,
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
        compute_lambda : bool, optional
            If True, collect variance statistics for lambda estimation during
            accumulation using Welford's algorithm. The variance-minimised density
            will be available via grid.rho_lambda. Default is False (faster, no
            lambda overhead).
        sections : int or None, optional
            Number of sections for lambda estimation. Only used when
            compute_lambda=True. If None, defaults to one section per frame
            (matching the original get_lambda behaviour). At least 2 sections
            are required across all ``accumulate()`` calls for variance
            estimation; accessing ``rho_lambda`` with fewer raises ValueError.

        Notes
        -----
        When ``compute_lambda=True``, variance statistics accumulate across
        multiple ``accumulate()`` calls, enabling lambda estimation from
        multiple trajectories.

        Calling ``accumulate(..., compute_lambda=False)`` clears any existing
        lambda statistics. After this, ``rho_lambda`` will return ``None``
        until ``compute_lambda=True`` is used again.

        Raises
        ------
        ValueError
            On invalid frame selection, unsupported density/kernel combinations,
            malformed inputs for rigid/centre options, or if ``rho_lambda`` is
            accessed with fewer than 2 accumulated sections (variance estimation
            requires at least 2 data points).
        """
        # --- Validate cell compatibility ---
        if not cells_are_compatible(trajectory.cell_matrix, self.cell_matrix):
            raise ValueError(
                "Trajectory cell does not match DensityGrid cell."
            )

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

        # Invalidate any previous derived state — accumulator data is changing,
        # so cached results would be stale.
        self._invalidate_derived_state()
        if not compute_lambda and self._welford is not None:
            warnings.warn(
                "Calling accumulate() with compute_lambda=False discards existing "
                "lambda statistics. Use compute_lambda=True to preserve them.",
                UserWarning,
                stacklevel=2,
            )
            self._welford = None

        if not compute_lambda:
            # Simple accumulation (existing behaviour, fast path)
            self._accumulate_simple(trajectory, start, stop, period)
        else:
            # Sectioned accumulation with lambda statistics
            if sections is not None and sections <= 0:
                raise ValueError("sections must be a positive integer")
            # Default to one section per frame (matches original get_lambda behaviour)
            effective_sections = sections if sections is not None else len(self.to_run)
            if effective_sections > len(self.to_run):
                raise ValueError(
                    f"sections ({effective_sections}) exceeds the number of frames "
                    f"to process ({len(self.to_run)})"
                )
            self._accumulate_with_sections(trajectory, effective_sections)

        self.frames_processed += len(self.to_run)

    def _invalidate_derived_state(self) -> None:
        """Clear cached densities and lambda weights.

        Called when accumulator data changes, making any previously computed
        densities stale.
        """
        self._rho_force = None
        self._rho_count = None
        self._rho_lambda = None
        self._lambda_weights = None

    def _accumulate_simple(
        self,
        trajectory: Trajectory,
        start: int,
        stop: int | None,
        period: int,
    ) -> None:
        """Accumulate without collecting lambda statistics (fast path)."""
        for positions, forces in tqdm(
            trajectory.iter_frames(start, stop, period), total=len(self.to_run)
        ):
            deposit_positions = self._selection.get_positions(positions)
            deposit_forces = self._selection.get_forces(forces)
            weights = self._selection.get_weights(positions)
            self.deposit(deposit_positions, deposit_forces, weights, kernel=self.kernel)

    def _accumulate_with_sections(
        self,
        trajectory: Trajectory,
        sections: int,
    ) -> None:
        """Accumulate while collecting lambda statistics via Welford's algorithm.

        Uses self.to_run (set by accumulate()) for frame indices.
        """
        # Initialise Welford accumulator if first call with compute_lambda
        if self._welford is None:
            self._welford = WelfordAccumulator3D(
                shape=(self.nbinsx, self.nbinsy, self.nbinsz)
            )

        # self.to_run is a range, which supports slicing without materialising
        # all indices into a list. This avoids memory overhead for large trajectories.
        frame_indices = self.to_run

        # Preallocate section buffers once, reused each iteration via .fill(0).
        # This avoids O(sections) allocations which can cause significant memory
        # churn for large grids (e.g., 100^3 grid = 32MB per allocation cycle).
        section_force_x = np.zeros_like(self.force_x)
        section_force_y = np.zeros_like(self.force_y)
        section_force_z = np.zeros_like(self.force_z)
        section_counter = np.zeros_like(self.counter)

        # Process each section with interleaved sampling
        for k in tqdm(range(sections), desc="Accumulating sections"):
            # Compute interleaved frame indices for this section
            # E.g., with 10 frames and 2 sections: section 0 gets [0,2,4,6,8], section 1 gets [1,3,5,7,9]
            section_frame_indices = frame_indices[k::sections]

            if len(section_frame_indices) == 0:
                continue

            # Reset section accumulators
            section_force_x.fill(0)
            section_force_y.fill(0)
            section_force_z.fill(0)
            section_counter.fill(0)
            section_count = 0

            # Process frames in this section
            for frame_idx in section_frame_indices:
                positions, forces = trajectory.get_frame(frame_idx)
                deposit_positions = self._selection.get_positions(positions)
                deposit_forces = self._selection.get_forces(forces)
                weights = self._selection.get_weights(positions)

                # Deposit to section accumulators
                self._deposit_to_arrays(
                    section_force_x, section_force_y, section_force_z, section_counter,
                    deposit_positions, deposit_forces, weights, self.kernel
                )

                # Keep section_count semantics consistent with _process_frame/deposit:
                # increment once per deposited array, not once per frame.
                if isinstance(deposit_positions, list):
                    section_count += len(deposit_positions)
                else:
                    section_count += 1

            # Add section data to main accumulators (for rho_force/rho_count)
            self.force_x += section_force_x
            self.force_y += section_force_y
            self.force_z += section_force_z
            self.counter += section_counter
            self.count += section_count

            # Compute section densities for Welford update
            section_rho_force, section_rho_count = self._compute_densities_from_arrays(
                section_force_x, section_force_y, section_force_z,
                section_counter, section_count
            )
            section_delta = section_rho_force - section_rho_count

            # Update Welford statistics
            self._welford.update(section_delta, section_rho_force)

    def _deposit_to_arrays(
        self,
        force_x: np.ndarray,
        force_y: np.ndarray,
        force_z: np.ndarray,
        counter: np.ndarray,
        positions: np.ndarray | list[np.ndarray],
        forces: np.ndarray | list[np.ndarray],
        weights: float | np.ndarray | list[np.ndarray],
        kernel: str,
    ) -> None:
        """Deposit to provided arrays (not self's accumulators)."""
        if isinstance(positions, list):
            if not isinstance(forces, list):
                raise TypeError("positions and forces must both be lists or both be arrays")
            if isinstance(weights, list):
                for pos, frc, wgt in zip(positions, forces, weights):
                    self._deposit_single_to_arrays(
                        force_x, force_y, force_z, counter, pos, frc, wgt, kernel
                    )
            else:
                scalar_weight: float | np.ndarray = weights
                for pos, frc in zip(positions, forces):
                    self._deposit_single_to_arrays(
                        force_x, force_y, force_z, counter, pos, frc, scalar_weight, kernel
                    )
        else:
            if isinstance(forces, list):
                raise TypeError("positions and forces must both be lists or both be arrays")
            if isinstance(weights, list):
                raise TypeError("weights cannot be a list when positions is a single array")
            self._deposit_single_to_arrays(
                force_x, force_y, force_z, counter, positions, forces, weights, kernel
            )

    def _deposit_single_to_arrays(
        self,
        force_x: np.ndarray,
        force_y: np.ndarray,
        force_z: np.ndarray,
        counter: np.ndarray,
        positions: np.ndarray,
        forces: np.ndarray,
        weight: float | np.ndarray,
        kernel: str,
    ) -> None:
        """Deposit a single set of positions/forces to provided arrays."""
        # Bring positions to the primary image
        homeX, homeY, homeZ = self._wrap_to_grid(positions)

        # Component forces (always Cartesian)
        fox = forces[:, 0]
        foy = forces[:, 1]
        foz = forces[:, 2]

        # Map to voxel indices
        x = np.clip(np.digitize(homeX, self.binsx), 1, self.nbinsx)
        y = np.clip(np.digitize(homeY, self.binsy), 1, self.nbinsy)
        z = np.clip(np.digitize(homeZ, self.binsz), 1, self.nbinsz)

        if kernel.lower() == "triangular":
            _triangular_allocation(
                force_x, force_y, force_z, counter,
                x, y, z, homeX, homeY, homeZ,
                fox, foy, foz, weight,
                self.lx, self.ly, self.lz,
                self.nbinsx, self.nbinsy, self.nbinsz,
            )
        elif kernel.lower() == "box":
            _box_allocation(
                force_x, force_y, force_z, counter,
                x - 1, y - 1, z - 1,
                fox, foy, foz, weight,
            )
        else:
            raise ValueError(f"Unsupported kernel: {kernel!r}")

    def _fft_force_to_density(
        self,
        force_x: np.ndarray,
        force_y: np.ndarray,
        force_z: np.ndarray,
        counter: np.ndarray,
        count: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert force/counter arrays to densities via FFT.

        This is the core density calculation implementing (Borgis et al.,
        Mol. Phys. 111, 3486-3492 (2013)):
        delta_rho(k) = i / (k_B T k^2) * k . F(k), with delta_rho(k=0) := 0,
        then rho(r) = <rho_count(r)> + F^-1[delta_rho(k)].

        Parameters
        ----------
        force_x, force_y, force_z : ndarray
            Accumulated force components on the grid.
        counter : ndarray
            Accumulated particle counts on the grid.
        count : int
            Total number of deposited samples (for normalisation).

        Returns
        -------
        rho_force : ndarray
            Force-based density.
        rho_count : ndarray
            Counting-based density.
        del_rho_k : ndarray
            Density perturbation in k-space.
        del_rho_n : ndarray
            Density perturbation in real space.
        """
        if count == 0:
            zeros = np.zeros_like(force_x)
            return zeros, zeros, zeros.astype(complex), zeros

        # Counting density
        with np.errstate(divide="ignore", invalid="ignore"):
            rho_count = counter / self.voxel_volume / count

        # FFT of normalised forces
        with np.errstate(divide="ignore", invalid="ignore"):
            fx_fft = np.fft.fftn(force_x / count / self.voxel_volume)
            fy_fft = np.fft.fftn(force_y / count / self.voxel_volume)
            fz_fft = np.fft.fftn(force_z / count / self.voxel_volume)

        # k . F(k) dot product using precomputed 3D k-vectors
        kx = self._k_vectors[..., 0]
        ky = self._k_vectors[..., 1]
        kz = self._k_vectors[..., 2]
        k_dot_F = kx * fx_fft + ky * fy_fft + kz * fz_fft

        ksq = self._ksquared.copy()
        ksq[0, 0, 0] = 1.0  # avoid division by zero
        del_rho_k = complex(0, 1) * self.beta / ksq * k_dot_F

        del_rho_k[0, 0, 0] = 0.0

        # Back to real space
        del_rho_n = -1.0 * np.real(np.fft.ifftn(del_rho_k))
        rho_force = del_rho_n + np.mean(rho_count)

        return rho_force, rho_count, del_rho_k, del_rho_n

    def _compute_densities_from_arrays(
        self,
        force_x: np.ndarray,
        force_y: np.ndarray,
        force_z: np.ndarray,
        counter: np.ndarray,
        count: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute rho_force and rho_count from provided arrays.

        Returns
        -------
        rho_force : ndarray
            Force-based density.
        rho_count : ndarray
            Counting-based density.
        """
        rho_force, rho_count, _, _ = self._fft_force_to_density(
            force_x, force_y, force_z, counter, count
        )
        return rho_force, rho_count

    def _finalise_lambda(self) -> None:
        """Compute final lambda weights and density from Welford statistics."""
        if self._welford is None or not self._welford.has_data:
            return

        if self._rho_lambda is not None:
            return  # Already finalised

        # Compute the expected densities from full accumulation (uses caching)
        expected_rho_force = self.rho_force
        expected_rho_count = self.rho_count

        # These can't be None here - we've accumulated data (count > 0)
        # and the properties trigger computation if needed
        assert expected_rho_force is not None
        assert expected_rho_count is not None

        if self._welford.count < 2:
            raise ValueError(
                f"Cannot compute lambda with fewer than 2 sections (have {self._welford.count}). "
                "Use sections >= 2 or accumulate from additional trajectories."
            )

        # Finalise Welford statistics
        var_buffer, cov_buffer_force = self._welford.finalise()

        # Compute lambda weights: lambda = 1 - Cov(delta, rho_force) / Var(delta)
        lambda_raw = compute_lambda_weights(var_buffer, cov_buffer_force)
        self._lambda_weights = 1.0 - lambda_raw

        # Compute variance-minimised density
        self._rho_lambda = combine_estimators(
            expected_rho_count,
            expected_rho_force,
            self._lambda_weights,
        )

    def get_real_density(self) -> None:
        """
        Convert accumulated force field to real-space density via FFT.

        .. deprecated::
            Densities are now computed on demand. Access ``rho_force`` or
            ``rho_count`` directly instead of calling this method.

        Notes
        -----
        Implements (Borgis et al., *Mol. Phys.* **111**, 3486-3492 (2013)):
        delta_rho(k) = i / (k_B T k^2) * k . F(k), with delta_rho(k=0) := 0,
        then rho(r) = <rho_count(r)> + F^-1[delta_rho(k)].

        Side Effects
        ------------
        Sets `self.del_rho_k`, `self.del_rho_n`, `self.rho_count`, `self.rho_force`.
        """
        warnings.warn(
            "get_real_density() is deprecated and will be removed in a future version. "
            "Access grid.rho_force or grid.rho_count directly; "
            "densities are now computed on demand.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._compute_real_densities()

    def _compute_real_densities(self) -> None:
        """Internal: compute rho_force and rho_count from accumulators via FFT."""
        if self.count == 0:
            raise RuntimeError("Run accumulate() before computing densities.")

        self._rho_force, self._rho_count, self.del_rho_k, self.del_rho_n = (
            self._fft_force_to_density(
                self.force_x, self.force_y, self.force_z, self.counter, self.count
            )
        )

    def get_kvectors(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return the Cartesian k-vectors per dimension.

        For orthorhombic cells these are the standard 1D FFT frequencies.
        For general cells, each 1D array is extracted from the precomputed
        3D k-vector grid along the corresponding axis (with the other
        indices set to zero).

        Returns
        -------
        (kx, ky, kz) : tuple of np.ndarray
            Frequency vectors for x, y, z (units: 1/length).
        """
        kx = self._k_vectors[:, 0, 0, 0]
        ky = self._k_vectors[0, :, 0, 1]
        kz = self._k_vectors[0, 0, :, 2]
        return kx, ky, kz

    def get_ksquared(self) -> np.ndarray:
        """
        Return |k|^2 on the 3D grid.

        Returns
        -------
        np.ndarray
            k^2 array with shape (nbinsx, nbinsy, nbinsz).
        """
        return self._ksquared.copy()

    def _build_kvectors_3d(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Build full 3D k-vector arrays for general (triclinic) cells.

        The k-vector at Miller indices (m1, m2, m3) is:
            k = 2 * pi * inv(M)^T @ [m1, m2, m3]^T
        where M is the cell matrix with rows = lattice vectors.

        Returns
        -------
        k_vectors : np.ndarray, shape (nbinsx, nbinsy, nbinsz, 3)
            Cartesian k-vectors at each reciprocal grid point.
        ksquared : np.ndarray, shape (nbinsx, nbinsy, nbinsz)
            |k|^2 at each reciprocal grid point.
        """
        m1 = np.fft.fftfreq(self.nbinsx, d=1.0 / self.nbinsx)
        m2 = np.fft.fftfreq(self.nbinsy, d=1.0 / self.nbinsy)
        m3 = np.fft.fftfreq(self.nbinsz, d=1.0 / self.nbinsz)
        M1, M2, M3 = np.meshgrid(m1, m2, m3, indexing='ij')
        m_stack = np.stack([M1, M2, M3], axis=-1)
        M_inv_T = self.cell_inverse.T
        k_vectors = 2 * np.pi * np.einsum('ab,ijkb->ijka', M_inv_T, m_stack)
        ksquared = np.sum(k_vectors ** 2, axis=-1)
        return k_vectors, ksquared

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

        .. deprecated::
            Use ``accumulate(..., compute_lambda=True)`` instead. This method
            re-reads the trajectory, whereas the new approach collects statistics
            during accumulation (significantly faster, supports multiple trajectories).

        Parameters
        ----------
        trajectory : Trajectory
            Trajectory-state providing per-frame positions and forces.
        sections : int, optional
            Number of interleaved frame-subsets used to accumulate covariance
            buffers. If None, defaults to `trajectory.frames`. Must be >= 2
            for variance estimation (a ValueError is raised otherwise).

        Raises
        ------
        RuntimeError
            If called before `accumulate`.
        """
        warnings.warn(
            "get_lambda() is deprecated. Use accumulate(..., compute_lambda=True) instead: "
            "grid.accumulate(traj, atom_names, compute_lambda=True)",
            DeprecationWarning,
            stacklevel=2,
        )
        if self.count == 0:
            raise RuntimeError("Run accumulate() before estimating lambda.")

        if sections is None:
            sections = trajectory.frames

        if sections > len(self.to_run):
            raise ValueError(
                f"sections ({sections}) exceeds the number of frames "
                f"to process ({len(self.to_run)})"
            )

        # Reset accumulators and re-accumulate with lambda statistics.
        self.force_x.fill(0)
        self.force_y.fill(0)
        self.force_z.fill(0)
        self.counter.fill(0)
        self.count = 0
        self._welford = None
        self._invalidate_derived_state()

        self._accumulate_with_sections(trajectory, sections)
        self._finalise_lambda()


def compute_density(
    trajectory: Trajectory,
    atom_names: str | list[str],
    *,
    rigid: bool = False,
    centre_location: bool | int = True,
    density_type: str = "number",
    nbins: int | tuple[int, int, int] = 100,
    kernel: str = "triangular",
    polarisation_axis: int = 0,
    start: int = 0,
    stop: int | None = None,
    period: int = 1,
    compute_lambda: bool = False,
    sections: int | None = None,
    integration: str | None = None,  # Deprecated
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
    compute_lambda : bool, optional
        If True, collect variance statistics for lambda estimation during
        accumulation. The variance-minimised density will be available via
        grid.rho_lambda. Default is False (faster, no lambda overhead).
    sections : int or None, optional
        Number of sections for lambda estimation. Only used when
        compute_lambda=True. If None, defaults to one section per frame.
    integration : str, optional
        .. deprecated::
            Use ``compute_lambda=True`` instead of ``integration='lambda'``.

    Returns
    -------
    DensityGrid
        Grid with computed density. Access rho_force for force-based density,
        rho_count for counting density, or rho_lambda for variance-minimised
        density (if compute_lambda=True).

    Notes
    -----
    Densities are computed lazily on first access to the corresponding
    properties (``rho_force``, ``rho_count``, and ``rho_lambda``).

    When ``compute_lambda=False`` (the default), lambda statistics are not
    accumulated and only ``rho_force`` and ``rho_count`` are available; their
    FFT-based real-space densities are computed on demand when those
    properties are first accessed.

    When ``compute_lambda=True``, lambda statistics are accumulated during
    :meth:`DensityGrid.accumulate`, and the variance-minimised density
    ``rho_lambda`` is computed on demand when that property is first accessed.
    Accessing ``rho_force`` or ``rho_count`` similarly triggers computation of
    their respective densities if they have not yet been evaluated.

    Examples
    --------
    >>> from revelsMD.density import compute_density
    >>> grid = compute_density(trajectory, 'O', nbins=50)
    >>> density = grid.rho_force

    >>> # With variance-minimised lambda estimation
    >>> grid = compute_density(trajectory, 'O', compute_lambda=True)
    >>> density = grid.rho_lambda
    """
    # Handle deprecated integration parameter
    if integration is not None:
        warnings.warn(
            "integration parameter is deprecated. Use compute_lambda=True instead: "
            "compute_density(..., compute_lambda=True)",
            DeprecationWarning,
            stacklevel=2,
        )
        if integration == "lambda":
            compute_lambda = True
        elif integration != "standard":
            raise ValueError(f"integration must be 'standard' or 'lambda', got '{integration}'")

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
        compute_lambda=compute_lambda,
        sections=sections,
    )
    # Densities are computed on demand when rho_force/rho_count/rho_lambda
    # properties are accessed, so no explicit computation call is needed here.

    return grid
