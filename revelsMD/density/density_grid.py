"""DensityGrid class for accumulating 3D force fields and converting to densities."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import numpy as np
import scipy.fft
from tqdm import tqdm


from revelsMD.backends import get_fft_workers
from revelsMD.frame_sources import BlockSource, contiguous_blocks, interleaved_blocks
from revelsMD.trajectories._base import Trajectory, normalize_bounds
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
        Trajectory._validate_cell_matrix(self.cell_matrix)
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
        """Variance-minimised density (available after accumulate with compute_lambda)."""
        if self._rho_lambda is None and self._welford is not None:
            self._finalise_lambda()
        return self._rho_lambda

    @property
    def lambda_weights(self) -> np.ndarray | None:
        """Per-voxel lambda weights (available after accumulate with compute_lambda)."""
        if self._lambda_weights is None and self._welford is not None:
            self._finalise_lambda()
        return self._lambda_weights

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
        self._deposit_to_arrays(
            self.force_x, self.force_y, self.force_z, self.counter,
            positions, forces, weights, kernel,
        )
        # Increment count: once per array for single species, once per
        # sub-array for multi-species (list of arrays).
        if isinstance(positions, list):
            self.count += len(positions)
        else:
            self.count += 1

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
        blocking: Literal["contiguous", "interleaved"] = "contiguous",
        block_size: int | None = None,
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
        blocking : {'contiguous', 'interleaved'}, optional
            How frames are grouped into blocks for lambda estimation
            (default: 'contiguous'). Only used when compute_lambda=True.

            - ``'contiguous'``: each block is a sequential slice of frames.
              Works with all trajectory backends (streaming, no random access
              required). Block size is controlled by ``block_size``.
            - ``'interleaved'``: section *k* gets every *k*-th frame
              (e.g. frames [0,2,4,...] and [1,3,5,...] for 2 sections).
              Requires a trajectory backend that supports random frame access
              via ``get_frame()``. Number of sections is controlled by
              ``sections``.
        block_size : int or None, optional
            Number of frames per block for contiguous blocking. Only used when
            blocking='contiguous' and compute_lambda=True. If None, defaults
            to one frame per block (i.e. each frame is its own block).
            The final block may contain fewer frames if the total is not
            evenly divisible.
        sections : int or None, optional
            Number of interleaved sections for lambda estimation. Only used
            when blocking='interleaved' and compute_lambda=True. If None,
            defaults to one section per frame.
            At least 2 blocks/sections are required across all
            ``accumulate()`` calls for variance estimation; accessing
            ``rho_lambda`` with fewer raises ValueError.

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

        # Validate frame bounds — reject out-of-range indices that are
        # almost certainly caller errors.  normalize_bounds (below) handles
        # the *representation* (negative indices, None stop) but not
        # *validity*; it will silently clamp out-of-range values, which
        # would hide mistakes at this user-facing boundary.
        if start > trajectory.frames:
            raise ValueError("First frame index exceeds frames in trajectory.")
        if start < -trajectory.frames:
            raise ValueError(
                f"Negative start index ({start}) exceeds trajectory length "
                f"({trajectory.frames})."
            )
        self.start = start

        if stop is not None and stop > trajectory.frames:
            raise ValueError("Final frame index exceeds frames in trajectory.")
        if stop is not None and stop < -trajectory.frames:
            raise ValueError(
                f"Negative stop index ({stop}) exceeds trajectory length "
                f"({trajectory.frames})."
            )
        self.stop = stop

        # Normalise the *representation* of validated bounds (negative
        # indices → positive, None stop → trajectory.frames) so that
        # frame_indices matches what iter_frames will produce.
        norm_start, norm_stop, _ = normalize_bounds(
            trajectory.frames, start, stop, period
        )
        frame_indices = range(int(norm_start), int(norm_stop), period)
        if len(frame_indices) == 0:
            raise ValueError(
                f"No frames selected: start={start}, stop={stop}, "
                f"period={period} yields an empty range."
            )
        self.period = period
        self.kernel = kernel

        # Validate centre_location
        if not isinstance(centre_location, (bool, int)):
            raise ValueError("centre_location must be True (COM) or int (specific atom index).")

        # Validate blocking parameter
        if blocking not in ("contiguous", "interleaved"):
            raise ValueError(
                f"blocking must be 'contiguous' or 'interleaved', got {blocking!r}"
            )

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
            # Simple accumulation — stream frames directly, no block overhead.
            for frame in tqdm(
                trajectory.iter_frames(start, stop, period), total=len(frame_indices)
            ):
                deposit_positions, deposit_forces, weights = self._selection.extract(frame)
                self.deposit(deposit_positions, deposit_forces, weights, kernel=kernel)
        else:
            # Block accumulation with lambda statistics
            if blocking == "contiguous" and sections is not None:
                warnings.warn(
                    "sections is ignored with blocking='contiguous'. "
                    "Use block_size to control contiguous block size, or "
                    "blocking='interleaved' to use sections.",
                    UserWarning,
                    stacklevel=2,
                )
            if blocking == "interleaved" and block_size is not None:
                warnings.warn(
                    "block_size is ignored with blocking='interleaved'. "
                    "Use sections to control the number of interleaved groups, "
                    "or blocking='contiguous' to use block_size.",
                    UserWarning,
                    stacklevel=2,
                )
            if blocking == "interleaved":
                effective_sections = sections if sections is not None else len(frame_indices)
                if effective_sections <= 0:
                    raise ValueError("sections must be a positive integer")
                # Requesting more sections than frames is almost certainly a
                # user error — fail fast rather than silently producing fewer
                # blocks than requested.  (interleaved_blocks itself skips
                # empty sections defensively, but that is an internal safety
                # net, not a public contract.)
                if effective_sections > len(frame_indices):
                    raise ValueError(
                        f"sections ({effective_sections}) exceeds the number of "
                        f"frames to process ({len(frame_indices)})"
                    )
                blocks = interleaved_blocks(trajectory, frame_indices, effective_sections)
                n_blocks = effective_sections
            else:
                effective_block_size = block_size if block_size is not None else 1
                if effective_block_size <= 0:
                    raise ValueError("block_size must be a positive integer")
                blocks = contiguous_blocks(
                    trajectory.iter_frames(start, stop, period),
                    effective_block_size,
                )
                n_blocks = -(-len(frame_indices) // effective_block_size)  # ceiling division

            self._accumulate_blocks(blocks, kernel, n_blocks=n_blocks)

        self.frames_processed += len(frame_indices)

    def _invalidate_derived_state(self) -> None:
        """Clear cached densities and lambda weights.

        Called when accumulator data changes, making any previously computed
        densities stale.
        """
        self._rho_force = None
        self._rho_count = None
        self._rho_lambda = None
        self._lambda_weights = None

    def _accumulate_blocks(
        self,
        blocks: BlockSource,
        kernel: str,
        n_blocks: int | None = None,
    ) -> None:
        """Accumulate blocks of frames while collecting lambda statistics.

        Parameters
        ----------
        blocks : BlockSource
            Iterator of blocks, where each block is an iterator of
            Frame instances.
        kernel : str
            Deposition kernel name.
        """
        # Initialise Welford accumulator if first call with compute_lambda
        if self._welford is None:
            self._welford = WelfordAccumulator3D(
                shape=(self.nbinsx, self.nbinsy, self.nbinsz)
            )

        # Preallocate block buffers once, reused each iteration via .fill(0).
        block_force_x = np.zeros_like(self.force_x)
        block_force_y = np.zeros_like(self.force_y)
        block_force_z = np.zeros_like(self.force_z)
        block_counter = np.zeros_like(self.counter)

        for block in tqdm(blocks, total=n_blocks, desc="Accumulating blocks"):
            # Reset block accumulators
            block_force_x.fill(0)
            block_force_y.fill(0)
            block_force_z.fill(0)
            block_counter.fill(0)
            block_count = 0

            for frame in block:
                deposit_positions, deposit_forces, weights = self._selection.extract(frame)

                self._deposit_to_arrays(
                    block_force_x, block_force_y, block_force_z, block_counter,
                    deposit_positions, deposit_forces, weights, kernel,
                )

                # Count semantics: increment once per deposited array,
                # not once per frame (consistent with deposit()).
                if isinstance(deposit_positions, list):
                    block_count += len(deposit_positions)
                else:
                    block_count += 1

            if block_count == 0:
                warnings.warn(
                    "Empty block encountered during accumulation — "
                    "this block contributes nothing to the variance estimate.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue

            # Add block data to main accumulators
            self.force_x += block_force_x
            self.force_y += block_force_y
            self.force_z += block_force_z
            self.counter += block_counter
            self.count += block_count

            # Compute block densities for Welford update
            block_rho_force, block_rho_count = self._compute_densities_from_arrays(
                block_force_x, block_force_y, block_force_z,
                block_counter, block_count
            )
            block_delta = block_rho_force - block_rho_count

            self._welford.update(block_delta, block_rho_force, weight=block_count)

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
            if len(positions) != len(forces):
                raise ValueError(
                    f"positions and forces must have the same length, "
                    f"got {len(positions)} and {len(forces)}"
                )
            if isinstance(weights, list):
                if len(weights) != len(positions):
                    raise ValueError(
                        f"weights must have the same length as positions, "
                        f"got {len(weights)} and {len(positions)}"
                    )
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
            Density perturbation in k-space (rfft layout).
        del_rho_n : ndarray
            Density perturbation in real space.
        """
        if count == 0:
            zeros = np.zeros_like(force_x)
            rfft_shape = (self.nbinsx, self.nbinsy, self.nbinsz // 2 + 1)
            return (
                np.zeros_like(force_x),
                np.zeros_like(force_x),
                np.zeros(rfft_shape, dtype=complex),
                np.zeros_like(force_x),
            )

        # Counting density (count > 0 guaranteed by early return above)
        rho_count = counter * (1.0 / (self.voxel_volume * count))

        # FFT of normalised forces — rfftn exploits real input symmetry
        workers = get_fft_workers()
        scale = 1.0 / (count * self.voxel_volume)
        fx_fft = scipy.fft.rfftn(force_x * scale, workers=workers)
        fy_fft = scipy.fft.rfftn(force_y * scale, workers=workers)
        fz_fft = scipy.fft.rfftn(force_z * scale, workers=workers)

        # k . F(k) dot product using precomputed 3D k-vectors
        kx = self._k_vectors[..., 0]
        ky = self._k_vectors[..., 1]
        kz = self._k_vectors[..., 2]
        k_dot_F = kx * fx_fft + ky * fy_fft + kz * fz_fft

        # Avoid ksquared.copy() — set-and-reset the DC component.
        # Not thread-safe, but accumulation is inherently sequential.
        saved_dc = self._ksquared[0, 0, 0]
        self._ksquared[0, 0, 0] = 1.0
        del_rho_k = (1j * self.beta / self._ksquared) * k_dot_F
        self._ksquared[0, 0, 0] = saved_dc

        del_rho_k[0, 0, 0] = 0.0

        # Back to real space — irfftn returns real directly
        real_shape = (self.nbinsx, self.nbinsy, self.nbinsz)
        del_rho_n = -scipy.fft.irfftn(del_rho_k, s=real_shape, workers=workers)
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
        if self._welford is None:
            return
        if not self._welford.has_data:
            warnings.warn(
                "compute_lambda was requested but the Welford accumulator "
                "has no data — rho_lambda will be None.",
                RuntimeWarning,
                stacklevel=2,
            )
            return

        if self._rho_lambda is not None:
            return  # Already finalised

        # Compute the expected densities from full accumulation (uses caching)
        expected_rho_force = self.rho_force
        expected_rho_count = self.rho_count

        if expected_rho_force is None or expected_rho_count is None:
            raise RuntimeError(
                "rho_force/rho_count unexpectedly None after accumulation"
            )

        if self._welford.count < 2:
            raise ValueError(
                f"Cannot compute lambda with fewer than 2 blocks (have {self._welford.count}). "
                "Use a smaller block_size (contiguous) or more sections (interleaved), "
                "or accumulate from additional trajectories."
            )

        # Finalise Welford statistics
        var_buffer, cov_buffer_force = self._welford.finalise()

        # lambda_raw = Cov(delta, rho_force) / Var(delta).
        # _lambda_weights = 1 - lambda_raw.
        # combine_estimators computes:
        #   rho_lambda = rho_count * lambda_raw + rho_force * (1 - lambda_raw)
        # i.e. rho_count * (1 - _lambda_weights) + rho_force * _lambda_weights
        # (see Coles et al., J. Phys. Chem. B 2021).
        lambda_raw = compute_lambda_weights(var_buffer, cov_buffer_force)
        self._lambda_weights = 1.0 - lambda_raw

        self._rho_lambda = combine_estimators(
            expected_rho_count,
            expected_rho_force,
            self._lambda_weights,
        )

    def _compute_real_densities(self) -> None:
        """Internal: compute rho_force and rho_count from accumulators via FFT."""
        if self.count == 0:
            raise RuntimeError("Run accumulate() before computing densities.")

        self._rho_force, self._rho_count, self.del_rho_k, self.del_rho_n = (
            self._fft_force_to_density(
                self.force_x, self.force_y, self.force_z, self.counter, self.count
            )
        )

    def _build_kvectors_3d(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Build 3D k-vector arrays in rfft layout for general (triclinic) cells.

        The k-vector at Miller indices (m1, m2, m3) is:
            k = 2 * pi * inv(M)^T @ [m1, m2, m3]^T
        where M is the cell matrix with rows = lattice vectors.

        Returns
        -------
        k_vectors : np.ndarray, shape (nbinsx, nbinsy, nbinsz // 2 + 1, 3)
            Cartesian k-vectors at each reciprocal grid point (rfft layout).
        ksquared : np.ndarray, shape (nbinsx, nbinsy, nbinsz // 2 + 1)
            |k|^2 at each reciprocal grid point (rfft layout).
        """
        # These k-vectors are multiplied element-wise with rfftn output
        # in _fft_force_to_density, so the grid must match its layout:
        # full frequencies on the first two axes, and rfftfreq on the last
        # which gives only non-negative frequencies (0 to Nyquist).
        m1 = np.fft.fftfreq(self.nbinsx, d=1.0 / self.nbinsx)
        m2 = np.fft.fftfreq(self.nbinsy, d=1.0 / self.nbinsy)
        m3 = np.fft.rfftfreq(self.nbinsz, d=1.0 / self.nbinsz)
        M1, M2, M3 = np.meshgrid(m1, m2, m3, indexing='ij')
        m_stack = np.stack([M1, M2, M3], axis=-1)
        M_inv_T = self.cell_inverse.T
        k_vectors = 2 * np.pi * np.einsum('ab,ijkb->ijka', M_inv_T, m_stack)
        ksquared = np.sum(k_vectors ** 2, axis=-1)
        return k_vectors, ksquared

    def write_to_cube(
        self,
        density: str,
        filename: str | Path,
        *,
        threshold: float | None = None,
    ) -> None:
        """Write a density grid to a Gaussian ``.cube`` file.

        Uses ``self.cell_matrix`` for the cell geometry.  No atomic
        positions are written -- the purpose of this method is density
        output, not atomic structure.

        Parameters
        ----------
        density : str
            Name of the density to write.  One of ``"force"``,
            ``"count"``, ``"lambda"``, or ``"hybrid"``.
        filename : str or Path
            Output file path.
        threshold : float, optional
            Required when *density* is ``"hybrid"``.  Passed to
            :meth:`rho_hybrid`.

        Raises
        ------
        ValueError
            If *density* is not recognised, ``"hybrid"`` is requested
            without *threshold*, or *threshold* is passed for a
            non-hybrid density.
        RuntimeError
            If the requested density has not been computed yet.
        """
        match density:
            case "force":
                if threshold is not None:
                    raise ValueError(f"threshold is only valid for 'hybrid', not {density!r}.")
                grid = self.rho_force
            case "count":
                if threshold is not None:
                    raise ValueError(f"threshold is only valid for 'hybrid', not {density!r}.")
                grid = self.rho_count
            case "lambda":
                if threshold is not None:
                    raise ValueError(f"threshold is only valid for 'hybrid', not {density!r}.")
                grid = self.rho_lambda
            case "hybrid":
                if threshold is None:
                    raise ValueError("threshold is required when density is 'hybrid'.")
                grid = self.rho_hybrid(threshold)
            case _:
                raise ValueError(
                    f"Unknown density {density!r}. "
                    f"Expected one of {sorted(['count', 'force', 'hybrid', 'lambda'])}."
                )

        if grid is None:
            hint = (
                "Call accumulate() with compute_lambda=True."
                if density == "lambda"
                else "Call accumulate() first."
            )
            raise RuntimeError(
                f"{density} density has not been computed yet. {hint}"
            )

        from revelsMD.density.writers.cube import write_cube

        write_cube(filename, grid, self.cell_matrix)

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
    blocking: Literal["contiguous", "interleaved"] = "contiguous",
    block_size: int | None = None,
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
    compute_lambda : bool, optional
        If True, collect variance statistics for lambda estimation during
        accumulation. The variance-minimised density will be available via
        grid.rho_lambda. Default is False (faster, no lambda overhead).
    blocking : {'contiguous', 'interleaved'}, optional
        How frames are grouped into blocks (default: 'contiguous').
        See :meth:`DensityGrid.accumulate` for details.
    block_size : int or None, optional
        Number of frames per block for contiguous blocking.
        See :meth:`DensityGrid.accumulate` for details.
    sections : int or None, optional
        Number of interleaved sections for lambda estimation.
        See :meth:`DensityGrid.accumulate` for details.

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
        blocking=blocking,
        block_size=block_size,
        sections=sections,
    )
    # Densities are computed on demand when rho_force/rho_count/rho_lambda
    # properties are accessed, so no explicit computation call is needed here.

    return grid
