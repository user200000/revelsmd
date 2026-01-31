"""
Force-based radial distribution function (RDF) estimators for RevelsMD.

This module provides the `RevelsRDF` class implementing reduced-variance,
force-weighted radial distribution functions using atomic positions and
forces from trajectory-state objects.

The estimators support both like- and unlike-species RDFs and can compute
variance-minimized λ-corrected RDFs via a linear combination of forward
and backward Heaviside integrations.

Notes
-----
- Assumes an orthorhombic (or cubic) periodic simulation cell.
- Algorithm uses a 1/r^3 weighting of projected forces between atom pairs.
- Physical sensibility of results depends on user-supplied forces/units.
"""

from collections.abc import Sequence

import numpy as np
from tqdm import tqdm

from revelsMD.rdf_helpers import (
    compute_pairwise_contributions_like,
    compute_pairwise_contributions_unlike,
    accumulate_binned_contributions,
)


class RevelsRDF:
    """
    Force-weighted radial distribution function (RDF) estimators.

    Implements reduced-variance RDF estimators using positional and
    force information from multiple trajectory backends (LAMMPS dump,
    MDAnalysis Universe, VASP parsed coordinates, or plain NumPy arrays).

    Notes
    -----
    Each single-frame estimator returns contributions accumulated at the
    user-supplied radial grid `bins`. Multi-frame drivers (`run_rdf`,
    `run_rdf_lambda`) build the radial array and average over frames,
    including unit-dependent prefactors.
    """

    # -------------------------------------------------------------------------
    # Single-frame RDF (like pairs)
    # -------------------------------------------------------------------------
    @staticmethod
    def single_frame_rdf_like(
        pos_array: np.ndarray,
        force_array: np.ndarray,
        indices: np.ndarray,
        box_x: float,
        box_y: float,
        box_z: float,
        bins: np.ndarray,
        return_conventional: bool = False,
    ) -> np.ndarray:
        """
        Compute a single-frame reduced-variance RDF for identical species.

        Parameters
        ----------
        pos_array : (N, 3) np.ndarray
            Atomic positions in Cartesian coordinates.
        force_array : (N, 3) np.ndarray
            Atomic forces in Cartesian coordinates.
        indices : np.ndarray
            Indices of atoms belonging to the species of interest.
        box_x, box_y, box_z : float
            Orthorhombic cell lengths.
        bins : np.ndarray
            Radial grid at which cumulative Heaviside contributions are accumulated.
        return_conventional : bool, optional
            Placeholder for a conventional histogram variant (not used).

        Returns
        -------
        np.ndarray
            Accumulated force-weighted contributions per bin for this frame.
        """
        n_bins = np.size(bins)
        if n_bins == 0:
            return np.zeros(1, dtype=np.float64)

        pos_ang = pos_array[indices, :]
        force_total = force_array[indices, :]

        # Vectorised pairwise calculation and binning
        r_flat, dot_flat = compute_pairwise_contributions_like(
            pos_ang, force_total, (box_x, box_y, box_z)
        )
        return accumulate_binned_contributions(dot_flat, r_flat, bins)


    # -------------------------------------------------------------------------
    # Single-frame RDF (unlike pairs)
    # -------------------------------------------------------------------------
    @staticmethod
    def single_frame_rdf_unlike(
        pos_array: np.ndarray,
        force_array: np.ndarray,
        indices: Sequence[np.ndarray],
        box_x: float,
        box_y: float,
        box_z: float,
        bins: np.ndarray,
        return_conventional: bool = False,
    ) -> np.ndarray:
        """
        Compute a single-frame reduced-variance RDF for unlike species.

        Parameters
        ----------
        pos_array : (N, 3) np.ndarray
            Atomic positions in Cartesian coordinates.
        force_array : (N, 3) np.ndarray
            Atomic forces in Cartesian coordinates.
        indices : sequence of two np.ndarray
            `[indices_species_A, indices_species_B]`.
        box_x, box_y, box_z : float
            Orthorhombic cell lengths.
        bins : np.ndarray
            Radial grid at which cumulative Heaviside contributions are accumulated.
        return_conventional : bool, optional
            Placeholder for a conventional histogram variant (not used).

        Returns
        -------
        np.ndarray
            Accumulated force-weighted contributions per bin for this frame.
        """
        n_bins = np.size(bins)
        if n_bins == 0:
            return np.zeros(1, dtype=np.float64)

        pos_ang_1 = pos_array[indices[0], :]
        force_total_1 = force_array[indices[0], :]
        pos_ang_2 = pos_array[indices[1], :]
        force_total_2 = force_array[indices[1], :]

        # Vectorised pairwise calculation and binning
        r_flat, dot_flat = compute_pairwise_contributions_unlike(
            pos_ang_1, pos_ang_2, force_total_1, force_total_2,
            (box_x, box_y, box_z)
        )
        return accumulate_binned_contributions(dot_flat, r_flat, bins)


    # -------------------------------------------------------------------------
    # Multi-frame RDF
    # -------------------------------------------------------------------------
    @staticmethod
    def run_rdf(
        trajectory,
        atom_a: str,
        atom_b: str,
        delr: float = 0.01,
        start: int = 0,
        stop: int | None = None,
        period: int = 1,
        rmax: bool | float = True,
        from_zero: bool = True,
    ) -> np.ndarray:
        """
        Compute the force-weighted RDF across multiple frames.

        Parameters
        ----------
        trajectory : TrajectoryState
            Trajectory state object providing positions, forces, box dimensions, and beta.
        atom_a, atom_b : str
            Species identifiers. If identical, computes like-pair RDF.
        delr : float, optional
            Bin spacing in distance (default: 0.01).
        start : int, optional
            First frame index (default: 0).
        stop : int or None, optional
            Stop frame index (default: None, meaning all frames).
            Negative indices count from end (e.g., -1 = all but last).
        period : int, optional
            Frame stride (default: 1).
        rmax : bool or float, optional
            If True, use half the minimum box dimension; otherwise, set numeric cutoff.
        from_zero : bool, optional
            If True, integrate from r=0, else from rmax.

        Returns
        -------
        numpy.ndarray of shape (2, n)
            RDF array ``[r, g(r)]``.

        Raises
        ------
        ValueError
            If frame bounds are invalid (start/stop exceed trajectory length,
            or the frame range is empty).
        """
        if atom_a == atom_b:
            single_frame_function = RevelsRDF.single_frame_rdf_like
            indices = trajectory.get_indices(atom_a)
            prefactor = float(trajectory.box_x * trajectory.box_y * trajectory.box_z) / (float(len(indices)) * float(len(indices) - 1))
        else:
            indices = [np.array(trajectory.get_indices(atom_a)), np.array(trajectory.get_indices(atom_b))]
            single_frame_function = RevelsRDF.single_frame_rdf_unlike  # type: ignore[assignment]
            prefactor = float(trajectory.box_x * trajectory.box_y * trajectory.box_z) / (float(len(indices[1])) * float(len(indices[0])))/2

        # Validate frame bounds
        if start > trajectory.frames:
            raise ValueError("First frame index exceeds frames in trajectory.")
        if stop is not None and stop > trajectory.frames:
            raise ValueError("Final frame index exceeds frames in trajectory.")

        # Calculate frame count for scaling
        effective_stop = trajectory.frames if stop is None else (trajectory.frames + stop if stop < 0 else stop)
        norm_start = start % trajectory.frames if start >= 0 else max(0, trajectory.frames + start)
        to_run = range(int(norm_start), int(effective_stop), period)
        if len(to_run) == 0:
            raise ValueError("Final frame occurs before first frame in trajectory.")

        # Bin grid: use half the minimum box dimension
        if rmax is True:
            rmax_value = min(trajectory.box_x, trajectory.box_y, trajectory.box_z) / 2
        else:
            rmax_value = float(rmax)
        bins = np.arange(0, rmax_value, delr)

        accumulated_storage_array = np.zeros(np.size(bins), dtype=np.float64)

        # Unified frame iteration using iter_frames
        for positions, forces in tqdm(trajectory.iter_frames(start, stop, period), total=len(to_run)):
            accumulated_storage_array += single_frame_function(
                positions, forces, indices, trajectory.box_x, trajectory.box_y, trajectory.box_z, bins
            )

        # Scale and integrate
        accumulated_storage_array = np.nan_to_num(accumulated_storage_array)
        accumulated_storage_array *= prefactor * trajectory.beta / (4 * np.pi * len(to_run))

        if from_zero is True:
            return np.array([bins, np.cumsum(accumulated_storage_array)])
        else:
            return np.array([bins, 1 - np.cumsum(accumulated_storage_array[::-1])[::-1]])

    # -------------------------------------------------------------------------
    # Multi-frame RDF with λ-combination
    # -------------------------------------------------------------------------
    @staticmethod
    def run_rdf_lambda(
        trajectory,
        atom_a: str,
        atom_b: str,
        delr: float = 0.01,
        start: int = 0,
        stop: int | None = None,
        period: int = 1,
        rmax: bool | float = True,
    ) -> np.ndarray:
        """
        Compute the λ-corrected RDF by combining forward and backward estimates.

        Parameters
        ----------
        trajectory : TrajectoryState
            Trajectory state object providing positions, forces, box dimensions, and beta.
        atom_a, atom_b : str
            Species identifiers. If identical, computes like-pair RDF.
        delr : float, optional
            Bin spacing in distance (default: 0.01).
        start : int, optional
            First frame index (default: 0).
        stop : int or None, optional
            Stop frame index (default: None, meaning all frames).
            Negative indices count from end (e.g., -1 = all but last).
        period : int, optional
            Frame stride (default: 1).
        rmax : bool or float, optional
            If True, use half the minimum box dimension; otherwise, set numeric cutoff.

        Returns
        -------
        numpy.ndarray of shape (n, 3)
            Columns: `[r, g_lambda(r), lambda(r)]`.

        Raises
        ------
        ValueError
            If frame bounds are invalid (start/stop exceed trajectory length,
            or the frame range is empty).

        Notes
        -----
        Uses covariance-based λ(r) = cov(delta, g_inf) / var(delta) with
        delta = g_inf - g_zero from the accumulated estimator.
        """
        if atom_a == atom_b:
            single_frame_function = RevelsRDF.single_frame_rdf_like
            indices = trajectory.get_indices(atom_a)
            prefactor = float(trajectory.box_x * trajectory.box_y * trajectory.box_z) / (float(len(indices)) * float(len(indices) - 1))
        else:
            indices = [trajectory.get_indices(atom_a), trajectory.get_indices(atom_b)]
            single_frame_function = RevelsRDF.single_frame_rdf_unlike  # type: ignore[assignment]
            prefactor = float(trajectory.box_x * trajectory.box_y * trajectory.box_z) / (float(len(indices[1])) * float(len(indices[0])))/2

        # Validate frame bounds
        if start > trajectory.frames:
            raise ValueError("First frame index exceeds frames in trajectory.")
        if stop is not None and stop > trajectory.frames:
            raise ValueError("Final frame index exceeds frames in trajectory.")

        # Calculate frame count for scaling
        effective_stop = trajectory.frames if stop is None else (trajectory.frames + stop if stop < 0 else stop)
        norm_start = start % trajectory.frames if start >= 0 else max(0, trajectory.frames + start)
        to_run = range(int(norm_start), int(effective_stop), period)
        if len(to_run) == 0:
            raise ValueError("Final frame occurs before first frame in trajectory.")

        # Bin grid: use half the minimum box dimension
        if rmax is True:
            rmax_value = min(trajectory.box_x, trajectory.box_y, trajectory.box_z) / 2
        else:
            rmax_value = float(rmax)
        bins = np.arange(0, rmax_value, delr)

        list_store: list[np.ndarray] = []
        accumulated_storage_array = np.zeros(np.size(bins), dtype=np.float64)

        # Unified frame iteration using iter_frames
        for positions, forces in tqdm(trajectory.iter_frames(start, stop, period), total=len(to_run)):
            this_frame = single_frame_function(
                positions, forces, indices, trajectory.box_x, trajectory.box_y, trajectory.box_z, bins
            )
            accumulated_storage_array += this_frame
            list_store.append(this_frame)

        # Build arrays and prefactors
        base_array = np.nan_to_num(np.array(list_store))
        base_array *= prefactor * trajectory.beta / (4 * np.pi)

        accumulated_storage_array = np.nan_to_num(accumulated_storage_array)
        accumulated_storage_array *= prefactor * trajectory.beta / (4 * np.pi * len(to_run))

        # Expectation curves from accumulated estimator
        exp_zero_rdf = np.array(np.cumsum(accumulated_storage_array)[:-1])
        exp_inf_rdf = np.array(1 - np.cumsum(accumulated_storage_array[::-1])[::-1][1:])
        exp_delta = exp_inf_rdf - exp_zero_rdf

        # Per-frame forward/backward curves
        base_zero_rdf = np.array(np.cumsum(base_array, axis=1))[:, :-1]
        base_inf_rdf = np.array(1 - np.cumsum(base_array[:, ::-1], axis=1)[:, ::-1][:, 1:])
        base_delta = base_inf_rdf - base_zero_rdf

        # λ(r) from covariance/variance with numerical safety
        var_del = np.mean((base_delta - exp_delta) ** 2, axis=0)
        cov_inf = np.mean((base_delta - exp_delta) * (base_inf_rdf - exp_inf_rdf), axis=0)
        var_del_safe = np.where(var_del == 0, 1.0, var_del)
        combination = np.divide(cov_inf, var_del_safe)
        combination = np.nan_to_num(combination, nan=0.0, posinf=0.0, neginf=0.0)

        g_lambda = np.nan_to_num(np.mean(base_inf_rdf * (1 - combination) + base_zero_rdf * combination, axis=0), nan=0.0, posinf=0.0, neginf=0.0)

        return np.transpose(np.array([bins[1:], g_lambda, combination]))

