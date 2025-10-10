"""
Force-based radial distribution function (RDF) estimators for RevelsMD.

This module provides the `RevelsRDF` class implementing reduced-variance,
force-weighted radial distribution functions using atomic positions and
forces from supported trajectory-state objects (TS).

The estimators support both like- and unlike-species RDFs and can compute
variance-minimized λ-corrected RDFs via a linear combination of forward
and backward Heaviside integrations.

Notes
-----
- Assumes an orthorhombic (or cubic) periodic simulation cell.
- Algorithm uses a 1/r³ weighting of projected forces between atom pairs.
- Supported trajectory varieties: 'lammps', 'mda', 'vasp', 'numpy'.
- Physical sensibility of results depends on user-supplied forces/units.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from tqdm import tqdm

from revelsMD.revels_tools.lammps_parser import define_strngdex, frame_skip, get_a_frame
from revelsMD.revels_tools.conversion_factors import generate_boltzmann


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
        indicies: np.ndarray,
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
        indicies : np.ndarray
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
            return np.zeros(1, dtype=np.longdouble)
        pos_ang = pos_array[indicies, :]
        force_total = force_array[indicies, :]
        storage_array = np.zeros(np.size(bins), dtype=np.longdouble)
        ns = len(indicies)

        # Pairwise displacements and force components
        rx = np.zeros((ns, ns))
        ry = np.zeros((ns, ns))
        rz = np.zeros((ns, ns))
        Fx = np.zeros((ns, ns))
        Fy = np.zeros((ns, ns))
        Fz = np.zeros((ns, ns))

        for x in range(ns):
            ry[x, :] = pos_ang[:, 1] - pos_ang[x, 1]
            rx[x, :] = pos_ang[:, 0] - pos_ang[x, 0]
            rz[x, :] = pos_ang[:, 2] - pos_ang[x, 2]
            Fx[x, :] = force_total[:, 0]
            Fy[x, :] = force_total[:, 1]
            Fz[x, :] = force_total[:, 2]

        # Minimum image convention
        rx -= (np.ceil((np.abs(rx) - box_x / 2) / box_x)) * (box_x) * np.sign(rx)
        ry -= (np.ceil((np.abs(ry) - box_y / 2) / box_y)) * (box_y) * np.sign(ry)
        rz -= (np.ceil((np.abs(rz) - box_z / 2) / box_z)) * (box_z) * np.sign(rz)

        r = (rx * rx + ry * ry + rz * rz) ** 0.5

        # 1/r^3 force projection (with safe error state)
        with np.errstate(divide="ignore", invalid="ignore"):
            dot_prod = ((Fz * rz) + (Fy * ry) + (Fx * rx)) / r / r / r

        # Extra sanity filter (legacy behavior)
        dot_prod[(rx > box_x / 2) + (ry > box_y / 2) + (rz > box_z / 2)] = 0

        dp = dot_prod.reshape(-1)
        rn = r.reshape(-1)

        digtized_array = np.digitize(rn, bins) - 1
        dp[digtized_array == np.size(bins) - 1] = 0

        # Heaviside-style accumulation from the largest bin downward
        n_bins = np.size(bins)
        storage_array[n_bins - 1] = np.sum(dp[digtized_array == n_bins - 1])

        for l in range(n_bins - 2, -1, -1):
            mask = digtized_array == l
            if np.any(mask):
                storage_array[l] = np.sum(dp[mask])

        return np.nan_to_num(storage_array, nan=0.0, posinf=0.0, neginf=0.0)


    # -------------------------------------------------------------------------
    # Single-frame RDF (unlike pairs)
    # -------------------------------------------------------------------------
    @staticmethod
    def single_frame_rdf_unlike(
        pos_array: np.ndarray,
        force_array: np.ndarray,
        indicies: Sequence[np.ndarray],
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
        indicies : sequence of two np.ndarray
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
            return np.zeros(1, dtype=np.longdouble)
        pos_ang_1 = pos_array[indicies[0], :]
        force_total_1 = force_array[indicies[0], :]
        pos_ang_2 = pos_array[indicies[1], :]
        force_total_2 = force_array[indicies[1], :]

        storage_array = np.zeros(np.size(bins), dtype=np.longdouble)
        n1 = len(indicies[0])
        n2 = len(indicies[1])

        rx = np.zeros((n2, n1))
        ry = np.zeros((n2, n1))
        rz = np.zeros((n2, n1))
        Fx = np.zeros((n2, n1))
        Fy = np.zeros((n2, n1))
        Fz = np.zeros((n2, n1))

        for x in range(n2):
            ry[x, :] = pos_ang_1[:, 1] - pos_ang_2[x, 1]
            rx[x, :] = pos_ang_1[:, 0] - pos_ang_2[x, 0]
            rz[x, :] = pos_ang_1[:, 2] - pos_ang_2[x, 2]
            Fx[x, :] = force_total_1[:, 0] - force_total_2[x, 0]
            Fy[x, :] = force_total_1[:, 1] - force_total_2[x, 1]
            Fz[x, :] = force_total_1[:, 2] - force_total_2[x, 2]

        # Minimum image convention
        rx -= (np.ceil((np.abs(rx) - box_x / 2) / box_x)) * (box_x) * np.sign(rx)
        ry -= (np.ceil((np.abs(ry) - box_y / 2) / box_y)) * (box_y) * np.sign(ry)
        rz -= (np.ceil((np.abs(rz) - box_z / 2) / box_z)) * (box_z) * np.sign(rz)

        r = (rx * rx + ry * ry + rz * rz) ** 0.5

        with np.errstate(divide="ignore", invalid="ignore"):
            dot_prod = ((Fz * rz) + (Fy * ry) + (Fx * rx)) / r / r / r
        dot_prod[(rx > box_x / 2) + (ry > box_y / 2) + (rz > box_z / 2)] = 0

        dp = dot_prod.reshape(-1)
        rn = r.reshape(-1)
        digtized_array = np.digitize(rn, bins) - 1
        dp[digtized_array == np.size(bins) - 1] = 0
        
        
        # Heaviside-style accumulation from the largest bin downward
        n_bins = np.size(bins)
        storage_array[n_bins - 1] = np.sum(dp[digtized_array == n_bins - 1])

        for l in range(n_bins - 2, -1, -1):
            mask = digtized_array == l
            if np.any(mask):
                storage_array[l] = np.sum(dp[mask])

        return np.nan_to_num(storage_array, nan=0.0, posinf=0.0, neginf=0.0)


    # -------------------------------------------------------------------------
    # Multi-frame RDF
    # -------------------------------------------------------------------------
    @staticmethod
    def run_rdf(
        TS,
        atom_a: str,
        atom_b: str,
        temp: float,
        delr: float = 0.01,
        start: int = 0,
        stop: int = -1,
        period: int = 1,
        rmax: Union[bool, float] = True,
        from_zero: bool = True,
    ) -> Optional[np.ndarray]:
        """
        Compute the force-weighted RDF across multiple frames.

        Parameters
        ----------
        TS : object
            Trajectory state (LAMMPS, MDAnalysis, VASP, or NumPy compatible).
        atom_a, atom_b : str
            Species identifiers. If identical, computes like-pair RDF.
        temp : float
            Temperature in Kelvin.
        delr : float, optional
            Bin spacing in distance (default: 0.01).
        start, stop : int, optional
            Frame range for averaging (default: 0 → -1).
        period : int, optional
            Frame stride (default: 1).
        rmax : bool or float, optional
            If True, use half-box length (or max half-dimension for LAMMPS);
            otherwise, set numeric cutoff.
        from_zero : bool, optional
            If True, integrate from r=0, else from rmax.

        Returns
        -------
        numpy.ndarray of shape (2, n) | None
            RDF array ``[r, g(r)]``, or None if invalid frame range specified.

        Notes
        -----
        - Preserves original control flow and scaling prefactors.
        - For LAMMPS backends, reads positions/forces per-frame using helper parser.
        """
        if atom_a == atom_b:
            single_frame_function = RevelsRDF.single_frame_rdf_like
            indicies = TS.get_indicies(atom_a)
            prefactor = float(TS.box_x * TS.box_y * TS.box_z) / (float(len(indicies)) * float(len(indicies) - 1))
        else:
            indicies = [np.array(TS.get_indicies(atom_a)), np.array(TS.get_indicies(atom_b))]
            single_frame_function = RevelsRDF.single_frame_rdf_unlike
            prefactor = float(TS.box_x * TS.box_y * TS.box_z) / (float(len(indicies[1])) * float(len(indicies[0])))/2

        if start > TS.frames:
            print("First frame index exceeds frames in trajectory")
            return None
        if stop > TS.frames:
            print("Final frame index exceeds frames in trajectory")
            return None

        to_run = range(int(start % TS.frames), int(stop % TS.frames), period)
        if len(to_run) == 0:
            print("Final frame ocurs before first frame in trajectory")
            return None

        # Bin grid
        if TS.variety == "lammps":
            if rmax:
                bins = np.arange(0, np.max([TS.box_x / 2, TS.box_y / 2, TS.box_z / 2]), delr)
            else:
                bins = np.arange(0, float(rmax), delr)
        else:
            if rmax:
                bins = np.arange(0, TS.box_x / 2, delr)
            else:
                bins = np.arange(0, float(rmax), delr)

        accumulated_storage_array = np.zeros(np.size(bins), dtype=np.longdouble)

        if TS.variety == "lammps":
            f = open(TS.trajectory_file)
            neededQuantities = ["x", "y", "z", "fx", "fy", "fz"]
            stringdex = define_strngdex(neededQuantities, TS.dic)
            for frame_count in tqdm(to_run):
                vars_trest = get_a_frame(f, TS.num_ats, TS.header_length, stringdex)
                accumulated_storage_array += single_frame_function(
                    vars_trest[:, :3], vars_trest[:, 3:], indicies, TS.box_x, TS.box_y, TS.box_z, bins
                )
                frame_skip(f, TS.num_ats, period - 1, TS.header_length)

        elif TS.variety == "mda":
            for frame_count in tqdm(TS.mdanalysis_universe.trajectory[int(start % TS.frames): int(stop % TS.frames): period]):
                accumulated_storage_array += single_frame_function(
                    TS.mdanalysis_universe.trajectory.atoms.positions,
                    TS.mdanalysis_universe.trajectory.atoms.forces,
                    indicies,
                    TS.box_x,
                    TS.box_y,
                    TS.box_z,
                    bins,
                )

        elif TS.variety == "vasp":
            for frame_count in tqdm(to_run):
                accumulated_storage_array += single_frame_function(
                    TS.positions[frame_count], TS.forces[frame_count], indicies, TS.box_x, TS.box_y, TS.box_z, bins
                )

        elif TS.variety == "numpy":
            for frame_count in tqdm(to_run):
                accumulated_storage_array += single_frame_function(
                    TS.positions[frame_count], TS.forces[frame_count], indicies, TS.box_x, TS.box_y, TS.box_z, bins
                )

        # Scale and integrate
        accumulated_storage_array = np.nan_to_num(accumulated_storage_array)
        accumulated_storage_array *= prefactor / (4 * np.pi * len(to_run) * generate_boltzmann(TS.units) * temp)

        if from_zero is True:
            return np.array([bins, np.cumsum(accumulated_storage_array)])
        else:
            return np.array([bins, 1 - np.cumsum(accumulated_storage_array[::-1])[::-1]])

    # -------------------------------------------------------------------------
    # Multi-frame RDF with λ-combination
    # -------------------------------------------------------------------------
    @staticmethod
    def run_rdf_lambda(
        TS,
        atom_a: str,
        atom_b: str,
        temp: float,
        delr: float = 0.01,
        start: int = 0,
        stop: int = -1,
        period: int = 1,
        rmax: Union[bool, float] = True,
    ) -> Optional[np.ndarray]:
        """
        Compute the λ-corrected RDF by combining forward and backward estimates.

        Parameters
        ----------
        TS : object
            Trajectory state (LAMMPS, MDAnalysis, VASP, or NumPy compatible).
        atom_a, atom_b : str
            Species identifiers. If identical, computes like-pair RDF.
        temp : float
            Temperature in Kelvin.
        delr : float, optional
            Bin spacing in distance (default: 0.01).
        start, stop : int, optional
            Frame range for averaging (default: 0 → -1).
        period : int, optional
            Frame stride (default: 1).
        rmax : bool or float, optional
            If True, use half-box length (or max half-dimension for LAMMPS);
            otherwise, set numeric cutoff.

        Returns
        -------
        numpy.ndarray of shape (n, 3) | None
            Columns: `[r, g_lambda(r), lambda(r)]`, or None if invalid frame range.

        Notes
        -----
        - Preserves original data flow and prefactor usage.
        - Uses covariance-based λ(r) = cov(delta, g_inf) / var(delta) with
          delta = g_inf - g_zero from the accumulated estimator.
        """
        if atom_a == atom_b:
            single_frame_function = RevelsRDF.single_frame_rdf_like
            indicies = TS.get_indicies(atom_a)
            prefactor = float(TS.box_x * TS.box_y * TS.box_z) / (float(len(indicies)) * float(len(indicies) - 1))
        else:
            indicies = [TS.get_indicies(atom_a), TS.get_indicies(atom_b)]
            single_frame_function = RevelsRDF.single_frame_rdf_unlike
            prefactor = float(TS.box_x * TS.box_y * TS.box_z) / (float(len(indicies[1])) * float(len(indicies[0])))/2

        if start > TS.frames:
            print("First frame index exceeds frames in trajectory")
            return None
        if stop > TS.frames:
            print("Final frame index exceeds frames in trajectory")
            return None

        to_run = range(int(start % TS.frames), int(stop % TS.frames), period)
        if len(to_run) == 0:
            print("Final frame ocurs before first frame in trajectory")
            return None

        # Bins
        if TS.variety == "lammps":
            if rmax:
                bins = np.arange(0, np.max([TS.box_x / 2, TS.box_y / 2, TS.box_z / 2]), delr)
            else:
                bins = np.arange(0, float(rmax), delr)
        else:
            if rmax:
                bins = np.arange(0, TS.box_x / 2, delr)
            else:
                bins = np.arange(0, float(rmax), delr)

        list_store: List[np.ndarray] = []
        accumulated_storage_array = np.zeros(np.size(bins), dtype=np.longdouble)

        if TS.variety == "lammps":
            f = open(TS.trajectory_file)
            neededQuantities = ["x", "y", "z", "fx", "fy", "fz"]
            stringdex = define_strngdex(neededQuantities, TS.dic)
            for frame_count in tqdm(to_run):
                vars_trest = get_a_frame(f, TS.num_ats, TS.header_length, stringdex)
                this_frame = single_frame_function(
                    vars_trest[:, :3], vars_trest[:, 3:], indicies, TS.box_x, TS.box_y, TS.box_z, bins
                )
                accumulated_storage_array += this_frame
                list_store.append(this_frame)
                frame_skip(f, TS.num_ats, period - 1, TS.header_length)

        elif TS.variety == "mda":
            for frame_count in tqdm(TS.mdanalysis_universe.trajectory[int(start % TS.frames): int(stop % TS.frames): period]):
                this_frame = single_frame_function(
                    TS.mdanalysis_universe.atoms.positions,
                    TS.mdanalysis_universe.atoms.forces,
                    indicies,
                    TS.box_x,
                    TS.box_y,
                    TS.box_z,
                    bins,
                )
                accumulated_storage_array += this_frame
                list_store.append(this_frame)

        elif TS.variety == "vasp":
            for frame_count in tqdm(to_run):
                this_frame = single_frame_function(
                    TS.positions[frame_count], TS.forces[frame_count], indicies, TS.box_x, TS.box_y, TS.box_z, bins
                )
                accumulated_storage_array += this_frame
                list_store.append(this_frame)

        elif TS.variety == "numpy":
            for frame_count in tqdm(to_run):
                this_frame = single_frame_function(
                    TS.positions[frame_count], TS.forces[frame_count], indicies, TS.box_x, TS.box_y, TS.box_z, bins
                )
                accumulated_storage_array += this_frame
                list_store.append(this_frame)

        # Build arrays and prefactors
        base_array = np.nan_to_num(np.array(list_store))
        base_array *= prefactor / (4 * np.pi * generate_boltzmann(TS.units) * temp)

        accumulated_storage_array = np.nan_to_num(accumulated_storage_array)
        accumulated_storage_array *= prefactor / (4 * np.pi * len(to_run) * generate_boltzmann(TS.units) * temp)

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


        g_lambda = np.nan_to_num(np.mean(base_inf_rdf * (1 - combination) + base_zero_rdf * combination, axis=0),nan=0.0,posinf=0.0,neginf=0.0)


        return np.transpose(np.array([bins[1:], g_lambda, combination]))

