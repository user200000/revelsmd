"""
3D force-density estimators and utilities for RevelsMD.

This module provides the `Revels3D` functionality based nested classes that operate on
a trajectory-state object to build 3D force grids, convert them to real-space
densities via Fourier-space relations, write densities to .cube files, and compute
optimal linear combinations (λ-method) of counting and force-based densities.

Notes
-----
- The code assumes an **orthorhombic** (or cubic) periodic box.
- All kernels and gridding logic preserve your existing numerical behavior.
"""

from __future__ import annotations

import copy

import numpy as np
from tqdm import tqdm
from ase import Atoms
from ase.io.cube import write_cube
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from revelsMD.trajectories._base import Trajectory, DataUnavailableError
from revelsMD.grid_helpers import get_backend_functions as _get_grid_backend_functions

# Module-level backend functions (loaded once at import)
_triangular_allocation, _box_allocation = _get_grid_backend_functions()


class Revels3D:
    """
    Namespace wrapper for grid building, estimators, selection state, and helpers.

    The nested classes are:
    - `GridState`: holds voxelized force accumulators and builds densities.
    - `Estimators`: per-frame estimators to deposit forces/weights on the grid.
    - `SelectionState`: encapsulates atom selections, masses/charges, and center logic.
    - `HelperFunctions`: numerics for COMs, dipoles, per-frame processing, kernels.
    """

    class GridState:
        """
        State for accumulating 3D force fields and converting to densities.

        Parameters
        ----------
        trajectory : Trajectory
            Trajectory-state object providing `box_x`, `box_y`, `box_z`, `units`, and `beta`.
        density_type : {'number', 'charge', 'polarisation'}
            Type of density to be constructed (controls the estimator weighting).
        nbins : int, optional
            Default number of voxels per box dimension (overridden by `nbinsx/y/z`).
        nbinsx, nbinsy, nbinsz : int or bool, optional
            Explicit voxel counts per dimension. If `False`, fall back to `nbins`.

        Attributes
        ----------
        forceX, forceY, forceZ : np.ndarray
            Accumulators for voxelized force components (shape: nbinsx×nbinsy×nbinsz).
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
            Simple state flag used by getters and λ estimator.
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
            self.SS = Revels3D.SelectionState(trajectory, atom_names=atom_names, centre_location=centre_location, rigid=rigid)

            # Choose estimator based on density type and rigid settings
            if self.density_type == "number":
                if not self.SS.indistinguishable_set:
                    if rigid:
                        if centre_location is True:
                            self.single_frame_function = Revels3D.Estimators.single_frame_rigid_number_com_grid
                        elif isinstance(centre_location, int):
                            self.single_frame_function = Revels3D.Estimators.single_frame_rigid_number_atom_grid
                        else:
                            raise ValueError("centre_location must be True (COM) or int (specific atom index).")
                    else:
                        self.single_frame_function = Revels3D.Estimators.single_frame_number_many_grid
                else:
                    self.single_frame_function = Revels3D.Estimators.single_frame_number_single_grid

            elif self.density_type == "charge":
                if not self.SS.indistinguishable_set:
                    if rigid:
                        if centre_location is True:
                            self.single_frame_function = Revels3D.Estimators.single_frame_rigid_charge_com_grid
                        elif isinstance(centre_location, int):
                            self.single_frame_function = Revels3D.Estimators.single_frame_rigid_charge_atom_grid
                        else:
                            raise ValueError("centre_location must be True (COM) or int (specific atom index).")
                    else:
                        self.single_frame_function = Revels3D.Estimators.single_frame_charge_many_grid
                else:
                    self.single_frame_function = Revels3D.Estimators.single_frame_number_single_grid

            elif self.density_type == "polarisation":
                if not self.SS.indistinguishable_set:
                    if rigid:
                        if centre_location is True:
                            self.single_frame_function = Revels3D.Estimators.single_frame_rigid_polarisation_com_grid
                            self.SS.polarisation_axis = polarisation_axis
                        elif isinstance(centre_location, int):
                            self.single_frame_function = Revels3D.Estimators.single_frame_rigid_polarisation_atom_grid
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
            Implements (Borgis et al., *Mol. Phys.* **111**, 3486–3492 (2013)):
            Δρ(k) = i / (k_B T k^2) * k · F(k), with Δρ(k=0) := 0,
            then ρ(r) = ⟨ρ_count(r)⟩ + ℱ⁻¹[Δρ(k)].

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

            # Δρ(k): i/(k_B T k^2) * (F·k); handle k^2=0 via errstate; enforce Δρ(0)=0
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
            Return the k-vectors (2π * FFT frequencies) per dimension.

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
            Compute optimal λ(r) to combine counting and force densities.

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
                covariance/variance buffers, `combination` (=1−cov_F/var),
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

            # λ = 1 − cov(force)/var(delta); optimal density = (1−λ)*count + λ*force
            GS_Lambda.combination = 1.0 - (GS_Lambda.cov_buffer_force / GS_Lambda.var_buffer)
            GS_Lambda.optimal_density = (1.0 - GS_Lambda.combination) * GS_Lambda.expected_particle_density + \
                GS_Lambda.combination * GS_Lambda.expected_rho
            GS_Lambda.grid_progress = "Lambda"
            return GS_Lambda

    class Estimators:
        """
        Per-frame estimators that compute positions/weights and deposit to the grid.

        Notes
        -----
        All estimators call `HelperFunctions.process_frame(...)` which:
        - Applies periodic imaging to positions
        - Assigns per-atom contributions using a chosen kernel
        - Updates force accumulators (X/Y/Z) and the counting `counter` field
        """

        @staticmethod
        def single_frame_rigid_number_com_grid(positions, forces, trajectory, GS, SS, kernel="triangular"):
            """Rigid molecule: number density at COM position; forces summed over rigid members."""
            coms = Revels3D.HelperFunctions.find_coms(positions, trajectory, GS, SS)
            rigid_forces = Revels3D.HelperFunctions.sum_forces(SS, forces)
            Revels3D.HelperFunctions.process_frame(trajectory, GS, coms, rigid_forces, kernel=kernel)

        @staticmethod
        def single_frame_rigid_number_atom_grid(positions, forces, trajectory, GS, SS, kernel="triangular"):
            """Rigid molecule: number density at a specific atom's position; forces summed over rigid members."""
            rigid_forces = Revels3D.HelperFunctions.sum_forces(SS, forces)
            Revels3D.HelperFunctions.process_frame(trajectory, GS, positions[SS.indices[SS.centre_location], :], rigid_forces, kernel=kernel)

        @staticmethod
        def single_frame_number_many_grid(positions, forces, trajectory, GS, SS, kernel="triangular"):
            """Non-rigid: number density for each species list; deposit per-entry."""
            for count in range(len(SS.indices)):
                Revels3D.HelperFunctions.process_frame(
                    trajectory, GS, positions[SS.indices[count], :], forces[SS.indices[count], :], kernel=kernel
                )

        @staticmethod
        def single_frame_number_single_grid(positions, forces, trajectory, GS, SS, kernel="triangular"):
            """Single species: number density at that species' positions."""
            Revels3D.HelperFunctions.process_frame(trajectory, GS, positions[SS.indices, :], forces[SS.indices, :], kernel=kernel)

        @staticmethod
        def single_frame_rigid_charge_atom_grid(positions, forces, trajectory, GS, SS, kernel="triangular"):
            """Rigid molecule: charge-weighted density at a specific atom's position."""
            rigid_forces = Revels3D.HelperFunctions.sum_forces(SS, forces)
            Revels3D.HelperFunctions.process_frame(
                trajectory, GS, positions[SS.indices[SS.centre_location], :], rigid_forces, a=SS.charges[SS.centre_location], kernel=kernel
            )

        @staticmethod
        def single_frame_charge_many_grid(positions, forces, trajectory, GS, SS, kernel="triangular"):
            """Non-rigid: charge-weighted density for each entry."""
            for count in range(len(SS.indices)):
                Revels3D.HelperFunctions.process_frame(
                    trajectory, GS, positions[SS.indices[count], :], forces[SS.indices[count], :], a=SS.charges[count], kernel=kernel
                )

        @staticmethod
        def single_frame_charge_single_grid(positions, forces, trajectory, GS, SS, kernel="triangular"):
            """Single species: charge-weighted density."""
            Revels3D.HelperFunctions.process_frame(
                trajectory, GS, positions[SS.indices, :], forces[SS.indices, :], a=SS.charges, kernel=kernel
            )

        @staticmethod
        def single_frame_rigid_charge_com_grid(positions, forces, trajectory, GS, SS, kernel="triangular"):
            """Rigid molecule: charge-weighted density at COM."""
            coms = Revels3D.HelperFunctions.find_coms(positions, trajectory, GS, SS)
            rigid_forces = Revels3D.HelperFunctions.sum_forces(SS, forces)
            Revels3D.HelperFunctions.process_frame(trajectory, GS, coms, rigid_forces, kernel=kernel, a=SS.charges)

        @staticmethod
        def single_frame_rigid_polarisation_com_grid(positions, forces, trajectory, GS, SS, kernel="triangular"):
            """Rigid molecule: polarisation density projected along `SS.polarisation_axis` at COM."""
            coms, molecular_dipole = Revels3D.HelperFunctions.find_coms(positions, trajectory, GS, SS, calc_dipoles=True)
            rigid_forces = Revels3D.HelperFunctions.sum_forces(SS, forces)
            Revels3D.HelperFunctions.process_frame(
                trajectory, GS, coms, rigid_forces, a=molecular_dipole[:, GS.SS.polarisation_axis], kernel=kernel
            )

        @staticmethod
        def single_frame_rigid_polarisation_atom_grid(positions, forces, trajectory, GS, SS, kernel="triangular"):
            """Rigid molecule: polarisation density projected along `SS.polarisation_axis` at COM (per original code)."""
            coms, molecular_dipole = Revels3D.HelperFunctions.find_coms(positions, trajectory, GS, SS, calc_dipoles=True)
            rigid_forces = Revels3D.HelperFunctions.sum_forces(SS, forces)
            Revels3D.HelperFunctions.process_frame(
                trajectory, GS, coms, rigid_forces, a=molecular_dipole[:, GS.SS.polarisation_axis], kernel=kernel
            )

    class SelectionState:
        """
        Atom selection, charges/masses, and center choice for grid deposition.

        Parameters
        ----------
        trajectory : Trajectory
            Trajectory-state with index/charge/mass accessors.
        atom_names : str or list of str
            For a single species, may be a string or single-element list.
            For a rigid molecule, provide a list of species names in the rigid group.
        centre_location : bool or int
            If a rigid group is provided: `True` selects COM; `int` selects one species'
            index within the rigid set as the center.

        Attributes
        ----------
        indistinguishable_set : bool
            True if a single species is selected; False for multi-species (rigid).
        indices : np.ndarray or list of np.ndarray
            Atom indices of the selection (kept name for compatibility).
        charges, masses : list or np.ndarray
            Per-species arrays (rigid) or single array (single species) if available.
        polarisation_axis : int
            Axis for polarisation projection (set by GridState when needed).
        """

        def __init__(self, trajectory: Trajectory, atom_names: str | list[str], centre_location: bool | int, rigid: bool = False):
            if isinstance(atom_names, list) and len(atom_names) > 1:
                self.indistinguishable_set = False
                self.indices: list[np.ndarray] = []
                self.charges: list[np.ndarray] = []
                self.masses: list[np.ndarray] = []
                for atom in atom_names:
                    self.indices.append(trajectory.get_indices(atom))
                    try:
                        self.charges.append(trajectory.get_charges(atom))
                        self.masses.append(trajectory.get_masses(atom))
                    except DataUnavailableError:
                        pass
                if rigid:
                    lengths = [len(idx) for idx in self.indices]
                    if len(set(lengths)) != 1:
                        raise ValueError(
                            f"When 'rigid=True', all atom selections must have the same number of indices, "
                            f"but got lengths {lengths} for atoms {atom_names}."
                        )
                if isinstance(centre_location, bool) or isinstance(centre_location, int):
                    if isinstance(centre_location, int) and centre_location >= len(atom_names):
                        raise ValueError("centre_location index exceeds number of provided atom names.")
                    self.centre_location = centre_location
                else:
                    raise ValueError("centre_location must be True (COM) or an integer index.")
            else:
                # Single species
                if isinstance(atom_names, list):
                    atom_names = atom_names[0]
                self.indistinguishable_set = True
                self.indices = trajectory.get_indices(atom_names)
                try:
                    self.charges = trajectory.get_charges(atom_names)
                    self.masses = trajectory.get_masses(atom_names)
                except DataUnavailableError:
                    pass

        def position_centre(self, species_number: int) -> None:
            """
            Set the active species index within a rigid group as the position center.

            Parameters
            ----------
            species_number : int
                Index into the multi-species selection list.

            Raises
            ------
            ValueError
                If `species_number` is out of range for the current selection.
            """
            if isinstance(self.indices, list) and species_number < len(self.indices):
                self.species_number = species_number
            else:
                raise ValueError("species_number out of range for current selection.")

    class HelperFunctions:
        """
        Helper numerics for per-frame deposition, COMs, dipoles, and rigid sums.
        """

        @staticmethod
        def process_frame(trajectory: Trajectory, GS: GridState, positions: np.ndarray, forces: np.ndarray, a: float = 1.0, kernel: str = "triangular") -> None:
            """
            Deposit a frame's positions/forces to the grid using a given kernel.

            Parameters
            ----------
            trajectory : Trajectory
                Trajectory state (box lengths used here).
            GS : GridState
                Grid accumulators and bin geometry.
            positions : (N, 3) np.ndarray
                Positions in Cartesian coordinates.
            forces : (N, 3) np.ndarray
                Forces corresponding to `positions`.
            a : float or np.ndarray, optional
                Scalar or per-particle weight (number/charge/polarisation projection).
            kernel : {'triangular','box'}, optional
                Assignment kernel.

            Notes
            -----
            - Positions are reduced into a primary image by component-wise remainders.
            - `np.digitize` is used to map positions to voxel indices; subsequent kernels
              deposit weighted contributions to neighbors (triangular) or the host voxel (box).
            """
            GS.count += 1

            # Bring positions to the primary image (periodic remainder)
            homeZ = np.remainder(positions[:, 2], GS.box_z)
            homeY = np.remainder(positions[:, 1], GS.box_y)
            homeX = np.remainder(positions[:, 0], GS.box_x)

            # Component forces (scalar arrays for vectorized deposition)
            fox = forces[:, 0]
            foy = forces[:, 1]
            foz = forces[:, 2]

            # Map to voxel indices (np.digitize returns 1..len(bins)-1)
            x = np.digitize(homeX, GS.binsx)
            y = np.digitize(homeY, GS.binsy)
            z = np.digitize(homeZ, GS.binsz)

            if kernel.lower() == "triangular":
                _triangular_allocation(
                    GS.forceX, GS.forceY, GS.forceZ, GS.counter,
                    x, y, z, homeX, homeY, homeZ,
                    fox, foy, foz, a,
                    GS.lx, GS.ly, GS.lz,
                    GS.nbinsx, GS.nbinsy, GS.nbinsz,
                )
            elif kernel.lower() == "box":
                # Convert to 0-based indices for box allocation
                _box_allocation(
                    GS.forceX, GS.forceY, GS.forceZ, GS.counter,
                    x - 1, y - 1, z - 1,
                    fox, foy, foz, a,
                )
            else:
                raise ValueError(f"Unsupported kernel: {kernel!r}")

        @staticmethod
        def box_allocation(GS: GridState, x: np.ndarray, y: np.ndarray, z: np.ndarray, fox: np.ndarray, foy: np.ndarray, foz: np.ndarray, a: float | np.ndarray) -> None:
            """
            Deposit contributions to the host voxel (no neighbour spreading).

            This method delegates to the backend allocation function selected
            at module import time. Uses np.add.at() or Numba JIT to correctly
            handle overlapping particles.

            Parameters
            ----------
            GS : GridState
                Grid state object with forceX, forceY, forceZ, counter arrays.
            x, y, z : np.ndarray
                Voxel indices (1-based from np.digitize).
            fox, foy, foz : np.ndarray
                Force components for each particle.
            a : float or np.ndarray
                Weight factor (scalar or per-particle array).
            """
            # Convert to 0-based voxel indices
            _box_allocation(
                GS.forceX, GS.forceY, GS.forceZ, GS.counter,
                x - 1, y - 1, z - 1,
                fox, foy, foz, a,
            )

        @staticmethod
        def triangular_allocation(
            GS: GridState,
            x: np.ndarray,
            y: np.ndarray,
            z: np.ndarray,
            homeX: np.ndarray,
            homeY: np.ndarray,
            homeZ: np.ndarray,
            fox: np.ndarray,
            foy: np.ndarray,
            foz: np.ndarray,
            a: float | np.ndarray,
        ) -> None:
            """
            Deposit contributions to the 8 neighbouring voxel vertices (CIC/triangular).

            This method delegates to the backend allocation function selected
            at module import time. Uses np.add.at() or Numba JIT to correctly
            handle overlapping particles.

            Parameters
            ----------
            GS : GridState
                Grid state object with forceX, forceY, forceZ, counter arrays
                and grid parameters (lx, ly, lz, nbinsx, nbinsy, nbinsz).
            x, y, z : np.ndarray
                Voxel indices (1-based from np.digitize).
            homeX, homeY, homeZ : np.ndarray
                Actual particle positions.
            fox, foy, foz : np.ndarray
                Force components for each particle.
            a : float or np.ndarray
                Weight factor (scalar or per-particle array).
            """
            _triangular_allocation(
                GS.forceX, GS.forceY, GS.forceZ, GS.counter,
                x, y, z, homeX, homeY, homeZ,
                fox, foy, foz, a,
                GS.lx, GS.ly, GS.lz,
                GS.nbinsx, GS.nbinsy, GS.nbinsz,
            )


        @staticmethod
        def find_coms(positions: np.ndarray, trajectory: Trajectory, GS: GridState, SS: SelectionState, calc_dipoles: bool = False):
            """
            Compute centers-of-mass (and optionally molecular dipoles) for a rigid set.

            Parameters
            ----------
            positions : (N, 3) np.ndarray
                Cartesian coordinates.
            trajectory : Trajectory
                Trajectory state containing box lengths.
            GS : GridState
                Unused numerically here; preserved for signature compatibility.
            SS : SelectionState
                Provides `indices`, `masses`, and `charges` (if available).
            calc_dipoles : bool, optional
                If True, also compute per-molecule dipole moments relative to COMs.

            Returns
            -------
            coms : (M, 3) np.ndarray
                Center-of-mass coordinates for M molecules.
            molecular_dipole : (M, 3) np.ndarray, optional
                Returned if `calc_dipoles=True`.

            Notes
            -----
            Enforces minimum-image displacements when aligning species to a reference
            for COM and dipole accumulation.
            """
            mass_tot = SS.masses[0]
            mass_cumulant = positions[SS.indices[0]] * SS.masses[0][:, np.newaxis]
            for species_index in range(1, len(SS.indices)):
                diffs = positions[SS.indices[0]] - positions[SS.indices[species_index]]
                logical_diffs = np.transpose(
                    np.array(
                        [
                            trajectory.box_x * (diffs[:, 0] < -trajectory.box_x / 2) - trajectory.box_x * (diffs[:, 0] > trajectory.box_x / 2),
                            trajectory.box_y * (diffs[:, 1] < -trajectory.box_y / 2) - trajectory.box_y * (diffs[:, 1] > trajectory.box_y / 2),
                            trajectory.box_z * (diffs[:, 2] < -trajectory.box_z / 2) - trajectory.box_z * (diffs[:, 2] > trajectory.box_z / 2),
                        ]
                    )
                )
                diffs += logical_diffs
                mass_tot += SS.masses[species_index]
                mass_cumulant += positions[SS.indices[species_index]] * SS.masses[species_index][:, np.newaxis]
            coms = mass_cumulant / mass_tot[:, np.newaxis]

            if calc_dipoles:
                charges_cumulant = GS.SS.charges[0][:, np.newaxis] * (positions[SS.indices[0]] - coms)
                for species_index in range(1, len(SS.indices)):
                    separation = (positions[SS.indices[species_index]] - coms)
                    # Minimum-image correction component-wise
                    separation[:, 0] -= (np.ceil((np.abs(separation[:, 0]) - trajectory.box_x / 2) / trajectory.box_x)) * (trajectory.box_x) * np.sign(separation[:, 0])
                    separation[:, 1] -= (np.ceil((np.abs(separation[:, 1]) - trajectory.box_y / 2) / trajectory.box_y)) * (trajectory.box_y) * np.sign(separation[:, 1])
                    separation[:, 2] -= (np.ceil((np.abs(separation[:, 2]) - trajectory.box_z / 2) / trajectory.box_z)) * (trajectory.box_z) * np.sign(separation[:, 2])
                    charges_cumulant += GS.SS.charges[species_index][:, np.newaxis] * separation
                molecular_dipole = charges_cumulant
                return coms, molecular_dipole
            else:
                return coms

        @staticmethod
        def sum_forces(SS: SelectionState, forces: np.ndarray) -> np.ndarray:
            """
            Sum forces across a rigid group (species lists).

            Parameters
            ----------
            SS : SelectionState
                Selection containing per-species atom index arrays.
            forces : (N, 3) np.ndarray
                Force array for all atoms.

            Returns
            -------
            (M, 3) np.ndarray
                Per-molecule net force for the rigid group (M = multiplicity).
            """
            rigid_forces = forces[SS.indices[0], :]
            for rigid_body_component in SS.indices[1:]:
                rigid_forces += forces[rigid_body_component, :]
            return rigid_forces

