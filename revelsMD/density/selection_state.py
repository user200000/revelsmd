"""SelectionState class for atom selection and center choice."""

from __future__ import annotations

import numpy as np

from revelsMD.trajectories._base import Trajectory, DataUnavailableError


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
        self.rigid = rigid
        self._trajectory = trajectory  # Keep reference for box dimensions in COM calculation

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

    def get_positions(self, positions: np.ndarray) -> np.ndarray | list[np.ndarray]:
        """
        Extract deposit positions from a frame based on selection configuration.

        Parameters
        ----------
        positions : (N, 3) np.ndarray
            Full frame positions for all atoms.

        Returns
        -------
        np.ndarray or list of np.ndarray
            - Single species: (M, 3) array of selected atom positions
            - Multi-species, not rigid: list of (M, 3) arrays per species
            - Rigid, COM: (M, 3) array of center-of-mass positions
            - Rigid, specific atom: (M, 3) array of that atom's positions
        """
        if self.indistinguishable_set:
            return positions[self.indices, :]

        if not self.rigid:
            return [positions[idx, :] for idx in self.indices]

        if self.centre_location is True:
            return self._compute_com(positions)
        return positions[self.indices[self.centre_location], :]

    def _compute_com(self, positions: np.ndarray) -> np.ndarray:
        """
        Compute center-of-mass positions for rigid molecules.

        Parameters
        ----------
        positions : (N, 3) np.ndarray
            Full frame positions.

        Returns
        -------
        (M, 3) np.ndarray
            Center-of-mass positions for M molecules.
        """
        mass_tot = self.masses[0].copy()
        mass_cumulant = positions[self.indices[0]] * self.masses[0][:, np.newaxis]

        for species_idx in range(1, len(self.indices)):
            species_mass = self.masses[species_idx]
            mass_tot = mass_tot + species_mass
            mass_cumulant = mass_cumulant + positions[self.indices[species_idx]] * species_mass[:, np.newaxis]

        return mass_cumulant / mass_tot[:, np.newaxis]

    def get_forces(self, forces: np.ndarray) -> np.ndarray | list[np.ndarray]:
        """
        Extract deposit forces from a frame based on selection configuration.

        Parameters
        ----------
        forces : (N, 3) np.ndarray
            Full frame forces for all atoms.

        Returns
        -------
        np.ndarray or list of np.ndarray
            - Single species: (M, 3) array of selected atom forces
            - Multi-species, not rigid: list of (M, 3) arrays per species
            - Rigid: (M, 3) array of summed forces per molecule
        """
        if self.indistinguishable_set:
            return forces[self.indices, :]

        if not self.rigid:
            return [forces[idx, :] for idx in self.indices]

        # Sum forces across molecule
        result = forces[self.indices[0], :].copy()
        for species_idx in range(1, len(self.indices)):
            result = result + forces[self.indices[species_idx], :]
        return result

    def get_weights(self, positions: np.ndarray | None = None) -> float | np.ndarray | list[np.ndarray]:
        """
        Get deposit weights based on density type.

        Parameters
        ----------
        positions : (N, 3) np.ndarray, optional
            Full frame positions. Required for polarisation density.

        Returns
        -------
        float, np.ndarray, or list of np.ndarray
            - Number density: 1.0
            - Charge density: charges (array or list of arrays)
            - Polarisation density: dipole projection along polarisation_axis
        """
        density_type = getattr(self, 'density_type', 'number')

        match density_type:
            case 'number':
                return 1.0

            case 'charge':
                if self.indistinguishable_set or not self.rigid:
                    return self.charges
                total_charge = self.charges[0].copy()
                for species_idx in range(1, len(self.charges)):
                    total_charge = total_charge + self.charges[species_idx]
                return total_charge

            case 'polarisation':
                if positions is None:
                    raise ValueError("positions required for polarisation density")
                return self._compute_dipole_projection(positions)

            case _:
                raise ValueError(f"Unknown density_type: {density_type!r}")

    def _compute_dipole_projection(self, positions: np.ndarray) -> np.ndarray:
        """
        Compute molecular dipole moment projected along polarisation_axis.

        Parameters
        ----------
        positions : (N, 3) np.ndarray
            Full frame positions.

        Returns
        -------
        (M,) np.ndarray
            Dipole projection for each molecule.
        """
        # Compute COM first
        coms = self._compute_com(positions)

        # Compute dipole: sum of q_i * (r_i - COM)
        dipole = np.zeros((coms.shape[0], 3))
        for species_idx in range(len(self.indices)):
            species_positions = positions[self.indices[species_idx]]
            species_charges = self.charges[species_idx]
            displacement = species_positions - coms
            dipole = dipole + species_charges[:, np.newaxis] * displacement

        return dipole[:, self.polarisation_axis]
