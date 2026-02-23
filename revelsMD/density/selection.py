"""Selection class for atom selection and center choice."""

from __future__ import annotations

import numpy as np

from revelsMD.cell import apply_minimum_image
from revelsMD.trajectories._base import Trajectory
from revelsMD.density.constants import validate_density_type


class Selection:
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
    rigid : bool
        Whether to treat the selection as a rigid molecule.
    density_type : {'number', 'charge', 'polarisation'}
        Type of density. Data requirements:
        - 'number': No charge/mass data required (but masses needed if rigid with COM)
        - 'charge': Trajectory must provide charges
        - 'polarisation': Trajectory must provide charges and masses
    polarisation_axis : int
        Axis for polarisation projection (0=x, 1=y, 2=z).

    Attributes
    ----------
    single_species : bool
        True if selection contains a single species (all atoms interchangeable).
    indices : np.ndarray or list of np.ndarray
        Atom indices of the selection (kept name for compatibility).
    charges, masses : list or np.ndarray
        Per-species arrays (rigid) or single array (single species). Only
        populated when required by density_type.
    """

    # Type declarations (union types until normalisation refactor)
    indices: np.ndarray | list[np.ndarray]
    charges: np.ndarray | list[np.ndarray]
    masses: np.ndarray | list[np.ndarray]

    @property
    def single_species(self) -> bool:
        """True if selection contains a single species (all atoms interchangeable)."""
        return not isinstance(self.indices, list)

    def __init__(
        self,
        trajectory: Trajectory,
        atom_names: str | list[str],
        centre_location: bool | int,
        rigid: bool = False,
        density_type: str = 'number',
        polarisation_axis: int = 0,
    ):
        self.rigid = rigid
        self.density_type = validate_density_type(density_type)
        self.polarisation_axis = polarisation_axis
        self._cell_matrix = np.array(trajectory.cell_matrix, dtype=np.float64)
        self._cell_inverse = np.linalg.inv(self._cell_matrix)

        # Determine what data we need
        needs_charges = density_type in ('charge', 'polarisation')
        needs_masses = density_type == 'polarisation' or (rigid and centre_location is True)

        if isinstance(atom_names, list) and len(atom_names) > 1:
            self.indices: list[np.ndarray] = []
            for atom in atom_names:
                self.indices.append(trajectory.get_indices(atom))

            if needs_charges:
                self.charges: list[np.ndarray] = []
                for atom in atom_names:
                    self.charges.append(trajectory.get_charges(atom))

            if needs_masses:
                self.masses: list[np.ndarray] = []
                for atom in atom_names:
                    self.masses.append(trajectory.get_masses(atom))

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
            self.indices = trajectory.get_indices(atom_names)

            if needs_charges:
                self.charges = trajectory.get_charges(atom_names)

            if needs_masses:
                self.masses = trajectory.get_masses(atom_names)

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
        if self.single_species:
            return positions[self.indices, :]

        if not self.rigid:
            return [positions[idx, :] for idx in self.indices]

        if self.centre_location is True:
            return self._compute_com(positions)
        return positions[self.indices[self.centre_location], :]

    def _compute_com(self, positions: np.ndarray) -> np.ndarray:
        """
        Compute center-of-mass positions for rigid molecules.

        Uses minimum image convention to handle molecules spanning periodic boundaries.

        Parameters
        ----------
        positions : (N, 3) np.ndarray
            Full frame positions.

        Returns
        -------
        (M, 3) np.ndarray
            Center-of-mass positions for M molecules.
        """
        ref_positions = positions[self.indices[0]]
        mass_tot = self.masses[0].copy()
        mass_cumulant = ref_positions * self.masses[0][:, np.newaxis]

        for species_idx in range(1, len(self.indices)):
            species_positions = positions[self.indices[species_idx]]
            species_mass = self.masses[species_idx]

            # Apply minimum image convention relative to reference species
            diff = ref_positions - species_positions
            mic_diff = apply_minimum_image(diff, self._cell_matrix, self._cell_inverse)
            species_positions_unwrapped = ref_positions - mic_diff

            mass_tot = mass_tot + species_mass
            mass_cumulant = mass_cumulant + species_positions_unwrapped * species_mass[:, np.newaxis]

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
        if self.single_species:
            return forces[self.indices, :]

        if not self.rigid:
            return [forces[idx, :] for idx in self.indices]

        # Sum forces across molecule
        result = forces[self.indices[0], :].copy()
        for species_idx in range(1, len(self.indices)):
            result = result + forces[self.indices[species_idx], :]
        return result

    def extract(
        self,
        positions: np.ndarray,
        forces: np.ndarray,
    ) -> tuple[
        np.ndarray | list[np.ndarray],
        np.ndarray | list[np.ndarray],
        float | np.ndarray | list[np.ndarray],
    ]:
        """
        Extract deposit positions, forces, and weights in a single call.

        Parameters
        ----------
        positions : (N, 3) np.ndarray
            Full frame positions for all atoms.
        forces : (N, 3) np.ndarray
            Full frame forces for all atoms.

        Returns
        -------
        tuple of (positions, forces, weights)
            Ready for passing to grid.deposit().
        """
        return (
            self.get_positions(positions),
            self.get_forces(forces),
            self.get_weights(positions),
        )


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
        match self.density_type:
            case 'number':
                return 1.0

            case 'charge':
                if self.single_species or not self.rigid:
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
                raise ValueError(f"Unknown density_type: {self.density_type!r}")

    def _compute_dipole_projection(self, positions: np.ndarray) -> np.ndarray:
        """
        Compute molecular dipole moment projected along polarisation_axis.

        Uses minimum image convention to handle molecules spanning periodic boundaries.

        Parameters
        ----------
        positions : (N, 3) np.ndarray
            Full frame positions.

        Returns
        -------
        (M,) np.ndarray
            Dipole projection for each molecule.
        """
        coms = self._compute_com(positions)

        dipole = np.zeros((coms.shape[0], 3))
        for species_idx in range(len(self.indices)):
            species_positions = positions[self.indices[species_idx]]
            species_charges = self.charges[species_idx]

            # Apply minimum image convention for displacement from COM
            displacement = species_positions - coms
            displacement = apply_minimum_image(displacement, self._cell_matrix, self._cell_inverse)

            dipole = dipole + species_charges[:, np.newaxis] * displacement

        return dipole[:, self.polarisation_axis]
