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
