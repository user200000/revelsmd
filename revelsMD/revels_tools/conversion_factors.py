"""
Unit conversion utilities for RevelsMD.

This module provides physical constants and simple conversion helpers
used to translate between different molecular dynamics unit systems.

Notes
-----
- The function :func:`generate_boltzmann` supports several common unit
  systems corresponding to LAMMPS and MDAnalysis conventions.
- Uses values from :mod:`scipy.constants`.
"""

import scipy.constants as constants


def generate_boltzmann(units: str) -> float:
    """
    Return the Boltzmann constant in the specified unit system.

    Parameters
    ----------
    units : str
        The desired unit system. Supported values are:

        - ``'lj'`` : Lennard-Jones reduced units (returns 1.0)
        - ``'real'`` : kcal·mol⁻¹·K⁻¹ (LAMMPS real units)
        - ``'metal'`` : eV·K⁻¹ (LAMMPS metal units)
        - ``'mda'`` : kJ·mol⁻¹·K⁻¹ (MDAnalysis units)

    Returns
    -------
    float
        Boltzmann constant in the requested unit system.

    Raises
    ------
    ValueError
        If the provided unit system is not recognized.

    Examples
    --------
    >>> from revelsMD.revels_tools.conversion_factors import generate_boltzmann
    >>> generate_boltzmann('metal')
    8.617333262145e-05
    >>> generate_boltzmann('lj')
    1.0
    """
    units = units.lower().strip()

    if units == "lj":
        return 1.0
    elif units == "real":
        # kcal·mol⁻¹·K⁻¹ = R / (calorie * 1000)
        return constants.physical_constants["molar gas constant"][0] / constants.calorie / 1000
    elif units == "metal":
        # eV·K⁻¹
        return constants.physical_constants["Boltzmann constant in eV/K"][0]
    elif units == "mda":
        # kJ·mol⁻¹·K⁻¹
        return constants.physical_constants["molar gas constant"][0] / 1000
    else:
        raise ValueError(
            f"Unsupported unit system: '{units}'. Expected one of ['lj', 'real', 'metal', 'mda']."
        )

