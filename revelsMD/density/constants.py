"""Constants for the density package."""

VALID_DENSITY_TYPES = ('number', 'charge', 'polarisation')


def validate_density_type(density_type: str) -> str:
    """Validate and normalise density_type.

    Parameters
    ----------
    density_type : str
        The density type to validate.

    Returns
    -------
    str
        The normalised density type (lowercase, stripped).

    Raises
    ------
    ValueError
        If density_type is not one of the valid types.
    """
    normalised = density_type.lower().strip()
    if normalised not in VALID_DENSITY_TYPES:
        raise ValueError(
            f"density_type must be one of {VALID_DENSITY_TYPES}, got {density_type!r}"
        )
    return normalised
