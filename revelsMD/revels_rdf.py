"""
Force-based radial distribution function (RDF) estimators for RevelsMD.

This module provides backward-compatible access to RDF functions.
The functions have been moved to the revelsMD.rdf package.

Preferred imports::

    from revelsMD.rdf import RDF, compute_rdf

Deprecated (but still works)::

    from revelsMD.revels_rdf import RevelsRDF
    RevelsRDF.run_rdf(...)
"""

from __future__ import annotations

import warnings

import numpy as np

from revelsMD.rdf import RDF, compute_rdf


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

    DEPRECATED: Use compute_rdf() instead.

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
    """
    # Convert rmax parameter
    if rmax is True:
        rmax_value = None  # RDF class will compute default
    else:
        rmax_value = float(rmax)

    # Use RDF class
    rdf = RDF(trajectory, atom_a, atom_b, delr=delr, rmax=rmax_value)
    rdf.accumulate(trajectory, start=start, stop=stop, period=period)

    integration = 'forward' if from_zero else 'backward'
    rdf.get_rdf(integration=integration)

    return np.array([rdf.r, rdf.g])


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
    Compute the lambda-corrected RDF by combining forward and backward estimates.

    DEPRECATED: Use compute_rdf(integration='lambda') instead.

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
    period : int, optional
        Frame stride (default: 1).
    rmax : bool or float, optional
        If True, use half the minimum box dimension; otherwise, set numeric cutoff.

    Returns
    -------
    numpy.ndarray of shape (n, 3)
        Columns: `[r, g_lambda(r), lambda(r)]`.
    """
    # Convert rmax parameter
    if rmax is True:
        rmax_value = None  # RDF class will compute default
    else:
        rmax_value = float(rmax)

    # Use RDF class
    rdf = RDF(trajectory, atom_a, atom_b, delr=delr, rmax=rmax_value)
    rdf.accumulate(trajectory, start=start, stop=stop, period=period)
    rdf.get_rdf(integration='lambda')

    return np.transpose(np.array([rdf.r, rdf.g, rdf.lam]))


class _DeprecatedMethodDescriptor:
    """Descriptor that emits a deprecation warning when accessing a method."""

    def __init__(self, func, method_name: str):
        self.func = func
        self.method_name = method_name

    def __get__(self, obj, objtype=None):
        warnings.warn(
            f"RevelsRDF.{self.method_name} is deprecated. "
            f"Use 'from revelsMD.rdf import compute_rdf' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.func


class RevelsRDF:
    """
    Force-weighted radial distribution function (RDF) estimators.

    DEPRECATED: This class is deprecated. Use the revelsMD.rdf module directly::

        from revelsMD.rdf import RDF, compute_rdf

    """

    # Deprecated method aliases
    run_rdf = _DeprecatedMethodDescriptor(run_rdf, "run_rdf")
    run_rdf_lambda = _DeprecatedMethodDescriptor(run_rdf_lambda, "run_rdf_lambda")
