"""
Force-based radial distribution function (RDF) estimators for RevelsMD.

This module provides backward-compatible access to RDF functions.
The functions have been moved to the revelsMD.rdf package.

Preferred imports::

    from revelsMD.rdf import run_rdf, run_rdf_lambda, single_frame_rdf

Deprecated (but still works)::

    from revelsMD.revels_rdf import RevelsRDF
    RevelsRDF.run_rdf(...)
"""

from __future__ import annotations

import warnings

from revelsMD.rdf import run_rdf, run_rdf_lambda, single_frame_rdf


class _DeprecatedMethodDescriptor:
    """Descriptor that emits a deprecation warning when accessing a method."""

    def __init__(self, func, method_name: str):
        self.func = func
        self.method_name = method_name

    def __get__(self, obj, objtype=None):
        warnings.warn(
            f"RevelsRDF.{self.method_name} is deprecated. "
            f"Use 'from revelsMD.rdf import {self.method_name}' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.func


class RevelsRDF:
    """
    Force-weighted radial distribution function (RDF) estimators.

    DEPRECATED: This class is deprecated. Use the revelsMD.rdf module directly::

        from revelsMD.rdf import run_rdf, run_rdf_lambda, single_frame_rdf

    The static methods are now module-level functions in revelsMD.rdf.
    """

    # Deprecated method aliases - use revelsMD.rdf imports instead
    single_frame_rdf = _DeprecatedMethodDescriptor(single_frame_rdf, "single_frame_rdf")
    run_rdf = _DeprecatedMethodDescriptor(run_rdf, "run_rdf")
    run_rdf_lambda = _DeprecatedMethodDescriptor(run_rdf_lambda, "run_rdf_lambda")
