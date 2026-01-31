"""
Radial distribution function (RDF) estimators for RevelsMD.

This package provides force-weighted RDF calculations:

- run_rdf: Multi-frame RDF calculation
- run_rdf_lambda: Variance-minimised lambda-corrected RDF
- single_frame_rdf: Single-frame RDF calculation

Preferred imports::

    from revelsMD.rdf import run_rdf, run_rdf_lambda, single_frame_rdf
"""

from revelsMD.rdf.rdf import run_rdf, run_rdf_lambda, single_frame_rdf

__all__ = ["run_rdf", "run_rdf_lambda", "single_frame_rdf"]
