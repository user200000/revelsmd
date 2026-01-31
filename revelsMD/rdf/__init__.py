"""
Radial distribution function (RDF) estimators for RevelsMD.

This package provides force-weighted RDF calculations:

Class-based API (recommended):
- RDF: Stateful RDF calculator (mirrors DensityGrid pattern)
- compute_rdf: Convenience function for one-liner RDF computation

Legacy function API:
- run_rdf: Multi-frame RDF calculation
- run_rdf_lambda: Variance-minimised lambda-corrected RDF
- single_frame_rdf: Single-frame RDF calculation

Preferred imports::

    from revelsMD.rdf import RDF, compute_rdf

    rdf = compute_rdf(trajectory, 'O', 'H', integration='forward')
    print(rdf.r, rdf.g)
"""

from revelsMD.rdf.rdf import run_rdf, run_rdf_lambda, single_frame_rdf
from revelsMD.rdf.rdf_class import RDF, compute_rdf

__all__ = [
    "RDF",
    "compute_rdf",
    "run_rdf",
    "run_rdf_lambda",
    "single_frame_rdf",
]
