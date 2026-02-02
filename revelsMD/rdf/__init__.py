"""
Radial distribution function (RDF) estimators for RevelsMD.

This package provides force-weighted RDF calculations:

- RDF: RDF calculator class (mirrors DensityGrid pattern)
- compute_rdf: Convenience function for one-liner RDF computation

Preferred imports::

    from revelsMD.rdf import RDF, compute_rdf

    rdf = compute_rdf(trajectory, 'O', 'H', integration='forward')
    print(rdf.r, rdf.g)
"""

from revelsMD.rdf.rdf import RDF, compute_rdf

__all__ = [
    "RDF",
    "compute_rdf",
]
