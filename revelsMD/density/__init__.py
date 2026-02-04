"""
Density estimation classes for RevelsMD.

This package provides classes for 3D force-density estimation:
- Selection: Atom selection, charges/masses, and center choice
- DensityGrid: Grid for accumulating 3D force fields
- compute_density: Convenience function for computing density in one call
"""

from revelsMD.density.selection import Selection
from revelsMD.density.density_grid import DensityGrid, compute_density

__all__ = ["Selection", "DensityGrid", "compute_density"]
