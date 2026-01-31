"""
Density estimation classes for RevelsMD.

This package provides classes for 3D force-density estimation:
- Selection: Atom selection, charges/masses, and center choice
- DensityGrid: Grid for accumulating 3D force fields
"""

from revelsMD.density.selection_state import Selection
from revelsMD.density.grid_state import DensityGrid

__all__ = ["Selection", "DensityGrid"]
