"""
Density estimation classes for RevelsMD.

This package provides classes for 3D force-density estimation:
- SelectionState: Atom selection, charges/masses, and center choice
- GridState: State for accumulating 3D force fields
"""

from revelsMD.density.selection_state import SelectionState
from revelsMD.density.grid_state import GridState

__all__ = ["SelectionState", "GridState"]
