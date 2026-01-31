"""
3D force-density estimators and utilities for RevelsMD.

This module provides backward-compatible access to density estimation classes.
The classes have been moved to revelsMD.density package.

Preferred imports:
    from revelsMD.density import DensityGrid, Selection

Deprecated (but still works):
    from revelsMD.revels_3D import Revels3D
    Revels3D.GridState, Revels3D.SelectionState
"""

from __future__ import annotations

import warnings

from revelsMD.density import Selection, DensityGrid

# Mapping from old names to new names for deprecation messages
_NAME_MAPPING = {
    "GridState": "DensityGrid",
    "SelectionState": "Selection",
}


class _DeprecatedClassDescriptor:
    """Descriptor that emits a deprecation warning when accessing a class attribute."""

    def __init__(self, cls: type, old_name: str):
        self.cls = cls
        self.old_name = old_name
        self.new_name = _NAME_MAPPING.get(old_name, old_name)

    def __get__(self, obj, objtype=None):
        warnings.warn(
            f"Revels3D.{self.old_name} is deprecated. "
            f"Use 'from revelsMD.density import {self.new_name}' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.cls


class Revels3D:
    """
    Namespace wrapper for grid building and selection state.

    The nested classes are deprecated. Use revelsMD.density imports instead:
    - `DensityGrid`: use `from revelsMD.density import DensityGrid`
    - `Selection`: use `from revelsMD.density import Selection`
    """

    # Deprecated aliases - use revelsMD.density imports instead
    GridState = _DeprecatedClassDescriptor(DensityGrid, "GridState")
    SelectionState = _DeprecatedClassDescriptor(Selection, "SelectionState")
