"""
3D force-density estimators and utilities for RevelsMD.

This module provides backward-compatible access to density estimation classes.
The classes have been moved to revelsMD.density package.

Preferred imports:
    from revelsMD.density import GridState, Selection

Deprecated (but still works):
    from revelsMD.revels_3D import Revels3D
    Revels3D.GridState, Revels3D.SelectionState
"""

from __future__ import annotations

import warnings

from revelsMD.density import Selection, GridState


class _DeprecatedClassDescriptor:
    """Descriptor that emits a deprecation warning when accessing a class attribute."""

    def __init__(self, cls: type, name: str):
        self.cls = cls
        self.name = name

    def __get__(self, obj, objtype=None):
        warnings.warn(
            f"Revels3D.{self.name} is deprecated. "
            f"Use 'from revelsMD.density import {self.name}' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.cls


class Revels3D:
    """
    Namespace wrapper for grid building and selection state.

    The nested classes are deprecated. Use revelsMD.density imports instead:
    - `GridState`: use `from revelsMD.density import GridState`
    - `Selection`: use `from revelsMD.density import Selection`
    """

    # Deprecated aliases - use revelsMD.density imports instead
    GridState = _DeprecatedClassDescriptor(GridState, "GridState")
    SelectionState = _DeprecatedClassDescriptor(Selection, "SelectionState")
