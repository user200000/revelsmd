"""
Trajectory handling package for RevelsMD.

This package provides classes for reading molecular dynamics trajectories
from various file formats and data sources.

Classes
-------
MDATrajectory
    MDAnalysis-based trajectory reader.
NumpyTrajectory
    In-memory NumPy array trajectory.
LammpsTrajectory
    LAMMPS dump file trajectory reader.
VaspTrajectory
    VASP vasprun.xml trajectory reader.

Exceptions
----------
DataUnavailableError
    Raised when requested data is not available for a trajectory type.

Deprecated Aliases
------------------
The following names are deprecated and will be removed in a future version:
- MDATrajectoryState -> MDATrajectory
- NumpyTrajectoryState -> NumpyTrajectory
- LammpsTrajectoryState -> LammpsTrajectory
- VaspTrajectoryState -> VaspTrajectory
"""

import warnings

from ._base import DataUnavailableError
from .lammps import LammpsTrajectory
from .mda import MDATrajectory
from .numpy import NumpyTrajectory
from .vasp import VaspTrajectory


# -----------------------------------------------------------------------------
# Deprecated Aliases
# -----------------------------------------------------------------------------

def _make_deprecated_alias(new_class, old_name, new_name):
    """Create a subclass that emits a deprecation warning on instantiation."""
    class DeprecatedAlias(new_class):
        def __init__(self, *args, **kwargs):
            warnings.warn(
                f"{old_name} is deprecated. Use {new_name} instead.",
                DeprecationWarning,
                stacklevel=2
            )
            super().__init__(*args, **kwargs)
    DeprecatedAlias.__name__ = old_name
    DeprecatedAlias.__qualname__ = old_name
    return DeprecatedAlias


# Deprecated class name aliases
MDATrajectoryState = _make_deprecated_alias(MDATrajectory, "MDATrajectoryState", "MDATrajectory")
NumpyTrajectoryState = _make_deprecated_alias(NumpyTrajectory, "NumpyTrajectoryState", "NumpyTrajectory")
LammpsTrajectoryState = _make_deprecated_alias(LammpsTrajectory, "LammpsTrajectoryState", "LammpsTrajectory")
VaspTrajectoryState = _make_deprecated_alias(VaspTrajectory, "VaspTrajectoryState", "VaspTrajectory")


__all__ = [
    # Trajectory classes
    "LammpsTrajectory",
    "MDATrajectory",
    "NumpyTrajectory",
    "VaspTrajectory",
    # Exception
    "DataUnavailableError",
    # Deprecated aliases
    "LammpsTrajectoryState",
    "MDATrajectoryState",
    "NumpyTrajectoryState",
    "VaspTrajectoryState",
]
