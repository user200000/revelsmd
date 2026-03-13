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
"""

from ._base import DataUnavailableError
from .lammps import LammpsTrajectory
from .mda import MDATrajectory
from .numpy import NumpyTrajectory
from .vasp import VaspTrajectory


__all__ = [
    "LammpsTrajectory",
    "MDATrajectory",
    "NumpyTrajectory",
    "VaspTrajectory",
    "DataUnavailableError",
]
