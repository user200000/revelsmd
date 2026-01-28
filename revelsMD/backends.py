"""
Backend configuration for RevelsMD.

This module provides unified backend selection for all numerically intensive
operations (RDF calculations, grid allocation, etc.).

The backend can be configured via the REVELSMD_BACKEND environment variable:
- 'numba': JIT-compiled implementations (default, requires numba)
- 'numpy': Pure NumPy implementations

Example
-------
>>> import os
>>> os.environ['REVELSMD_BACKEND'] = 'numpy'  # Before importing revelsMD
"""

from __future__ import annotations

import os

BACKEND_ENV_VAR = 'REVELSMD_BACKEND'
AVAILABLE_BACKENDS = frozenset({'numpy', 'numba'})
DEFAULT_BACKEND = 'numba'


def get_backend() -> str:
    """
    Get the current backend from environment variable.

    Returns
    -------
    str
        Backend name ('numpy' or 'numba').
    """
    return os.environ.get(BACKEND_ENV_VAR, DEFAULT_BACKEND).lower()
