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


def _resolve_backend() -> str:
    """Resolve and validate backend from environment variable.

    Called once at module import time to ensure the backend is valid.

    Returns
    -------
    str
        Validated backend name ('numpy' or 'numba').

    Raises
    ------
    ValueError
        If the environment variable contains an invalid backend name.
    """
    value = os.environ.get(BACKEND_ENV_VAR, DEFAULT_BACKEND).lower().strip()
    if not value:
        return DEFAULT_BACKEND
    if value not in AVAILABLE_BACKENDS:
        raise ValueError(
            f"Invalid REVELSMD_BACKEND '{value}'. "
            f"Must be one of: {', '.join(sorted(AVAILABLE_BACKENDS))}"
        )
    return value


BACKEND = _resolve_backend()


def get_backend() -> str:
    """
    Get the current backend.

    Returns
    -------
    str
        Backend name ('numpy' or 'numba').
    """
    return BACKEND
