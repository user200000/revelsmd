"""
Backend configuration for RevelsMD.

This module provides unified backend selection for all numerically intensive
operations (RDF calculations, grid allocation, etc.).

The backend can be configured via the REVELSMD_BACKEND environment variable:
- 'numba': JIT-compiled implementations (default, requires numba)
- 'numpy': Pure NumPy implementations

FFT parallelism can be configured via REVELSMD_FFT_WORKERS:
- 1: single-threaded (default)
- N: use N threads
- -1: use all available cores

Example
-------
>>> import os
>>> os.environ['REVELSMD_BACKEND'] = 'numpy'  # Before importing revelsMD
>>> os.environ['REVELSMD_FFT_WORKERS'] = '4'  # Use 4 threads for FFTs
"""

from __future__ import annotations

import os

BACKEND_ENV_VAR = 'REVELSMD_BACKEND'
AVAILABLE_BACKENDS = frozenset({'numpy', 'numba'})
DEFAULT_BACKEND = 'numba'

FFT_WORKERS_ENV_VAR = 'REVELSMD_FFT_WORKERS'
DEFAULT_FFT_WORKERS = 1


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


def _resolve_fft_workers() -> int:
    """Resolve FFT worker count from environment variable.

    Returns
    -------
    int
        Number of FFT worker threads. 1 for single-threaded,
        -1 for all available cores.

    Raises
    ------
    ValueError
        If the environment variable is not a valid integer.
    """
    value = os.environ.get(FFT_WORKERS_ENV_VAR, '')
    if not value:
        return DEFAULT_FFT_WORKERS
    try:
        workers = int(value)
    except ValueError:
        raise ValueError(
            f"Invalid {FFT_WORKERS_ENV_VAR} '{value}'. Must be an integer."
        )
    if workers == 0:
        raise ValueError(
            f"Invalid {FFT_WORKERS_ENV_VAR} '{value}'. Must be non-zero."
        )
    return workers


BACKEND = _resolve_backend()
FFT_WORKERS = _resolve_fft_workers()


def get_backend() -> str:
    """
    Get the current backend.

    Returns
    -------
    str
        Backend name ('numpy' or 'numba').
    """
    return BACKEND


def get_fft_workers() -> int:
    """
    Get the number of FFT worker threads.

    Returns
    -------
    int
        Number of worker threads. 1 for single-threaded,
        -1 for all available cores.
    """
    return FFT_WORKERS
