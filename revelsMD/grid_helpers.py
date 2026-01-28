"""
Grid allocation helper functions for depositing particle contributions to voxel grids.

This module provides backend-agnostic functions for Cloud-in-Cell (CIC/triangular)
and box kernel grid deposition. Two backends are available:
- NumPy: Uses np.add.at() for correct accumulation
- Numba: JIT-compiled for performance (default)

The NumPy implementation uses np.add.at() to correctly handle the case where
multiple particles deposit to the same voxel indices. Standard NumPy fancy
indexing with += only keeps the last value when indices contain duplicates.

Backend selection is controlled by the REVELSMD_BACKEND environment variable.
See `revelsMD.backends` for configuration details.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from revelsMD.backends import get_backend, AVAILABLE_BACKENDS


def _get_numba_functions() -> tuple[Callable, Callable]:
    """Import and return Numba backend functions."""
    try:
        from revelsMD.grid_helpers_numba import (
            triangular_allocation_numba,
            box_allocation_numba,
        )
        return triangular_allocation_numba, box_allocation_numba
    except ImportError as e:
        raise ImportError(
            "Numba backend requested but numba is not installed. "
            "Install with: pip install numba"
        ) from e


def get_backend_functions(
    backend: str | None = None,
) -> tuple[Callable, Callable]:
    """
    Get the grid allocation functions for the specified backend.

    Parameters
    ----------
    backend : str, optional
        Backend to use: 'numpy' or 'numba'. If not specified, uses the
        REVELSMD_BACKEND environment variable, defaulting to 'numba'.

    Returns
    -------
    tuple[Callable, Callable]
        Tuple of (triangular_allocation, box_allocation) functions.

    Raises
    ------
    ValueError
        If an unknown backend is specified.
    """
    if backend is None:
        backend = get_backend()

    if backend not in AVAILABLE_BACKENDS:
        raise ValueError(
            f"Unknown grid backend: {backend!r}. "
            f"Available backends: {sorted(AVAILABLE_BACKENDS)}"
        )

    if backend == 'numba':
        return _get_numba_functions()

    # NumPy backend
    return triangular_allocation, box_allocation


def triangular_allocation(
    forceX: np.ndarray,
    forceY: np.ndarray,
    forceZ: np.ndarray,
    counter: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    homeX: np.ndarray,
    homeY: np.ndarray,
    homeZ: np.ndarray,
    fox: np.ndarray,
    foy: np.ndarray,
    foz: np.ndarray,
    a: float | np.ndarray,
    lx: float,
    ly: float,
    lz: float,
    nbinsx: int,
    nbinsy: int,
    nbinsz: int,
) -> None:
    """
    Deposit contributions to the 8 neighbouring voxel vertices (CIC/triangular kernel).

    Uses trilinear interpolation to distribute each particle's contribution
    among the 8 surrounding grid vertices based on its position within the voxel.

    Parameters
    ----------
    forceX, forceY, forceZ : np.ndarray
        3D arrays for force components. Modified in place.
    counter : np.ndarray
        3D array for counting/weighting. Modified in place.
    x, y, z : np.ndarray
        Voxel indices from np.digitize (1-based upper edge).
    homeX, homeY, homeZ : np.ndarray
        Actual particle positions.
    fox, foy, foz : np.ndarray
        Force components for each particle.
    a : float or np.ndarray
        Weight factor (scalar or per-particle array).
    lx, ly, lz : float
        Voxel dimensions (bin widths).
    nbinsx, nbinsy, nbinsz : int
        Number of bins in each dimension.

    Notes
    -----
    This implementation uses np.add.at() to correctly accumulate contributions
    when multiple particles deposit to the same voxel vertices. Standard NumPy
    fancy indexing with += would only keep the last value for duplicate indices.
    """
    # Fractions within current voxel (digitize returns upper-edge index)
    fracx = 1 + ((homeX - (x * lx)) / lx)
    fracy = 1 + ((homeY - (y * ly)) / ly)
    fracz = 1 + ((homeZ - (z * lz)) / lz)

    # Vertex weights (trilinear factors)
    f_000 = (1 - fracx) * (1 - fracy) * (1 - fracz)
    f_001 = (1 - fracx) * (1 - fracy) * fracz
    f_010 = (1 - fracx) * fracy * (1 - fracz)
    f_100 = fracx * (1 - fracy) * (1 - fracz)
    f_101 = fracx * (1 - fracy) * fracz
    f_011 = (1 - fracx) * fracy * fracz
    f_110 = fracx * fracy * (1 - fracz)
    f_111 = fracx * fracy * fracz

    # Neighbour voxel indices (modulo wrap to preserve PBC)
    gx_0 = (x - 1) % nbinsx
    gx_1 = x % nbinsx
    gy_0 = (y - 1) % nbinsy
    gy_1 = y % nbinsy
    gz_0 = (z - 1) % nbinsz
    gz_1 = z % nbinsz

    # Grid shape for ravelling
    shape = (nbinsx, nbinsy, nbinsz)

    # Helper to deposit to a vertex using np.add.at
    def deposit(gx: np.ndarray, gy: np.ndarray, gz: np.ndarray, weight: np.ndarray) -> None:
        flat_idx = np.ravel_multi_index((gx, gy, gz), shape)
        weighted = weight * a
        np.add.at(forceX.ravel(), flat_idx, fox * weighted)
        np.add.at(forceY.ravel(), flat_idx, foy * weighted)
        np.add.at(forceZ.ravel(), flat_idx, foz * weighted)
        np.add.at(counter.ravel(), flat_idx, weighted)

    # Deposit to all 8 vertices
    deposit(gx_0, gy_0, gz_0, f_000)
    deposit(gx_0, gy_0, gz_1, f_001)
    deposit(gx_0, gy_1, gz_0, f_010)
    deposit(gx_1, gy_0, gz_0, f_100)
    deposit(gx_1, gy_0, gz_1, f_101)
    deposit(gx_0, gy_1, gz_1, f_011)
    deposit(gx_1, gy_1, gz_0, f_110)
    deposit(gx_1, gy_1, gz_1, f_111)


def box_allocation(
    forceX: np.ndarray,
    forceY: np.ndarray,
    forceZ: np.ndarray,
    counter: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    fox: np.ndarray,
    foy: np.ndarray,
    foz: np.ndarray,
    a: float | np.ndarray,
) -> None:
    """
    Deposit contributions to the host voxel (no neighbour spreading).

    Each particle's contribution goes entirely to its containing voxel.

    Parameters
    ----------
    forceX, forceY, forceZ : np.ndarray
        3D arrays for force components. Modified in place.
    counter : np.ndarray
        3D array for counting/weighting. Modified in place.
    x, y, z : np.ndarray
        Voxel indices (0-based).
    fox, foy, foz : np.ndarray
        Force components for each particle.
    a : float or np.ndarray
        Weight factor (scalar or per-particle array).

    Notes
    -----
    This implementation uses np.add.at() to correctly accumulate contributions
    when multiple particles are in the same voxel. Standard NumPy fancy indexing
    with += would only keep the last value for duplicate indices.
    """
    shape = forceX.shape
    flat_idx = np.ravel_multi_index((x, y, z), shape)

    np.add.at(forceX.ravel(), flat_idx, fox * a)
    np.add.at(forceY.ravel(), flat_idx, foy * a)
    np.add.at(forceZ.ravel(), flat_idx, foz * a)
    np.add.at(counter.ravel(), flat_idx, a if np.isscalar(a) else a)
