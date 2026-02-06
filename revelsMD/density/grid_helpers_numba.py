"""
Numba JIT-compiled grid allocation functions.

This module provides high-performance implementations of the grid allocation
functions using Numba's JIT compilation. The implementations use explicit
loops which naturally handle the overlapping particles case correctly.

Note: We use sequential loops (not prange) because parallel execution would
have race conditions when multiple particles deposit to the same voxel.
"""

from __future__ import annotations

import numpy as np
from numba import jit  # type: ignore[import-untyped]


@jit(nopython=True, cache=True)
def _triangular_allocation_numba(
    force_x: np.ndarray,
    force_y: np.ndarray,
    force_z: np.ndarray,
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
    a_arr: np.ndarray,
    lx: float,
    ly: float,
    lz: float,
    nbinsx: int,
    nbinsy: int,
    nbinsz: int,
) -> None:
    """
    JIT-compiled triangular allocation with explicit loops.

    The explicit loop naturally handles overlapping particles correctly
    since each iteration accumulates to the grid independently.
    """
    n_particles = len(x)

    for i in range(n_particles):
        # Get weight for this particle
        a_val = a_arr[i]

        # Fractions within current voxel
        fracx = 1.0 + ((homeX[i] - (x[i] * lx)) / lx)
        fracy = 1.0 + ((homeY[i] - (y[i] * ly)) / ly)
        fracz = 1.0 + ((homeZ[i] - (z[i] * lz)) / lz)

        # Vertex weights (trilinear factors)
        f_000 = (1.0 - fracx) * (1.0 - fracy) * (1.0 - fracz)
        f_001 = (1.0 - fracx) * (1.0 - fracy) * fracz
        f_010 = (1.0 - fracx) * fracy * (1.0 - fracz)
        f_100 = fracx * (1.0 - fracy) * (1.0 - fracz)
        f_101 = fracx * (1.0 - fracy) * fracz
        f_011 = (1.0 - fracx) * fracy * fracz
        f_110 = fracx * fracy * (1.0 - fracz)
        f_111 = fracx * fracy * fracz

        # Neighbour voxel indices (modulo wrap for PBC)
        gx_0 = (x[i] - 1) % nbinsx
        gx_1 = x[i] % nbinsx
        gy_0 = (y[i] - 1) % nbinsy
        gy_1 = y[i] % nbinsy
        gz_0 = (z[i] - 1) % nbinsz
        gz_1 = z[i] % nbinsz

        # Get force values
        fx = fox[i]
        fy = foy[i]
        fz = foz[i]

        # Deposit to all 8 vertices
        # Vertex (0, 0, 0)
        w = f_000 * a_val
        force_x[gx_0, gy_0, gz_0] += fx * w
        force_y[gx_0, gy_0, gz_0] += fy * w
        force_z[gx_0, gy_0, gz_0] += fz * w
        counter[gx_0, gy_0, gz_0] += w

        # Vertex (0, 0, 1)
        w = f_001 * a_val
        force_x[gx_0, gy_0, gz_1] += fx * w
        force_y[gx_0, gy_0, gz_1] += fy * w
        force_z[gx_0, gy_0, gz_1] += fz * w
        counter[gx_0, gy_0, gz_1] += w

        # Vertex (0, 1, 0)
        w = f_010 * a_val
        force_x[gx_0, gy_1, gz_0] += fx * w
        force_y[gx_0, gy_1, gz_0] += fy * w
        force_z[gx_0, gy_1, gz_0] += fz * w
        counter[gx_0, gy_1, gz_0] += w

        # Vertex (1, 0, 0)
        w = f_100 * a_val
        force_x[gx_1, gy_0, gz_0] += fx * w
        force_y[gx_1, gy_0, gz_0] += fy * w
        force_z[gx_1, gy_0, gz_0] += fz * w
        counter[gx_1, gy_0, gz_0] += w

        # Vertex (1, 0, 1)
        w = f_101 * a_val
        force_x[gx_1, gy_0, gz_1] += fx * w
        force_y[gx_1, gy_0, gz_1] += fy * w
        force_z[gx_1, gy_0, gz_1] += fz * w
        counter[gx_1, gy_0, gz_1] += w

        # Vertex (0, 1, 1)
        w = f_011 * a_val
        force_x[gx_0, gy_1, gz_1] += fx * w
        force_y[gx_0, gy_1, gz_1] += fy * w
        force_z[gx_0, gy_1, gz_1] += fz * w
        counter[gx_0, gy_1, gz_1] += w

        # Vertex (1, 1, 0)
        w = f_110 * a_val
        force_x[gx_1, gy_1, gz_0] += fx * w
        force_y[gx_1, gy_1, gz_0] += fy * w
        force_z[gx_1, gy_1, gz_0] += fz * w
        counter[gx_1, gy_1, gz_0] += w

        # Vertex (1, 1, 1)
        w = f_111 * a_val
        force_x[gx_1, gy_1, gz_1] += fx * w
        force_y[gx_1, gy_1, gz_1] += fy * w
        force_z[gx_1, gy_1, gz_1] += fz * w
        counter[gx_1, gy_1, gz_1] += w


def triangular_allocation_numba(
    force_x: np.ndarray,
    force_y: np.ndarray,
    force_z: np.ndarray,
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
    Deposit contributions to 8 neighbouring voxel vertices (CIC/triangular kernel).

    This is a wrapper around the JIT-compiled implementation that handles
    the conversion of the 'a' parameter to an array.

    Parameters
    ----------
    force_x, force_y, force_z : np.ndarray
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
    """
    # Ensure contiguous arrays with correct dtype
    x = np.ascontiguousarray(x, dtype=np.int64)
    y = np.ascontiguousarray(y, dtype=np.int64)
    z = np.ascontiguousarray(z, dtype=np.int64)
    homeX = np.ascontiguousarray(homeX, dtype=np.float64)
    homeY = np.ascontiguousarray(homeY, dtype=np.float64)
    homeZ = np.ascontiguousarray(homeZ, dtype=np.float64)
    fox = np.ascontiguousarray(fox, dtype=np.float64)
    foy = np.ascontiguousarray(foy, dtype=np.float64)
    foz = np.ascontiguousarray(foz, dtype=np.float64)

    # Convert scalar 'a' to array
    n_particles = len(x)
    if np.isscalar(a):
        a_arr = np.full(n_particles, a, dtype=np.float64)
    else:
        a_arr = np.ascontiguousarray(a, dtype=np.float64)

    _triangular_allocation_numba(
        force_x, force_y, force_z, counter,
        x, y, z, homeX, homeY, homeZ,
        fox, foy, foz, a_arr,
        lx, ly, lz, nbinsx, nbinsy, nbinsz,
    )


@jit(nopython=True, cache=True)
def _box_allocation_numba(
    force_x: np.ndarray,
    force_y: np.ndarray,
    force_z: np.ndarray,
    counter: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    fox: np.ndarray,
    foy: np.ndarray,
    foz: np.ndarray,
    a_arr: np.ndarray,
) -> None:
    """
    JIT-compiled box allocation with explicit loops.

    The explicit loop naturally handles overlapping particles correctly.
    """
    n_particles = len(x)

    for i in range(n_particles):
        a_val = a_arr[i]
        xi = x[i]
        yi = y[i]
        zi = z[i]

        force_x[xi, yi, zi] += fox[i] * a_val
        force_y[xi, yi, zi] += foy[i] * a_val
        force_z[xi, yi, zi] += foz[i] * a_val
        counter[xi, yi, zi] += a_val


def box_allocation_numba(
    force_x: np.ndarray,
    force_y: np.ndarray,
    force_z: np.ndarray,
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

    This is a wrapper around the JIT-compiled implementation that handles
    the conversion of the 'a' parameter to an array.

    Parameters
    ----------
    force_x, force_y, force_z : np.ndarray
        3D arrays for force components. Modified in place.
    counter : np.ndarray
        3D array for counting/weighting. Modified in place.
    x, y, z : np.ndarray
        Voxel indices (0-based).
    fox, foy, foz : np.ndarray
        Force components for each particle.
    a : float or np.ndarray
        Weight factor (scalar or per-particle array).
    """
    # Ensure contiguous arrays with correct dtype
    x = np.ascontiguousarray(x, dtype=np.int64)
    y = np.ascontiguousarray(y, dtype=np.int64)
    z = np.ascontiguousarray(z, dtype=np.int64)
    fox = np.ascontiguousarray(fox, dtype=np.float64)
    foy = np.ascontiguousarray(foy, dtype=np.float64)
    foz = np.ascontiguousarray(foz, dtype=np.float64)

    # Convert scalar 'a' to array
    n_particles = len(x)
    if np.isscalar(a):
        a_arr = np.full(n_particles, a, dtype=np.float64)
    else:
        a_arr = np.ascontiguousarray(a, dtype=np.float64)

    _box_allocation_numba(
        force_x, force_y, force_z, counter,
        x, y, z, fox, foy, foz, a_arr,
    )
