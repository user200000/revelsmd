"""
Tests for grid allocation helper functions.

These tests verify that the grid allocation backends (NumPy and Numba) correctly
deposit particle contributions to grid voxels, including proper handling of
overlapping particles (the main bug being fixed).
"""

import os
import pytest
import numpy as np


# ---------------------------------------------------------------------------
# Backend Selection Tests
# ---------------------------------------------------------------------------

class TestBackendSelection:
    """Test backend selection logic."""

    def test_default_backend_is_numba(self):
        """Default backend should be numba."""
        from revelsMD.density.grid_helpers import get_backend_functions
        tri_fn, box_fn = get_backend_functions()
        # Check that we got the numba versions
        assert 'numba' in tri_fn.__module__

    def test_explicit_numpy_backend(self):
        """Can explicitly request numpy backend."""
        from revelsMD.density.grid_helpers import get_backend_functions
        tri_fn, box_fn = get_backend_functions('numpy')
        assert 'grid_helpers' in tri_fn.__module__
        assert 'numba' not in tri_fn.__module__

    def test_explicit_numba_backend(self):
        """Can explicitly request numba backend."""
        from revelsMD.density.grid_helpers import get_backend_functions
        tri_fn, box_fn = get_backend_functions('numba')
        assert 'numba' in tri_fn.__module__

    def test_invalid_backend_raises(self):
        """Invalid backend name should raise ValueError."""
        from revelsMD.density.grid_helpers import get_backend_functions
        with pytest.raises(ValueError, match="Unknown grid backend"):
            get_backend_functions('invalid')

    def test_environment_variable_backend(self, monkeypatch):
        """Environment variable should control backend selection."""
        from revelsMD.density import grid_helpers
        monkeypatch.setenv('REVELSMD_BACKEND', 'numpy')
        tri_fn, box_fn = grid_helpers.get_backend_functions()
        assert 'numba' not in tri_fn.__module__


# ---------------------------------------------------------------------------
# Test Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def grid_arrays():
    """Create empty grid arrays for testing."""
    def _make_arrays(nbins=10):
        shape = (nbins, nbins, nbins)
        return {
            'forceX': np.zeros(shape, dtype=np.float64),
            'forceY': np.zeros(shape, dtype=np.float64),
            'forceZ': np.zeros(shape, dtype=np.float64),
            'counter': np.zeros(shape, dtype=np.float64),
            'nbinsx': nbins,
            'nbinsy': nbins,
            'nbinsz': nbins,
        }
    return _make_arrays


@pytest.fixture
def box_params():
    """Standard box parameters for testing."""
    return {
        'box_x': 10.0,
        'box_y': 10.0,
        'box_z': 10.0,
        'lx': 1.0,  # bin width = box / nbins
        'ly': 1.0,
        'lz': 1.0,
    }


# ---------------------------------------------------------------------------
# Triangular Allocation Tests
# ---------------------------------------------------------------------------

class TestTriangularAllocation:
    """Test triangular (CIC) kernel allocation."""

    def test_weights_sum_to_one(self, grid_arrays, box_params):
        """Trilinear weights for any position should sum to 1.0."""
        from revelsMD.density.grid_helpers import get_backend_functions
        tri_fn, _ = get_backend_functions('numpy')

        arrays = grid_arrays(nbins=10)

        # Single particle at arbitrary position
        homeX = np.array([3.7])
        homeY = np.array([5.2])
        homeZ = np.array([8.1])

        # Digitize to get voxel indices
        binsx = np.linspace(0, box_params['box_x'], arrays['nbinsx'] + 1)
        binsy = np.linspace(0, box_params['box_y'], arrays['nbinsy'] + 1)
        binsz = np.linspace(0, box_params['box_z'], arrays['nbinsz'] + 1)

        x = np.digitize(homeX, binsx)
        y = np.digitize(homeY, binsy)
        z = np.digitize(homeZ, binsz)

        tri_fn(
            arrays['forceX'], arrays['forceY'], arrays['forceZ'], arrays['counter'],
            x, y, z, homeX, homeY, homeZ,
            fox=np.array([0.0]), foy=np.array([0.0]), foz=np.array([0.0]),
            a=1.0,
            lx=box_params['lx'], ly=box_params['ly'], lz=box_params['lz'],
            nbinsx=arrays['nbinsx'], nbinsy=arrays['nbinsy'], nbinsz=arrays['nbinsz'],
        )

        # Counter should sum to 1.0 (weights sum to 1)
        assert np.isclose(arrays['counter'].sum(), 1.0), \
            f"Weights should sum to 1.0, got {arrays['counter'].sum()}"

    def test_particle_at_voxel_centre(self, grid_arrays, box_params):
        """Particle at voxel centre should distribute equally to 8 corners."""
        from revelsMD.density.grid_helpers import get_backend_functions
        tri_fn, _ = get_backend_functions('numpy')

        arrays = grid_arrays(nbins=10)

        # Particle at centre of voxel (5.5, 5.5, 5.5)
        homeX = np.array([5.5])
        homeY = np.array([5.5])
        homeZ = np.array([5.5])

        binsx = np.linspace(0, box_params['box_x'], arrays['nbinsx'] + 1)
        binsy = np.linspace(0, box_params['box_y'], arrays['nbinsy'] + 1)
        binsz = np.linspace(0, box_params['box_z'], arrays['nbinsz'] + 1)

        x = np.digitize(homeX, binsx)
        y = np.digitize(homeY, binsy)
        z = np.digitize(homeZ, binsz)

        tri_fn(
            arrays['forceX'], arrays['forceY'], arrays['forceZ'], arrays['counter'],
            x, y, z, homeX, homeY, homeZ,
            fox=np.array([0.0]), foy=np.array([0.0]), foz=np.array([0.0]),
            a=1.0,
            lx=box_params['lx'], ly=box_params['ly'], lz=box_params['lz'],
            nbinsx=arrays['nbinsx'], nbinsy=arrays['nbinsy'], nbinsz=arrays['nbinsz'],
        )

        # Should have exactly 8 non-zero entries, each with weight 0.125
        non_zero = arrays['counter'][arrays['counter'] > 0]
        assert len(non_zero) == 8, f"Expected 8 non-zero weights, got {len(non_zero)}"
        assert np.allclose(non_zero, 0.125), \
            f"Centre particle should give equal weights of 0.125, got {non_zero}"

    def test_force_direction_preserved(self, grid_arrays, box_params):
        """Force vectors should deposit with correct direction."""
        from revelsMD.density.grid_helpers import get_backend_functions
        tri_fn, _ = get_backend_functions('numpy')

        arrays = grid_arrays(nbins=10)

        homeX = np.array([5.5])
        homeY = np.array([5.5])
        homeZ = np.array([5.5])

        binsx = np.linspace(0, box_params['box_x'], arrays['nbinsx'] + 1)
        binsy = np.linspace(0, box_params['box_y'], arrays['nbinsy'] + 1)
        binsz = np.linspace(0, box_params['box_z'], arrays['nbinsz'] + 1)

        x = np.digitize(homeX, binsx)
        y = np.digitize(homeY, binsy)
        z = np.digitize(homeZ, binsz)

        # Force in +X direction only
        tri_fn(
            arrays['forceX'], arrays['forceY'], arrays['forceZ'], arrays['counter'],
            x, y, z, homeX, homeY, homeZ,
            fox=np.array([10.0]), foy=np.array([0.0]), foz=np.array([0.0]),
            a=1.0,
            lx=box_params['lx'], ly=box_params['ly'], lz=box_params['lz'],
            nbinsx=arrays['nbinsx'], nbinsy=arrays['nbinsy'], nbinsz=arrays['nbinsz'],
        )

        # Total force should be preserved
        assert np.isclose(arrays['forceX'].sum(), 10.0), \
            f"Total forceX should be 10.0, got {arrays['forceX'].sum()}"
        assert np.isclose(arrays['forceY'].sum(), 0.0), \
            f"Total forceY should be 0.0, got {arrays['forceY'].sum()}"
        assert np.isclose(arrays['forceZ'].sum(), 0.0), \
            f"Total forceZ should be 0.0, got {arrays['forceZ'].sum()}"

    def test_multiple_distinct_particles(self, grid_arrays, box_params):
        """Multiple particles at different positions should accumulate correctly."""
        from revelsMD.density.grid_helpers import get_backend_functions
        tri_fn, _ = get_backend_functions('numpy')

        arrays = grid_arrays(nbins=10)

        # Two particles at different positions
        homeX = np.array([2.5, 7.5])
        homeY = np.array([2.5, 7.5])
        homeZ = np.array([2.5, 7.5])

        binsx = np.linspace(0, box_params['box_x'], arrays['nbinsx'] + 1)
        binsy = np.linspace(0, box_params['box_y'], arrays['nbinsy'] + 1)
        binsz = np.linspace(0, box_params['box_z'], arrays['nbinsz'] + 1)

        x = np.digitize(homeX, binsx)
        y = np.digitize(homeY, binsy)
        z = np.digitize(homeZ, binsz)

        tri_fn(
            arrays['forceX'], arrays['forceY'], arrays['forceZ'], arrays['counter'],
            x, y, z, homeX, homeY, homeZ,
            fox=np.array([1.0, 2.0]), foy=np.array([0.0, 0.0]), foz=np.array([0.0, 0.0]),
            a=1.0,
            lx=box_params['lx'], ly=box_params['ly'], lz=box_params['lz'],
            nbinsx=arrays['nbinsx'], nbinsy=arrays['nbinsy'], nbinsz=arrays['nbinsz'],
        )

        # Counter should sum to 2.0 (one per particle)
        assert np.isclose(arrays['counter'].sum(), 2.0), \
            f"Total count should be 2.0, got {arrays['counter'].sum()}"
        # Total force should be 1 + 2 = 3
        assert np.isclose(arrays['forceX'].sum(), 3.0), \
            f"Total forceX should be 3.0, got {arrays['forceX'].sum()}"

    def test_overlapping_particles_accumulate(self, grid_arrays, box_params):
        """Two particles at identical positions should accumulate, not overwrite.

        This is the main bug fix test. The old implementation with NumPy fancy
        indexing would only keep the last value when indices contain duplicates.
        """
        from revelsMD.density.grid_helpers import get_backend_functions
        tri_fn, _ = get_backend_functions('numpy')

        arrays = grid_arrays(nbins=4)

        # Two particles at IDENTICAL positions
        homeX = np.array([5.0, 5.0])
        homeY = np.array([5.0, 5.0])
        homeZ = np.array([5.0, 5.0])

        binsx = np.linspace(0, box_params['box_x'], arrays['nbinsx'] + 1)
        binsy = np.linspace(0, box_params['box_y'], arrays['nbinsy'] + 1)
        binsz = np.linspace(0, box_params['box_z'], arrays['nbinsz'] + 1)

        x = np.digitize(homeX, binsx)
        y = np.digitize(homeY, binsy)
        z = np.digitize(homeZ, binsz)

        # Different forces: particle 1 has [1,0,0], particle 2 has [2,0,0]
        tri_fn(
            arrays['forceX'], arrays['forceY'], arrays['forceZ'], arrays['counter'],
            x, y, z, homeX, homeY, homeZ,
            fox=np.array([1.0, 2.0]), foy=np.array([0.0, 0.0]), foz=np.array([0.0, 0.0]),
            a=1.0,
            lx=box_params['box_x'] / arrays['nbinsx'],
            ly=box_params['box_y'] / arrays['nbinsy'],
            lz=box_params['box_z'] / arrays['nbinsz'],
            nbinsx=arrays['nbinsx'], nbinsy=arrays['nbinsy'], nbinsz=arrays['nbinsz'],
        )

        # Expected: total forceX = 1 + 2 = 3, counter = 2
        # Bug: only the last particle's contribution is kept (forceX = 2, counter = 1)
        assert np.isclose(arrays['counter'].sum(), 2.0), \
            f"Total count should be 2, got {arrays['counter'].sum()}"
        assert np.isclose(arrays['forceX'].sum(), 3.0), \
            f"Total forceX should be 3 (1+2), got {arrays['forceX'].sum()}"

    def test_periodic_boundary(self, grid_arrays, box_params):
        """Particles near box edge should wrap correctly via periodic boundaries."""
        from revelsMD.density.grid_helpers import get_backend_functions
        tri_fn, _ = get_backend_functions('numpy')

        arrays = grid_arrays(nbins=10)

        # Particle very close to origin (inside first voxel)
        # With bins [0, 1, 2, ..., 10], position 0.1 falls in voxel 1 (digitize returns 1)
        # CIC deposits to vertices (x-1, x) = (0, 1), which are valid indices
        homeX = np.array([0.1])
        homeY = np.array([0.1])
        homeZ = np.array([0.1])

        binsx = np.linspace(0, box_params['box_x'], arrays['nbinsx'] + 1)
        binsy = np.linspace(0, box_params['box_y'], arrays['nbinsy'] + 1)
        binsz = np.linspace(0, box_params['box_z'], arrays['nbinsz'] + 1)

        x = np.digitize(homeX, binsx)
        y = np.digitize(homeY, binsy)
        z = np.digitize(homeZ, binsz)

        tri_fn(
            arrays['forceX'], arrays['forceY'], arrays['forceZ'], arrays['counter'],
            x, y, z, homeX, homeY, homeZ,
            fox=np.array([1.0]), foy=np.array([0.0]), foz=np.array([0.0]),
            a=1.0,
            lx=box_params['lx'], ly=box_params['ly'], lz=box_params['lz'],
            nbinsx=arrays['nbinsx'], nbinsy=arrays['nbinsy'], nbinsz=arrays['nbinsz'],
        )

        # Weights should still sum to 1.0
        assert np.isclose(arrays['counter'].sum(), 1.0), \
            f"Weights should sum to 1.0, got {arrays['counter'].sum()}"
        # Deposits should be in voxels 0 and 1 (the 8 corners span these)
        assert arrays['counter'][0, :, :].sum() > 0, "Expected weight at index 0"
        assert arrays['counter'][1, :, :].sum() > 0, "Expected weight at index 1"


# ---------------------------------------------------------------------------
# Box Allocation Tests
# ---------------------------------------------------------------------------

class TestBoxAllocation:
    """Test box kernel allocation."""

    def test_single_particle(self, grid_arrays, box_params):
        """Single particle should deposit entirely to one voxel."""
        from revelsMD.density.grid_helpers import get_backend_functions
        _, box_fn = get_backend_functions('numpy')

        arrays = grid_arrays(nbins=10)

        # Single particle
        x = np.array([5])
        y = np.array([5])
        z = np.array([5])

        box_fn(
            arrays['forceX'], arrays['forceY'], arrays['forceZ'], arrays['counter'],
            x, y, z,
            fox=np.array([3.0]), foy=np.array([2.0]), foz=np.array([1.0]),
            a=1.0,
        )

        # Only one voxel should have non-zero values
        assert np.count_nonzero(arrays['counter']) == 1
        assert arrays['counter'][5, 5, 5] == 1.0
        assert arrays['forceX'][5, 5, 5] == 3.0
        assert arrays['forceY'][5, 5, 5] == 2.0
        assert arrays['forceZ'][5, 5, 5] == 1.0

    def test_multiple_distinct_particles(self, grid_arrays, box_params):
        """Multiple particles at different voxels should accumulate."""
        from revelsMD.density.grid_helpers import get_backend_functions
        _, box_fn = get_backend_functions('numpy')

        arrays = grid_arrays(nbins=10)

        # Two particles at different voxels
        x = np.array([3, 7])
        y = np.array([3, 7])
        z = np.array([3, 7])

        box_fn(
            arrays['forceX'], arrays['forceY'], arrays['forceZ'], arrays['counter'],
            x, y, z,
            fox=np.array([1.0, 2.0]), foy=np.array([0.0, 0.0]), foz=np.array([0.0, 0.0]),
            a=1.0,
        )

        assert np.isclose(arrays['counter'].sum(), 2.0)
        assert np.isclose(arrays['forceX'].sum(), 3.0)

    def test_overlapping_particles_accumulate(self, grid_arrays, box_params):
        """Two particles in same voxel should accumulate, not overwrite.

        This is the main bug fix test for box allocation.
        """
        from revelsMD.density.grid_helpers import get_backend_functions
        _, box_fn = get_backend_functions('numpy')

        arrays = grid_arrays(nbins=10)

        # Two particles in the SAME voxel
        x = np.array([5, 5])
        y = np.array([5, 5])
        z = np.array([5, 5])

        box_fn(
            arrays['forceX'], arrays['forceY'], arrays['forceZ'], arrays['counter'],
            x, y, z,
            fox=np.array([1.0, 2.0]), foy=np.array([0.0, 0.0]), foz=np.array([0.0, 0.0]),
            a=1.0,
        )

        # Expected: counter = 2, forceX = 3 (not just the last value)
        assert np.isclose(arrays['counter'].sum(), 2.0), \
            f"Total count should be 2, got {arrays['counter'].sum()}"
        assert np.isclose(arrays['forceX'].sum(), 3.0), \
            f"Total forceX should be 3 (1+2), got {arrays['forceX'].sum()}"


# ---------------------------------------------------------------------------
# Backend Equivalence Tests
# ---------------------------------------------------------------------------

class TestBackendEquivalence:
    """Test that NumPy and Numba backends produce identical results."""

    def test_triangular_backends_identical(self, grid_arrays, box_params):
        """NumPy and Numba triangular allocation should produce identical results."""
        from revelsMD.density.grid_helpers import get_backend_functions

        np.random.seed(42)
        n_particles = 50
        nbins = 10

        # Random particle positions
        homeX = np.random.uniform(0, box_params['box_x'], n_particles)
        homeY = np.random.uniform(0, box_params['box_y'], n_particles)
        homeZ = np.random.uniform(0, box_params['box_z'], n_particles)

        # Random forces
        fox = np.random.randn(n_particles)
        foy = np.random.randn(n_particles)
        foz = np.random.randn(n_particles)

        # Digitize
        binsx = np.linspace(0, box_params['box_x'], nbins + 1)
        binsy = np.linspace(0, box_params['box_y'], nbins + 1)
        binsz = np.linspace(0, box_params['box_z'], nbins + 1)

        x = np.digitize(homeX, binsx)
        y = np.digitize(homeY, binsy)
        z = np.digitize(homeZ, binsz)

        lx = box_params['box_x'] / nbins
        ly = box_params['box_y'] / nbins
        lz = box_params['box_z'] / nbins

        # NumPy backend
        arrays_np = grid_arrays(nbins=nbins)
        tri_np, _ = get_backend_functions('numpy')
        tri_np(
            arrays_np['forceX'], arrays_np['forceY'], arrays_np['forceZ'], arrays_np['counter'],
            x.copy(), y.copy(), z.copy(), homeX.copy(), homeY.copy(), homeZ.copy(),
            fox.copy(), foy.copy(), foz.copy(),
            a=1.0,
            lx=lx, ly=ly, lz=lz,
            nbinsx=nbins, nbinsy=nbins, nbinsz=nbins,
        )

        # Numba backend
        arrays_nb = grid_arrays(nbins=nbins)
        tri_nb, _ = get_backend_functions('numba')
        tri_nb(
            arrays_nb['forceX'], arrays_nb['forceY'], arrays_nb['forceZ'], arrays_nb['counter'],
            x.copy(), y.copy(), z.copy(), homeX.copy(), homeY.copy(), homeZ.copy(),
            fox.copy(), foy.copy(), foz.copy(),
            a=1.0,
            lx=lx, ly=ly, lz=lz,
            nbinsx=nbins, nbinsy=nbins, nbinsz=nbins,
        )

        # Compare results
        np.testing.assert_allclose(arrays_np['forceX'], arrays_nb['forceX'], rtol=1e-10)
        np.testing.assert_allclose(arrays_np['forceY'], arrays_nb['forceY'], rtol=1e-10)
        np.testing.assert_allclose(arrays_np['forceZ'], arrays_nb['forceZ'], rtol=1e-10)
        np.testing.assert_allclose(arrays_np['counter'], arrays_nb['counter'], rtol=1e-10)

    def test_box_backends_identical(self, grid_arrays, box_params):
        """NumPy and Numba box allocation should produce identical results."""
        from revelsMD.density.grid_helpers import get_backend_functions

        np.random.seed(42)
        n_particles = 50
        nbins = 10

        # Random voxel indices
        x = np.random.randint(0, nbins, n_particles)
        y = np.random.randint(0, nbins, n_particles)
        z = np.random.randint(0, nbins, n_particles)

        # Random forces
        fox = np.random.randn(n_particles)
        foy = np.random.randn(n_particles)
        foz = np.random.randn(n_particles)

        # NumPy backend
        arrays_np = grid_arrays(nbins=nbins)
        _, box_np = get_backend_functions('numpy')
        box_np(
            arrays_np['forceX'], arrays_np['forceY'], arrays_np['forceZ'], arrays_np['counter'],
            x.copy(), y.copy(), z.copy(),
            fox.copy(), foy.copy(), foz.copy(),
            a=1.0,
        )

        # Numba backend
        arrays_nb = grid_arrays(nbins=nbins)
        _, box_nb = get_backend_functions('numba')
        box_nb(
            arrays_nb['forceX'], arrays_nb['forceY'], arrays_nb['forceZ'], arrays_nb['counter'],
            x.copy(), y.copy(), z.copy(),
            fox.copy(), foy.copy(), foz.copy(),
            a=1.0,
        )

        # Compare results
        np.testing.assert_allclose(arrays_np['forceX'], arrays_nb['forceX'], rtol=1e-10)
        np.testing.assert_allclose(arrays_np['forceY'], arrays_nb['forceY'], rtol=1e-10)
        np.testing.assert_allclose(arrays_np['forceZ'], arrays_nb['forceZ'], rtol=1e-10)
        np.testing.assert_allclose(arrays_np['counter'], arrays_nb['counter'], rtol=1e-10)

    def test_overlapping_particles_both_backends(self, grid_arrays, box_params):
        """Both backends should correctly handle overlapping particles."""
        from revelsMD.density.grid_helpers import get_backend_functions

        nbins = 4

        # Two particles at identical positions
        homeX = np.array([5.0, 5.0])
        homeY = np.array([5.0, 5.0])
        homeZ = np.array([5.0, 5.0])

        binsx = np.linspace(0, box_params['box_x'], nbins + 1)
        binsy = np.linspace(0, box_params['box_y'], nbins + 1)
        binsz = np.linspace(0, box_params['box_z'], nbins + 1)

        x = np.digitize(homeX, binsx)
        y = np.digitize(homeY, binsy)
        z = np.digitize(homeZ, binsz)

        lx = box_params['box_x'] / nbins
        ly = box_params['box_y'] / nbins
        lz = box_params['box_z'] / nbins

        fox = np.array([1.0, 2.0])
        foy = np.array([0.0, 0.0])
        foz = np.array([0.0, 0.0])

        for backend in ['numpy', 'numba']:
            arrays = grid_arrays(nbins=nbins)
            tri_fn, _ = get_backend_functions(backend)
            tri_fn(
                arrays['forceX'], arrays['forceY'], arrays['forceZ'], arrays['counter'],
                x.copy(), y.copy(), z.copy(), homeX.copy(), homeY.copy(), homeZ.copy(),
                fox.copy(), foy.copy(), foz.copy(),
                a=1.0,
                lx=lx, ly=ly, lz=lz,
                nbinsx=nbins, nbinsy=nbins, nbinsz=nbins,
            )

            assert np.isclose(arrays['counter'].sum(), 2.0), \
                f"{backend} backend: count should be 2, got {arrays['counter'].sum()}"
            assert np.isclose(arrays['forceX'].sum(), 3.0), \
                f"{backend} backend: forceX should be 3, got {arrays['forceX'].sum()}"


# ---------------------------------------------------------------------------
# Scalar/Array 'a' Parameter Tests
# ---------------------------------------------------------------------------

class TestWeightParameter:
    """Test handling of scalar vs array 'a' parameter."""

    def test_scalar_a_parameter(self, grid_arrays, box_params):
        """Scalar 'a' should be applied uniformly."""
        from revelsMD.density.grid_helpers import get_backend_functions
        tri_fn, _ = get_backend_functions('numpy')

        arrays = grid_arrays(nbins=10)

        homeX = np.array([5.5])
        homeY = np.array([5.5])
        homeZ = np.array([5.5])

        binsx = np.linspace(0, box_params['box_x'], arrays['nbinsx'] + 1)
        x = np.digitize(homeX, binsx)
        y = np.digitize(homeX, binsx)
        z = np.digitize(homeX, binsx)

        tri_fn(
            arrays['forceX'], arrays['forceY'], arrays['forceZ'], arrays['counter'],
            x, y, z, homeX, homeY, homeZ,
            fox=np.array([1.0]), foy=np.array([0.0]), foz=np.array([0.0]),
            a=2.0,  # Scalar weight
            lx=box_params['lx'], ly=box_params['ly'], lz=box_params['lz'],
            nbinsx=arrays['nbinsx'], nbinsy=arrays['nbinsy'], nbinsz=arrays['nbinsz'],
        )

        # Counter should sum to 2.0 (1.0 weight * 2.0 scaling)
        assert np.isclose(arrays['counter'].sum(), 2.0)
        assert np.isclose(arrays['forceX'].sum(), 2.0)

    def test_array_a_parameter(self, grid_arrays, box_params):
        """Array 'a' should apply per-particle weights."""
        from revelsMD.density.grid_helpers import get_backend_functions
        tri_fn, _ = get_backend_functions('numpy')

        arrays = grid_arrays(nbins=10)

        # Two particles at different positions
        homeX = np.array([2.5, 7.5])
        homeY = np.array([2.5, 7.5])
        homeZ = np.array([2.5, 7.5])

        binsx = np.linspace(0, box_params['box_x'], arrays['nbinsx'] + 1)
        binsy = np.linspace(0, box_params['box_y'], arrays['nbinsy'] + 1)
        binsz = np.linspace(0, box_params['box_z'], arrays['nbinsz'] + 1)

        x = np.digitize(homeX, binsx)
        y = np.digitize(homeY, binsy)
        z = np.digitize(homeZ, binsz)

        tri_fn(
            arrays['forceX'], arrays['forceY'], arrays['forceZ'], arrays['counter'],
            x, y, z, homeX, homeY, homeZ,
            fox=np.array([1.0, 1.0]), foy=np.array([0.0, 0.0]), foz=np.array([0.0, 0.0]),
            a=np.array([1.0, 3.0]),  # Per-particle weights
            lx=box_params['lx'], ly=box_params['ly'], lz=box_params['lz'],
            nbinsx=arrays['nbinsx'], nbinsy=arrays['nbinsy'], nbinsz=arrays['nbinsz'],
        )

        # Counter should sum to 4.0 (1.0 + 3.0)
        assert np.isclose(arrays['counter'].sum(), 4.0)
        # ForceX should sum to 4.0 (1.0*1.0 + 1.0*3.0)
        assert np.isclose(arrays['forceX'].sum(), 4.0)
