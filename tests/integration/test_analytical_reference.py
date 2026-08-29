"""
Analytical reference tests for RevelsMD.

These tests validate numerical correctness against mathematically known results
using synthetic NumpyTrajectory data. They require no external data files.
"""

import pytest
import numpy as np
from scipy.integrate import trapezoid

from revelsMD.rdf import compute_rdf
from revelsMD.density import DensityGrid


# ---------------------------------------------------------------------------
# Harmonic pair potential: exact analytic reference
# ---------------------------------------------------------------------------

# Two atoms bound by u(r) = k/2 (r - r0)^2 in a cubic box. The pair
# separation r is sampled exactly from the equilibrium distribution
# p(r) proportional to r^2 exp(-beta u(r)), with a uniform random
# orientation per frame, and the stored forces are the true forces of
# the potential. The analytic pair distribution is then
#
#     g(r) = V exp(-beta u(r)) / integral(4 pi r'^2 exp(-beta u(r')) dr')
#
# Parameters chosen so the distribution (width sigma = 1/sqrt(beta k)
# = 0.2 around r0 = 2.5) sits comfortably inside the minimum-image
# range (rmax = 5.0) and the peak spans several bins at delr = 0.05.
# beta is deliberately not 1: the force estimator scales as beta^1, so
# a non-unit beta makes any misplaced beta power in the estimator
# visible as an overall scale error (~2.5x here).
HARMONIC_K = 62.5
HARMONIC_R0 = 2.5
HARMONIC_BETA = 0.4
HARMONIC_BOX = 10.0
HARMONIC_DELR = 0.05
HARMONIC_N_FRAMES = 5000
HARMONIC_SEED = 20260825
HARMONIC_R_LO = 1.0   # rejection-sampling support (p(r) is negligible outside)
HARMONIC_R_HI = 4.0


def _harmonic_radial_weight(r):
    """Unnormalised radial measure r^2 exp(-beta u(r)) for the harmonic pair."""
    u = 0.5 * HARMONIC_K * (r - HARMONIC_R0) ** 2
    return r**2 * np.exp(-HARMONIC_BETA * u)


def _sample_harmonic_separations(rng, n):
    """Rejection-sample n separations from p(r) ~ r^2 exp(-beta u(r))."""
    grid = np.linspace(HARMONIC_R_LO, HARMONIC_R_HI, 4001)
    envelope = _harmonic_radial_weight(grid).max() * 1.05
    out = np.empty(n)
    filled = 0
    while filled < n:
        need = n - filled
        r_try = rng.uniform(HARMONIC_R_LO, HARMONIC_R_HI, size=2 * need)
        accept = rng.uniform(0.0, envelope, size=2 * need) < _harmonic_radial_weight(r_try)
        got = r_try[accept][:need]
        out[filled:filled + got.size] = got
        filled += got.size
    return out


def _harmonic_analytic_g(r_vals):
    """Exact g(r) for the two-particle harmonic system, normalised numerically."""
    grid = np.linspace(0.0, HARMONIC_BOX / 2, 200001)
    z = trapezoid(4.0 * np.pi * _harmonic_radial_weight(grid), grid)
    u = 0.5 * HARMONIC_K * (r_vals - HARMONIC_R0) ** 2
    return HARMONIC_BOX**3 * np.exp(-HARMONIC_BETA * u) / z


@pytest.fixture(scope="module")
def harmonic_pair_rdf():
    """
    Compute the RDF of a synthetic harmonic-pair trajectory once per module.

    Returns a dict with the RDF object, the analytic reference curve, its
    peak value, and a mask selecting the region where p(r) is non-negligible.
    """
    from revelsMD.trajectories import NumpyTrajectory

    rng = np.random.default_rng(HARMONIC_SEED)
    r = _sample_harmonic_separations(rng, HARMONIC_N_FRAMES)
    v = rng.normal(size=(HARMONIC_N_FRAMES, 3))
    u_hat = v / np.linalg.norm(v, axis=1, keepdims=True)

    positions = np.empty((HARMONIC_N_FRAMES, 2, 3))
    forces = np.zeros((HARMONIC_N_FRAMES, 2, 3))
    centre = np.array([HARMONIC_BOX / 2] * 3)
    positions[:, 0, :] = centre
    positions[:, 1, :] = centre + r[:, None] * u_hat

    # True forces of the sampled potential: F_B = -du/dr r_hat, F_A = -F_B.
    f_b = (-HARMONIC_K * (r - HARMONIC_R0))[:, None] * u_hat
    forces[:, 1, :] = f_b
    forces[:, 0, :] = -f_b

    # kB = 1 in lj units, so trajectory.beta == HARMONIC_BETA exactly.
    ts = NumpyTrajectory(
        positions, forces,
        HARMONIC_BOX, HARMONIC_BOX, HARMONIC_BOX,
        ['A', 'B'],
        temperature=1.0 / HARMONIC_BETA, units='lj',
    )

    rdf = compute_rdf(ts, 'A', 'B', delr=HARMONIC_DELR, integration='backward')

    g_ref = _harmonic_analytic_g(rdf.r)
    peak = g_ref.max()
    mask = g_ref > 1e-3 * peak

    return {'rdf': rdf, 'g_ref': g_ref, 'peak': peak, 'mask': mask}


@pytest.mark.analytical
@pytest.mark.integration
class TestRDFAnalyticalReference:
    """Tests for RDF calculation against known analytical results."""

    def test_uniform_gas_rdf_approaches_unity(self, uniform_gas_trajectory):
        """
        Uniform random gas should have g(r) approaching 1 at large r.

        The backward integration should give g(r)~1 in bulk.
        Uses bulk region 2.0 < r < 4.5 to avoid edge effects.
        """
        ts = uniform_gas_trajectory

        # Use backward integration which should give g(r) ~ 1 for uniform gas
        rdf = compute_rdf(ts, '1', '1', delr=0.1, start=0, stop=-1, integration='backward')

        assert rdf.r is not None
        assert rdf.g is not None
        assert np.all(np.isfinite(rdf.g))

        # Use bulk region away from both short-range and cutoff edges
        mask = (rdf.r > 2.0) & (rdf.r < 4.5)
        assert np.any(mask), "No bins in bulk region"

        mean_gr = np.mean(rdf.g[mask])
        # Force-based g(r) has more variance due to integration
        assert abs(mean_gr - 1.0) < 0.3, f"Mean g(r) in bulk region = {mean_gr}, expected ~1.0"

    def test_two_atoms_rdf_peak_at_separation(self, two_atom_trajectory):
        """
        Two atoms at fixed separation should produce g(r) peak at that distance.

        With atoms at separation d = 3.0, the RDF should show a sharp peak
        at r = 3.0.
        """
        ts = two_atom_trajectory

        rdf = compute_rdf(ts, '1', '1', delr=0.1, start=0, stop=-1, integration='forward')

        assert rdf.r is not None
        assert rdf.g is not None
        assert np.all(np.isfinite(rdf.g))

        # Find the peak location
        peak_idx = np.argmax(rdf.g)
        peak_r = rdf.r[peak_idx]

        # Peak should be near r = 3.0 (the separation distance)
        expected_separation = 3.0
        assert abs(peak_r - expected_separation) < 0.5, \
            f"Peak at r = {peak_r}, expected near {expected_separation}"


@pytest.mark.analytical
@pytest.mark.integration
class TestDensityAnalyticalReference:
    """Tests for 3D density calculation against known analytical results."""

    def test_single_atom_density_peak(self, single_atom_trajectory):
        """
        Single atom at known position should produce density peak at that location.

        With one atom at (5, 5, 5) in a 10x10x10 box, the density should peak
        near the centre of the grid.
        """
        ts = single_atom_trajectory

        gs = DensityGrid(ts, 'number', nbins=20)
        gs.accumulate(ts, '1', kernel='triangular', rigid=False)

        assert gs.count > 0  # Data has been accumulated

        assert hasattr(gs, 'rho_force')
        assert gs.rho_force.shape == (20, 20, 20)
        assert np.all(np.isfinite(gs.rho_force))

        # The density should have its maximum near the centre
        # (atom is at 5,5,5 in a 10x10x10 box -> should be near bin 10,10,10)
        max_idx = np.unravel_index(np.argmax(gs.rho_force), gs.rho_force.shape)

        # Check that max is in the central region (within 5 bins of centre)
        centre = 10
        assert all(abs(idx - centre) < 6 for idx in max_idx), \
            f"Density peak at {max_idx}, expected near ({centre}, {centre}, {centre})"

    def test_density_conserves_total_count(self, uniform_gas_trajectory):
        """
        Total integrated density equals the number of atoms near-exactly.

        Both estimators conserve the particle count by construction: the
        counting estimator because trilinear deposit weights sum to 1 per
        particle, and the force estimator because its k = 0 mode is pinned
        to the mean counting density. Only float rounding remains
        (measured ~7e-16 relative for this fixture).
        """
        ts = uniform_gas_trajectory

        gs = DensityGrid(ts, 'number', nbins=20)
        gs.accumulate(ts, '1', kernel='triangular', rigid=False)

        # Calculate voxel volume
        voxel_vol = (ts.box_x / 20) * (ts.box_y / 20) * (ts.box_z / 20)

        n_atoms = len(ts.get_indices('1'))

        total_count = np.sum(gs.rho_count) * voxel_vol
        assert abs(total_count - n_atoms) / n_atoms < 1e-12, \
            f"Integrated rho_count = {total_count}, expected {n_atoms}"

        total_force = np.sum(gs.rho_force) * voxel_vol
        assert abs(total_force - n_atoms) / n_atoms < 1e-12, \
            f"Integrated rho_force = {total_force}, expected {n_atoms}"


@pytest.mark.analytical
@pytest.mark.integration
class TestMultispeciesRDF:
    """Tests for RDF calculations with multiple species."""

    def test_unlike_pair_rdf(self, multispecies_trajectory):
        """
        Unlike-pair RDF should work for two different species.
        """
        ts = multispecies_trajectory

        # Like pairs (1-1) with backward integration for g(r) ~ 1
        rdf_like = compute_rdf(ts, '1', '1', delr=0.2, integration='backward')

        # Unlike pairs (1-2) with backward integration
        rdf_unlike = compute_rdf(ts, '1', '2', delr=0.2, integration='backward')

        assert rdf_like.r is not None
        assert rdf_unlike.r is not None
        assert np.all(np.isfinite(rdf_like.g))
        assert np.all(np.isfinite(rdf_unlike.g))

        # With backward integration, both should approach 1 in bulk
        bulk_mask = rdf_like.r > 2.0
        if np.any(bulk_mask):
            mean_like = np.mean(rdf_like.g[bulk_mask])
            mean_unlike = np.mean(rdf_unlike.g[bulk_mask])

            # Both should be roughly 1 (with tolerance for statistics)
            assert abs(mean_like - 1.0) < 0.5, f"Like-pair bulk g(r) = {mean_like}"
            assert abs(mean_unlike - 1.0) < 0.5, f"Unlike-pair bulk g(r) = {mean_unlike}"


@pytest.mark.analytical
@pytest.mark.integration
class TestHistogramRDFAnalytical:
    """Tests for histogram-based g(r) against known analytical results."""

    def test_uniform_gas_histogram_rdf_approaches_unity(self, uniform_gas_trajectory):
        """
        Uniform random gas should have histogram g(r) ~ 1 at all r.

        This is the key validation: for an ideal gas, g_count should be ~1.0
        everywhere, unlike force-based g(r) which requires integration.

        Excludes r < 0.5 where statistical fluctuations are larger due to the
        small shell volume (not a boundary effect, just finite statistics).
        With 500 atoms and 50 frames, expect mean within 0.01 of 1.0.
        """
        ts = uniform_gas_trajectory

        rdf = compute_rdf(ts, '1', '1', delr=0.1, integration='backward')

        assert rdf.g_count is not None
        assert np.all(np.isfinite(rdf.g_count))

        # Exclude small r where shell volume is tiny and statistics are poor
        # This is a finite-size effect, not a boundary effect
        valid_mask = rdf.r > 0.5
        assert np.any(valid_mask), "No valid bins"

        mean_g_count = np.mean(rdf.g_count[valid_mask])
        assert abs(mean_g_count - 1.0) < 0.01, f"Mean g_count = {mean_g_count}, expected ~1.0"

    def test_two_atoms_histogram_shows_peak(self, two_atom_trajectory):
        """
        Two atoms at fixed separation should show peak in histogram g(r).

        The peak should be at the separation distance, smoothed by the
        triangular kernel.
        """
        ts = two_atom_trajectory

        rdf = compute_rdf(ts, '1', '1', delr=0.1, integration='forward')

        assert rdf.g_count is not None
        assert np.all(np.isfinite(rdf.g_count))

        # Find peak in histogram g(r)
        peak_idx = np.argmax(rdf.g_count)
        peak_r = rdf.r[peak_idx]

        # Peak should be near r = 3.0 (the separation distance)
        expected_separation = 3.0
        assert abs(peak_r - expected_separation) < 0.5, \
            f"Peak at r = {peak_r}, expected near {expected_separation}"

    def test_histogram_and_force_consistency(self, uniform_gas_trajectory):
        """
        For equilibrium systems, histogram and force-based g(r) should agree in bulk.

        Both g_count and g_force should be approximately 1.0 for a uniform gas.
        Excludes r < 0.5 due to poor statistics at small r (tiny shell volume).
        """
        ts = uniform_gas_trajectory

        rdf = compute_rdf(ts, '1', '1', delr=0.2, integration='backward')

        assert rdf.g_count is not None
        assert rdf.g_force is not None

        # Exclude small r where shell volume is tiny and statistics are poor
        valid_mask = rdf.r > 0.5
        assert np.any(valid_mask), "No valid bins"

        g_count_mean = np.mean(rdf.g_count[valid_mask])
        g_force_mean = np.mean(rdf.g_force[valid_mask])

        # Histogram g(r) should be very close to 1 (tight tolerance)
        assert abs(g_count_mean - 1.0) < 0.02, f"g_count mean = {g_count_mean}"
        # Force-based g(r) has more variance due to integration
        assert abs(g_force_mean - 1.0) < 0.3, f"g_force mean = {g_force_mean}"


@pytest.mark.analytical
@pytest.mark.integration
class TestHarmonicPotentialRDF:
    """
    Validate both RDF estimators against an exact analytic g(r).

    A two-atom system bound by a harmonic pair potential has a known
    closed-form pair distribution, giving external ground truth for both
    the force-sampled and histogram estimators, including their absolute
    normalisation. The force estimator is sensitive to prefactor errors
    (beta, volume, pair counting) that the uniform-gas tests cannot see.

    The backward integration pins g(rmax) = 1, but for a bound pair the
    true g(r) decays to 0 at large r, so the whole backward curve carries
    a constant offset of +1 relative to the analytic reference. The force
    assertions subtract this anchoring constant before comparing.

    Tolerances are set roughly 3x above the deviation measured for the
    committed seed (see each test), so failures indicate broken physics
    rather than statistical noise.
    """

    def test_force_rdf_matches_analytic(self, harmonic_pair_rdf):
        """
        Backward-integrated force g(r) matches the analytic curve.

        Measured max deviation for the committed seed: 0.034 of the peak
        height; asserted tolerance 0.12.
        """
        rdf = harmonic_pair_rdf['rdf']
        g_ref = harmonic_pair_rdf['g_ref']
        peak = harmonic_pair_rdf['peak']
        mask = harmonic_pair_rdf['mask']

        assert np.all(np.isfinite(rdf.g_force))

        # Remove the backward anchoring offset (true tail is 0, not 1)
        g_force = rdf.g_force - 1.0

        max_dev = np.max(np.abs(g_force[mask] - g_ref[mask])) / peak
        assert max_dev < 0.12, \
            f"Force g(r) deviates from analytic curve by {max_dev:.3f} of peak height"

    def test_force_rdf_tail_stays_at_anchor(self, harmonic_pair_rdf):
        """Backward g(r) remains at its anchor value beyond the sampled support."""
        rdf = harmonic_pair_rdf['rdf']

        tail = rdf.r > HARMONIC_R_HI + 0.2
        assert np.any(tail)
        assert np.max(np.abs(rdf.g_force[tail] - 1.0)) < 1e-10, \
            "Backward g(r) should remain at its anchor value beyond the sampled support"

    def test_histogram_rdf_matches_analytic(self, harmonic_pair_rdf):
        """
        Histogram g_count matches the analytic curve including normalisation.

        Measured max deviation for the committed seed: 0.055 of the peak
        height; asserted tolerance 0.17.
        """
        rdf = harmonic_pair_rdf['rdf']
        g_ref = harmonic_pair_rdf['g_ref']
        peak = harmonic_pair_rdf['peak']
        mask = harmonic_pair_rdf['mask']

        assert np.all(np.isfinite(rdf.g_count))

        max_dev = np.max(np.abs(rdf.g_count[mask] - g_ref[mask])) / peak
        assert max_dev < 0.17, \
            f"Histogram g(r) deviates from analytic curve by {max_dev:.3f} of peak height"

    def test_force_and_histogram_estimators_agree(self, harmonic_pair_rdf):
        """
        Force-sampled and histogram estimators agree with each other.

        Measured max deviation for the committed seed: 0.051 of the peak
        height; asserted tolerance 0.16.
        """
        rdf = harmonic_pair_rdf['rdf']
        peak = harmonic_pair_rdf['peak']
        mask = harmonic_pair_rdf['mask']

        g_force = rdf.g_force - 1.0  # remove backward anchoring offset

        max_dev = np.max(np.abs(g_force[mask] - rdf.g_count[mask])) / peak
        assert max_dev < 0.16, \
            f"Force and histogram estimators disagree by {max_dev:.3f} of peak height"

