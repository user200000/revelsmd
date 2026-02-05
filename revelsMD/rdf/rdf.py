"""Stateful RDF class for revelsMD, following the DensityGrid pattern."""

from __future__ import annotations

import numpy as np
from tqdm import tqdm

from revelsMD.rdf.rdf_helpers import (
    compute_pairwise_contributions,
    accumulate_binned_contributions,
    accumulate_triangular_counts,
)
from revelsMD.statistics import compute_lambda_weights, combine_estimators


class RDF:
    """
    Radial distribution function calculator.

    Follows the same pattern as DensityGrid:
    - Constructor sets up bins and indices
    - deposit() adds single-frame contributions (low-level, user-controlled iteration)
    - accumulate() iterates frames and calls deposit (convenience wrapper)
    - get_rdf() computes final g(r) with chosen integration

    Parameters
    ----------
    trajectory : Trajectory
        Trajectory object providing positions, forces, and box dimensions.
        Used to set up bins, indices, and prefactor.
    species_a : str
        First species name.
    species_b : str
        Second species name (same as species_a for like-species RDF).
    delr : float
        Bin spacing (default: 0.01).
    rmax : float or None
        Maximum r value. If None, uses half the minimum box dimension.

    Attributes
    ----------
    r : np.ndarray or None
        Bin centres (available after get_rdf).
    g : np.ndarray or None
        g(r) values (available after get_rdf).
    g_count : np.ndarray or None
        Histogram-based g(r) using triangular deposition (available after get_rdf).
    g_force : np.ndarray or None
        Force-based g(r), alias for g (available after get_rdf).
    lam : np.ndarray or None
        Lambda(r) values (only for integration='lambda').
    progress : str
        State: 'initialized', 'accumulated', or 'computed'.
    """

    def __init__(
        self,
        trajectory,
        species_a: str,
        species_b: str,
        delr: float = 0.01,
        rmax: float | None = None,
    ):
        self._trajectory = trajectory
        self.species_a = species_a
        self.species_b = species_b
        self.delr = delr

        # Store box dimensions for use in deposit
        self._box_x = trajectory.box_x
        self._box_y = trajectory.box_y
        self._box_z = trajectory.box_z
        self._beta = trajectory.beta

        # Compute rmax
        if rmax is None:
            self.rmax = min(trajectory.box_x, trajectory.box_y, trajectory.box_z) / 2
        else:
            self.rmax = rmax

        # Set up bins - use rmax + delr to ensure proper boundary handling
        # The returned r values will exclude the first (r=0) and last bin
        self._bins = np.arange(0, self.rmax + delr, delr)

        # Get indices and compute prefactor
        self._like_species = (species_a == species_b)
        indices_a = self._get_species_indices(trajectory, species_a)
        if self._like_species:
            n_a = len(indices_a)
            if n_a < 2:
                raise ValueError(
                    f"Like-species RDF requires at least 2 atoms of species '{species_a}', "
                    f"but only {n_a} found."
                )
            self._indices = [indices_a, indices_a]
            self._prefactor = float(trajectory.box_x * trajectory.box_y * trajectory.box_z) / (float(n_a) * float(n_a - 1))
        else:
            indices_b = self._get_species_indices(trajectory, species_b)
            self._indices = [indices_a, indices_b]
            self._prefactor = float(trajectory.box_x * trajectory.box_y * trajectory.box_z) / (float(len(indices_b)) * float(len(indices_a))) / 2

        # State
        self.progress = 'initialized'
        self._accumulated = np.zeros(len(self._bins), dtype=np.float64)
        self._counts = np.zeros(len(self._bins), dtype=np.float64)
        self._frame_data: list[np.ndarray] = []
        self._frame_count = 0

        # Results (set by get_rdf())
        self._r: np.ndarray | None = None
        self._g: np.ndarray | None = None
        self._g_count: np.ndarray | None = None
        self._lam: np.ndarray | None = None

    @staticmethod
    def _get_species_indices(trajectory, species: str) -> np.ndarray:
        """Get indices for a species, with RDF-specific error message."""
        try:
            return trajectory.get_indices(species)
        except ValueError:
            raise ValueError(f"No atoms found for species '{species}'.")

    @property
    def r(self) -> np.ndarray | None:
        """Bin centres."""
        return self._r

    @property
    def g(self) -> np.ndarray | None:
        """g(r) values."""
        return self._g

    @property
    def lam(self) -> np.ndarray | None:
        """Lambda(r) values (only available after get_rdf(integration='lambda'))."""
        return self._lam

    @property
    def g_count(self) -> np.ndarray | None:
        """Histogram-based g(r) using triangular deposition (available after get_rdf)."""
        return self._g_count

    @property
    def g_force(self) -> np.ndarray | None:
        """Force-based g(r), alias for g (available after get_rdf)."""
        return self._g

    def deposit(self, positions: np.ndarray, forces: np.ndarray) -> None:
        """
        Deposit a single frame's contribution to the RDF accumulator.

        Low-level method for user-controlled iteration. Mirrors DensityGrid.deposit().

        Parameters
        ----------
        positions : (N, 3) np.ndarray
            Atomic positions for this frame.
        forces : (N, 3) np.ndarray
            Atomic forces for this frame.
        """
        # Compute and accumulate
        force_result, count_result = self._single_frame(positions, forces)
        self._accumulated += force_result
        self._counts += count_result
        self._frame_data.append(force_result)
        self._frame_count += 1

        self.progress = 'accumulated'

    def accumulate(
        self,
        trajectory,
        start: int = 0,
        stop: int | None = None,
        period: int = 1,
    ) -> None:
        """
        Accumulate RDF contributions from trajectory frames.

        Convenience wrapper that handles frame iteration. Mirrors DensityGrid.accumulate().

        Parameters
        ----------
        trajectory : Trajectory
            Trajectory object to iterate over.
        start : int
            First frame index (default: 0).
        stop : int or None
            Stop frame index (default: None for all frames).
        period : int
            Frame stride (default: 1).
        """
        # Validate frame bounds
        if start > trajectory.frames:
            raise ValueError("First frame index exceeds frames in trajectory.")
        if stop is not None and stop > trajectory.frames:
            raise ValueError("Final frame index exceeds frames in trajectory.")

        # Calculate frame range for progress bar
        effective_stop = trajectory.frames if stop is None else (
            trajectory.frames + stop if stop < 0 else stop
        )
        norm_start = start % trajectory.frames if start >= 0 else max(0, trajectory.frames + start)
        to_run = range(int(norm_start), int(effective_stop), period)
        if len(to_run) == 0:
            raise ValueError("Final frame occurs before first frame in trajectory.")

        # Process frames using deposit
        for positions, forces in tqdm(trajectory.iter_frames(start, stop, period), total=len(to_run)):
            self.deposit(positions, forces)

    def _single_frame(
        self, positions: np.ndarray, forces: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute single-frame RDF contribution (forces and counts)."""
        pos_a = positions[self._indices[0], :]
        force_a = forces[self._indices[0], :]

        if self._like_species:
            pos_b = pos_a
            force_b = force_a
        else:
            pos_b = positions[self._indices[1], :]
            force_b = forces[self._indices[1], :]

        r_flat, dot_flat = compute_pairwise_contributions(
            pos_a, pos_b, force_a, force_b,
            (self._box_x, self._box_y, self._box_z)
        )

        force_result = accumulate_binned_contributions(dot_flat, r_flat, self._bins)
        count_result = accumulate_triangular_counts(r_flat, self._bins)

        return force_result, count_result

    def get_rdf(self, integration: str = 'forward') -> None:
        """
        Compute g(r) from accumulated data.

        Mirrors DensityGrid.get_real_density() pattern.

        Parameters
        ----------
        integration : {'forward', 'backward', 'lambda'}
            Integration direction:
            - 'forward': integrate from r=0, g(0)=0
            - 'backward': integrate from r=inf, g(inf)=1
            - 'lambda': variance-minimised combination
        """
        if self.progress == 'initialized':
            raise RuntimeError("Call accumulate() or deposit() before get_rdf().")

        if integration not in ('forward', 'backward', 'lambda'):
            raise ValueError(f"integration must be 'forward', 'backward', or 'lambda', got {integration!r}")

        if integration == 'lambda':
            self._compute_lambda()
        else:
            self._compute_standard(integration)

        self.progress = 'computed'

    def _compute_g_count(self) -> None:
        """Compute histogram-based g(r) from accumulated counts.

        Uses triangular-deposited counts, normalised by ideal gas expectation.
        The result is evaluated at the same r points as the force-based g(r).

        Notes
        -----
        For triangular (CIC) deposition, pairs at distance d between bin edges
        r_i and r_{i+1} distribute weight linearly between them. The effective
        volume for normalisation is derived by integrating the triangular weight
        function over both adjacent shells:

            V_eff(r) = (2*pi/3) * delr * (delr^2 + 6*r^2)

        At r=0, only the upper shell [0, delr] contributes, giving:

            V_eff(0) = pi * delr^3 / 3

        The last bin edge (at r_max + delr) also has only one contributing shell,
        but we discard this bin from returned results, so no correction is needed.

        See docs/triangular_deposition_normalisation.md for the full derivation.
        """
        delr = self.delr
        r_vals = self._bins

        # Exact effective volume for triangular deposition (interior edges)
        # V_eff(r) = (2*pi/3) * delr * (delr^2 + 6*r^2)
        eff_vol = (2.0 * np.pi / 3.0) * delr * (delr**2 + 6.0 * r_vals**2)

        # Boundary correction: at r=0, only the upper shell contributes
        # V_eff(0) = pi*delr^3/3 (half the general formula)
        eff_vol[0] = np.pi * delr**3 / 3.0

        # Box volume and particle counts
        volume = self._box_x * self._box_y * self._box_z
        n_ref = len(self._indices[0])

        # Ideal count at each bin edge:
        # N_ideal = N_ref × (N_target / V_box) × V_eff × n_frames
        #
        # For unlike species (A-B): N_ref = N_A, N_target = N_B
        # For like species (A-A):   Use N_ref × (N_ref - 1) / 2 to count
        #                           each pair once (upper triangle)
        if self._like_species:
            # Like-species: N × (N-1) / 2 pairs, density = N / V
            ideal_count = (
                (n_ref * (n_ref - 1) / 2) * (1.0 / volume) * eff_vol * self._frame_count
            )
        else:
            # Unlike-species: N_ref × N_target pairs, density = N_target / V
            n_target = len(self._indices[1])
            rho_target = n_target / volume
            ideal_count = n_ref * rho_target * eff_vol * self._frame_count

        # Compute g(r) = actual_count / ideal_count
        with np.errstate(divide='ignore', invalid='ignore'):
            g_count = self._counts / ideal_count

        g_count = np.nan_to_num(g_count, nan=0.0, posinf=0.0, neginf=0.0)

        # Trim to match self._r (excludes last bin due to boundary effect)
        # For lambda integration, r starts at bins[1], so trim accordingly
        if self._r is not None and len(self._r) == len(self._bins) - 2:
            # Lambda case: r = bins[1:-1], so g_count = g_count[1:-1]
            self._g_count = g_count[1:-1]
        else:
            # Standard case: r = bins[:-1], so g_count = g_count[:-1]
            self._g_count = g_count[:-1]

    def _compute_standard(self, integration: str) -> None:
        """Compute forward or backward integrated g(r)."""
        scaled = np.nan_to_num(self._accumulated.copy())
        scaled *= self._prefactor * self._beta / (4 * np.pi * self._frame_count)

        if integration == 'forward':
            g_full = np.cumsum(scaled)
        else:  # backward
            g_full = 1 - np.cumsum(scaled[::-1])[::-1]

        # Exclude only the last bin (boundary effect from triangular deposition)
        self._r = self._bins[:-1]
        self._g = g_full[:-1]
        self._lam = None
        self._compute_g_count()

    def _compute_lambda(self) -> None:
        """Compute lambda-corrected g(r)."""
        base_array = np.nan_to_num(np.array(self._frame_data))
        base_array *= self._prefactor * self._beta / (4 * np.pi)

        mean_scaled = np.nan_to_num(self._accumulated.copy())
        mean_scaled *= self._prefactor * self._beta / (4 * np.pi * self._frame_count)

        # Expectation curves
        exp_zero_rdf = np.cumsum(mean_scaled)[:-1]
        exp_inf_rdf = 1 - np.cumsum(mean_scaled[::-1])[::-1][1:]
        exp_delta = exp_inf_rdf - exp_zero_rdf

        # Per-frame curves
        base_zero_rdf = np.cumsum(base_array, axis=1)[:, :-1]
        base_inf_rdf = 1 - np.cumsum(base_array[:, ::-1], axis=1)[:, ::-1][:, 1:]
        base_delta = base_inf_rdf - base_zero_rdf

        # Lambda from covariance/variance
        var_del = np.mean((base_delta - exp_delta) ** 2, axis=0)
        cov_inf = np.mean((base_delta - exp_delta) * (base_inf_rdf - exp_inf_rdf), axis=0)
        combination = compute_lambda_weights(var_del, cov_inf)

        per_frame_combined = combine_estimators(base_inf_rdf, base_zero_rdf, combination)
        g_lambda = np.mean(per_frame_combined, axis=0)

        # Array length tracking:
        # - Forward/backward alignment ([:-1] and [1:]) gives n_bins - 1 elements
        # - g_lambda therefore has length n_bins - 1
        # - Excluding the last bin (g_lambda[:-1]) gives n_bins - 2 elements
        # - self._r = bins[1:-1] also has length n_bins - 2 (excludes first and last)
        # See issue #26 for discussion of the grid point loss from alignment.
        self._r = self._bins[1:-1]
        self._g = g_lambda[:-1]
        self._lam = combination[:-1]
        self._compute_g_count()


def compute_rdf(
    trajectory,
    species_a: str,
    species_b: str,
    delr: float = 0.01,
    rmax: float | None = None,
    start: int = 0,
    stop: int | None = None,
    period: int = 1,
    integration: str = 'forward',
) -> RDF:
    """
    Compute RDF from trajectory with a single function call.

    Mirrors compute_density() pattern.

    Parameters
    ----------
    trajectory : Trajectory
        Trajectory object providing positions, forces, and box dimensions.
    species_a : str
        First species name.
    species_b : str
        Second species name (same as species_a for like-species RDF).
    delr : float
        Bin spacing (default: 0.01).
    rmax : float or None
        Maximum r value. If None, uses half the minimum box dimension.
    start : int
        First frame index (default: 0).
    stop : int or None
        Stop frame index (default: None for all frames).
    period : int
        Frame stride (default: 1).
    integration : {'forward', 'backward', 'lambda'}
        Integration method (default: 'forward').

    Returns
    -------
    RDF
        RDF object with computed results available as properties.

    Examples
    --------
    >>> from revelsMD.rdf import compute_rdf
    >>> rdf = compute_rdf(trajectory, 'O', 'H', integration='forward')
    >>> print(rdf.r, rdf.g)
    """
    rdf = RDF(
        trajectory,
        species_a=species_a,
        species_b=species_b,
        delr=delr,
        rmax=rmax,
    )
    rdf.accumulate(trajectory, start=start, stop=stop, period=period)
    rdf.get_rdf(integration=integration)
    return rdf
