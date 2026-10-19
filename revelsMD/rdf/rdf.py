"""Stateful RDF class for revelsMD."""

from __future__ import annotations

import contextlib

import numpy as np
from tqdm import tqdm

from revelsMD.cell import inscribed_sphere_radius
from revelsMD.frame_sources import Frame
from revelsMD.rdf.rdf_helpers import get_backend_functions
from revelsMD.statistics import control_variate_combination
from revelsMD.trajectories._base import check_same_system, frame_range


class RDF:
    """
    Radial distribution function calculator.

    The constructor takes the system from ``trajectory``: its cell, beta and
    the number of atoms of each species. accumulate() adds the frames of a
    trajectory of that system, which may be another trajectory, such as an
    independent run; get_rdf() computes g(r) with the chosen integration.

    Parameters
    ----------
    trajectory : Trajectory
        A trajectory of the system, for its cell, beta and species counts.
        Its frames are not read; pass it to accumulate() for that.
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
        self.species_a = species_a
        self.species_b = species_b
        self.delr = delr

        # The system: frames from any trajectory with this cell, beta and
        # species counts can be accumulated.
        self._cell_matrix = np.array(trajectory.cell_matrix, dtype=np.float64)
        self._cell_inverse = np.linalg.inv(self._cell_matrix)
        self._cell_volume = float(abs(np.linalg.det(self._cell_matrix)))
        self._beta = trajectory.beta

        # Compute rmax
        if rmax is None:
            self.rmax = inscribed_sphere_radius(self._cell_matrix)
        else:
            self.rmax = rmax

        # Set up bins from 0 to rmax + delr (inclusive), with spacing delr.
        # The extra bin beyond rmax is an overflow bin: the accumulation
        # helpers zero/skip the last bin to handle pairs at distances > rmax,
        # so the overflow bin absorbs this rather than losing valid data.
        # The returned r values exclude the overflow bin.
        #
        # Use round() rather than np.arange to avoid floating point
        # sensitivity: inscribed_sphere_radius can introduce ~1e-14 noise
        # that causes np.arange to produce an extra bin edge.
        n_edges = int(round(self.rmax / delr)) + 2  # +1 fence-post, +1 overflow
        self._bins = np.arange(n_edges) * delr

        # Species counts and prefactor
        self._like_species = (species_a == species_b)
        self._indices = self._species_indices(trajectory)
        n_a, n_b = (len(indices) for indices in self._indices)
        self._species_counts = (n_a, n_b)
        if self._like_species:
            if n_a < 2:
                raise ValueError(
                    f"Like-species RDF requires at least 2 atoms of species '{species_a}', "
                    f"but only {n_a} found."
                )
            self._prefactor = self._cell_volume / (float(n_a) * float(n_a - 1))
        else:
            self._prefactor = self._cell_volume / (float(n_b) * float(n_a)) / 2

        # Get backend-selected functions
        (self._compute_pairwise,
         self._accumulate_binned,
         self._accumulate_triangular) = get_backend_functions()

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

    def _species_indices(self, trajectory) -> list[np.ndarray]:
        """The indices of species_a and species_b in ``trajectory``."""
        indices = []
        for species in (self.species_a, self.species_b):
            try:
                indices.append(trajectory.get_indices(species))
            except ValueError:
                raise ValueError(f"No atoms found for species '{species}'.")
        return indices

    @property
    def r(self) -> np.ndarray | None:
        """Bin centres.

        Returns
        -------
        np.ndarray or None
            1D array of bin centre values in the same units as the trajectory
            positions, or None before get_rdf() is called.
        """
        return self._r

    @property
    def g(self) -> np.ndarray | None:
        """g(r) values.

        Returns
        -------
        np.ndarray or None
            1D array of radial distribution function values, or None before
            get_rdf() is called.
        """
        return self._g

    @property
    def lam(self) -> np.ndarray | None:
        """Lambda(r) values (only available after get_rdf(integration='lambda')).

        This is the Eq. 3 weight of Coles et al. (2021) applied to the
        forward estimator g_0 (E_1 in the paper's notation; the backward
        estimator g_inf is E_0 and carries weight 1 - lambda). Bins where
        Delta is the same in every frame, to within rounding, report
        lambda = 0, i.e. the backward estimator.

        Returns
        -------
        np.ndarray or None
            1D array of per-bin lambda weights, or None if
            integration='lambda' was not used. The first entry (r = 0)
            is padding and is always 0.
        """
        return self._lam

    @property
    def g_count(self) -> np.ndarray | None:
        """Histogram-based g(r) using triangular deposition (available after get_rdf).

        Returns
        -------
        np.ndarray or None
            1D array of counting-based g(r) values, or None before
            get_rdf() is called.
        """
        return self._g_count

    @property
    def g_force(self) -> np.ndarray | None:
        """Force-based g(r), alias for g (available after get_rdf).

        Returns
        -------
        np.ndarray or None
            1D array of force-based g(r) values, or None before
            get_rdf() is called.
        """
        return self._g

    def _deposit(self, frame: Frame) -> None:
        """
        Deposit a single frame's contribution to the RDF accumulator.

        Each call deposits exactly one frame; depositing the same frame
        twice double-counts it.

        Parameters
        ----------
        frame : Frame
            Trajectory frame with positions and forces.
        """
        # Compute and accumulate
        force_result, count_result = self._single_frame(frame)
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

        Calling accumulate() again adds more frames, from the same trajectory
        or another of the same system. The atoms of each species are found by
        name in each trajectory, so their order may differ.

        Parameters
        ----------
        trajectory : Trajectory
            Trajectory whose frames to add; its cell, temperature, units and
            species counts must be those the RDF was built with.
        start : int
            First frame index (default: 0).
        stop : int or None
            Stop frame index (default: None for all frames).
        period : int
            Frame stride (default: 1).

        Raises
        ------
        ValueError
            If the trajectory's cell, temperature, units or species counts
            differ from the RDF's; if start or stop lies outside the
            trajectory, or the selected range is empty. Negative values count
            from the end of the trajectory.

        Notes
        -----
        If a frame cannot be read, the frames deposited before it are kept.
        Results from an earlier get_rdf() are cleared.
        """
        check_same_system(trajectory, self._cell_matrix, self._beta, "RDF")
        indices = self._species_indices(trajectory)
        counts = tuple(len(i) for i in indices)
        if counts != self._species_counts:
            raise ValueError(
                f"the trajectory has {counts[0]} '{self.species_a}' and {counts[1]} "
                f"'{self.species_b}' atoms, but the RDF was built for "
                f"{self._species_counts[0]} and {self._species_counts[1]}"
            )
        to_run = frame_range(trajectory.frames, start, stop, period)
        self._indices = indices
        self._r = self._g = self._g_count = self._lam = None
        with contextlib.closing(
            trajectory.iter_frames(to_run.start, to_run.stop, to_run.step)
        ) as frames:
            for frame in tqdm(frames, total=len(to_run)):
                self._deposit(frame)

    def _single_frame(
        self, frame: Frame
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute single-frame RDF contribution (forces and counts)."""
        pos_a = frame.positions[self._indices[0], :]
        force_a = frame.forces[self._indices[0], :]

        if self._like_species:
            pos_b = pos_a
            force_b = force_a
        else:
            pos_b = frame.positions[self._indices[1], :]
            force_b = frame.forces[self._indices[1], :]

        r_flat, dot_flat = self._compute_pairwise(
            pos_a, pos_b, force_a, force_b,
            self._cell_matrix, self._cell_inverse,
        )

        force_result = self._accumulate_binned(dot_flat, r_flat, self._bins)
        count_result = self._accumulate_triangular(r_flat, self._bins)

        return force_result, count_result

    def get_rdf(self, integration: str = 'forward') -> None:
        """
        Compute g(r) from accumulated data.

        Parameters
        ----------
        integration : {'forward', 'backward', 'lambda'}
            Integration direction:
            - 'forward': integrate from r=0, g(0)=0
            - 'backward': integrate from r=inf, g(inf)=1
            - 'lambda': variance-minimised combination

        Raises
        ------
        RuntimeError
            If called before accumulate().
        ValueError
            If integration is not one of the recognised options.
        """
        if self.progress == 'initialized':
            raise RuntimeError("Call accumulate() before get_rdf().")

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

        See the kernels page of the documentation (section "CIC normalisation
        for the RDF") for the full derivation.
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
        volume = self._cell_volume
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

        # Exclude the overflow bin — all integration methods use r = bins[:-1]
        self._g_count = g_count[:-1]

    def _compute_standard(self, integration: str) -> None:
        """Compute forward or backward integrated g(r)."""
        scaled = np.nan_to_num(self._accumulated.copy())
        scaled *= self._prefactor * self._beta / (4 * np.pi * self._frame_count)

        if integration == 'forward':
            g_full = np.cumsum(scaled)
        else:  # backward
            g_full = 1 - np.cumsum(scaled[::-1])[::-1]

        # Exclude the overflow bin — all integration methods use r = bins[:-1]
        self._r = self._bins[:-1]
        self._g = g_full[:-1]
        self._lam = None
        self._compute_g_count()

    def _compute_lambda(self) -> None:
        """Compute the optimal linear combination of forward and backward g(r).

        Uses control_variate_combination (Coles et al. (2021), Eqs. 3-4)
        with frames as blocks. For the RDF the paper assigns E_0 = g_inf
        (backward estimator, integrated down from r -> infinity) and
        E_1 = g_0 (forward estimator, integrated up from r = 0). Note the
        notation collision: the paper's E_0 is NOT g_0.

        Where Delta is the same in every frame to within rounding
        (degenerate input, e.g. a single frame), lambda = 0 and the result
        is the backward estimator g_inf.
        """
        base_array = np.nan_to_num(np.array(self._frame_data))
        base_array *= self._prefactor * self._beta / (4 * np.pi)

        mean_scaled = np.nan_to_num(self._accumulated.copy())
        mean_scaled *= self._prefactor * self._beta / (4 * np.pi * self._frame_count)

        # Expectation curves. The forward/backward alignment ([:-1] and
        # [1:]) places both estimators on the same r grid, excluding r=0
        # where both are trivially zero.
        exp_forward = np.cumsum(mean_scaled)[:-1]
        exp_backward = 1 - np.cumsum(mean_scaled[::-1])[::-1][1:]

        # Per-frame estimator curves: E_1 = g_0 (forward), E_0 = g_inf
        # (backward).
        g_forward = np.cumsum(base_array, axis=1)[:, :-1]
        g_backward = 1 - np.cumsum(base_array[:, ::-1], axis=1)[:, ::-1][:, 1:]

        delta = g_forward - g_backward
        exp_delta = exp_forward - exp_backward

        var_delta = np.mean((delta - exp_delta) ** 2, axis=0)
        cov_e0_delta = np.mean(
            (delta - exp_delta) * (g_backward - exp_backward), axis=0
        )
        # g_backward subtracts cumulative sums from 1, so the rounding in
        # Delta is set by 1 and the largest cumulative sum, in every bin.
        g_lambda, lam = control_variate_combination(
            exp_backward, exp_forward, var_delta, cov_e0_delta,
            scale=float(max(1.0, np.max(np.abs(np.cumsum(mean_scaled))))),
            fallback=0.0,
        )

        # Pad the result so all integration methods return the same grid:
        # bins[:-1] = [0, delr, ..., rmax], with g(0) = 0.
        self._r = self._bins[:-1]
        self._g = np.concatenate([[0.0], g_lambda[:-1]])
        self._lam = np.concatenate([[0.0], lam[:-1]])
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
