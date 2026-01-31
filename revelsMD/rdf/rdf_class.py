"""Stateful RDF class for revelsMD, following the DensityGrid pattern."""

from __future__ import annotations

import numpy as np
from tqdm import tqdm

from revelsMD.rdf.rdf_helpers import (
    compute_pairwise_contributions,
    accumulate_binned_contributions,
)


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

        # Set up bins
        self._bins = np.arange(0, self.rmax, delr)

        # Get indices and compute prefactor
        self._like_species = (species_a == species_b)
        indices_a = trajectory.get_indices(species_a)
        if self._like_species:
            self._indices = [indices_a, indices_a]
            n_a = len(indices_a)
            self._prefactor = float(trajectory.box_x * trajectory.box_y * trajectory.box_z) / (float(n_a) * float(n_a - 1))
        else:
            indices_b = trajectory.get_indices(species_b)
            self._indices = [indices_a, indices_b]
            self._prefactor = float(trajectory.box_x * trajectory.box_y * trajectory.box_z) / (float(len(indices_b)) * float(len(indices_a))) / 2

        # State
        self.progress = 'initialized'
        self._accumulated: np.ndarray | None = None
        self._frame_data: list[np.ndarray] = []
        self._frame_count = 0

        # Results (set by get_rdf())
        self._r: np.ndarray | None = None
        self._g: np.ndarray | None = None
        self._lam: np.ndarray | None = None

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
        # Initialize accumulator on first call
        if self._accumulated is None:
            self._accumulated = np.zeros(len(self._bins), dtype=np.float64)

        # Compute and accumulate
        frame_result = self._single_frame(positions, forces)
        self._accumulated += frame_result
        self._frame_data.append(frame_result)
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

        Convenience wrapper that handles frame iteration. Mirrors DensityGrid.make_force_grid().

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

    def _single_frame(self, positions: np.ndarray, forces: np.ndarray) -> np.ndarray:
        """Compute single-frame RDF contribution."""
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
        return accumulate_binned_contributions(dot_flat, r_flat, self._bins)

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

    def _compute_standard(self, integration: str) -> None:
        """Compute forward or backward integrated g(r)."""
        scaled = np.nan_to_num(self._accumulated.copy())
        scaled *= self._prefactor * self._beta / (4 * np.pi * self._frame_count)

        if integration == 'forward':
            self._r = self._bins
            self._g = np.cumsum(scaled)
        else:  # backward
            self._r = self._bins
            self._g = 1 - np.cumsum(scaled[::-1])[::-1]

        self._lam = None

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
        var_del_safe = np.where(var_del == 0, 1.0, var_del)
        combination = np.divide(cov_inf, var_del_safe)
        combination = np.nan_to_num(combination, nan=0.0, posinf=0.0, neginf=0.0)

        g_lambda = np.nan_to_num(
            np.mean(base_inf_rdf * (1 - combination) + base_zero_rdf * combination, axis=0),
            nan=0.0, posinf=0.0, neginf=0.0
        )

        self._r = self._bins[1:]
        self._g = g_lambda
        self._lam = combination


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
