"""
Unit conversion utilities for RevelsMD.

This module provides physical constants and simple conversion helpers
used to translate between different molecular dynamics unit systems.

Note: The `generate_boltzmann` function was removed in favour of storing
`beta = 1/(kB*T)` directly in trajectory objects. See `compute_beta()`
in `revelsMD.trajectories._base` for the replacement.
"""
