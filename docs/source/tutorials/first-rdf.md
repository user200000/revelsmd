# Radial distribution functions

This tutorial computes a radial distribution function for a Lennard-Jones fluid
using the example data in `examples/example_1_LJ/`.

## Load the trajectory

```python
from revelsMD.trajectories import LammpsTrajectory

traj = LammpsTrajectory(
    'examples/example_1_LJ/dump.nh.lammps',
    'examples/example_1_LJ/data.fin.nh.data',
    temperature=1.35,
    units='lj',
    atom_style="id resid type q x y z ix iy iz",
)
```

`temperature` sets the inverse thermal energy used in the force estimator.
For LJ reduced units, this is the reduced temperature T*.

The topology file (`data.fin.nh.data`) is required: it supplies atom types
and box geometry. `atom_style` describes the column layout in the data file.

## Compute the RDF

The quickest route is `compute_rdf`, which accumulates all frames and
integrates in one call:

```python
from revelsMD.rdf import compute_rdf

rdf = compute_rdf(traj, '1', '1', integration='lambda')
```

Species are identified by LAMMPS atom type number, passed as a string.
This computes the type-1/type-1 RDF. Use `'1'` and `'2'` for a cross-species RDF.

`integration='lambda'` selects the variance-minimised estimator. The
alternatives are `'forward'` (integrates from g(0) = 0) and `'backward'`
(integrates from g(inf) = 1).

## Access the results

```python
import numpy as np

r       = rdf.r        # bin centres, shape (N,)
g_force = rdf.g_force  # force-based g(r), alias for rdf.g
g_count = rdf.g_count  # histogram-based g(r)
```

All three are NumPy arrays of the same length.

## Plot: force-based vs histogram

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(r, g_count, label='histogram')
ax.plot(r, g_force, label='force-based (lambda)')
ax.set_xlabel('r')
ax.set_ylabel('g(r)')
ax.legend()
plt.show()
```

```{image} /_static/images/tutorial_rdf.png
:alt: Comparison of histogram and force-based g(r) for a Lennard-Jones fluid
:width: 60%
```

The force-based estimator resolves pair structure with far less noise than
the histogram. This is a single frame with fine bins (`delr=0.005`) — the
histogram is barely usable while the force-based curve is smooth.

## Using the RDF class directly

`compute_rdf` is a convenience wrapper. For separate control over
accumulation and integration, use the `RDF` class:

```python
from revelsMD.rdf import RDF

rdf = RDF(traj, '1', '1', delr=0.005, rmax=4.0)
rdf.accumulate(traj, start=10, stop=None, period=2)
rdf.get_rdf(integration='lambda')
```

Separating `accumulate()` from `get_rdf()` lets you try different integration
methods without re-reading the trajectory:

```python
rdf.get_rdf(integration='forward')
g_fwd = rdf.g_force.copy()

rdf.get_rdf(integration='backward')
g_bwd = rdf.g_force.copy()

rdf.get_rdf(integration='lambda')
g_lam = rdf.g_force.copy()
lam   = rdf.lam      # lambda weights; available only after integration='lambda'
```

The `lam` array shows where the estimator blends between forward (lam = 0)
and backward (lam = 1) integration.

## Frame selection

`accumulate()` accepts `start`, `stop`, and `period`:

```python
# Use every other frame from frame 50 onwards
rdf.accumulate(traj, start=50, period=2)

# Use only the first 100 frames
rdf.accumulate(traj, stop=100)
```

These have the same meaning as Python slice indices.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `delr` | `0.01` | Bin width in distance units |
| `rmax` | `None` | Maximum r; defaults to the inscribed sphere radius |
| `start` | `0` | First frame index |
| `stop` | `None` | Stop frame index (exclusive); `None` = all frames |
| `period` | `1` | Frame stride |
| `integration` | `'forward'` | `'forward'`, `'backward'`, or `'lambda'` |
