# Contributing to RevelsMD

## Development setup

Clone the repository and install in editable mode with test dependencies:

```bash
git clone https://github.com/user200000/revelsmd
cd revelsmd
pip install -e ".[test]"
```

To also build the documentation, install with:

```bash
pip install -e ".[test,docs]"
```

**Running tests:**

```bash
pytest                        # full suite
pytest -m "not slow"          # skip slow tests
```

**Building the documentation:**

```bash
cd docs && make html
```

---

## Conventions

### Testing philosophy

- Tests should be as simple as possible while testing the desired behaviour.
- Tests should be well isolated.
- Tests should test intended behaviour, not legacy behaviour targeted for removal.

### Branch workflow

Always work on a feature branch. Branch from `main` unless there is a specific reason to branch from elsewhere. Open a pull request against `main` when the work is ready for review.

---

## Adding a new trajectory backend

### 1. Subclass `Trajectory`

Create a new file in `revelsMD/trajectories/`. Subclass `Trajectory` from `revelsMD.trajectories._base`:

```python
from typing import Iterator

import numpy as np

from revelsMD.frame_sources import Frame
from ._base import Trajectory, DataUnavailableError


class MyTrajectory(Trajectory):
    def __init__(self, ..., units: str, temperature: float) -> None:
        super().__init__(units=units, temperature=temperature)
        # Base __init__ sets self.units, self.temperature, and self.beta.
        # You must also set:
        self.frames: int = ...          # total number of frames
        self.cell_matrix: np.ndarray = ...  # shape (3, 3), rows = lattice vectors
```

Calling `super().__init__()` is required — it sets `units`, `temperature`, and `beta`. After that, your `__init__` must set `frames` (an `int`) and `cell_matrix` (a `(3, 3)` NumPy array with rows equal to the lattice vectors).

### 2. Implement the required abstract methods

**`get_indices(atype)`** — return the integer indices of all atoms of a given type:

```python
def get_indices(self, atype: str) -> np.ndarray:
    ...
```

**`get_frame(index)`** — return a single frame by index:

```python
def get_frame(self, index: int) -> Frame:
    positions = ...  # shape (n_atoms, 3)
    forces = ...     # shape (n_atoms, 3)
    return Frame(positions=positions, forces=forces)
```

**`_iter_frames_impl(start, stop, stride)`** — yield frames over a normalised slice. This is called by the public `iter_frames()` method after bounds normalisation; `start`, `stop`, and `stride` are guaranteed to be non-negative integers:

```python
def _iter_frames_impl(
    self, start: int, stop: int, stride: int
) -> Iterator[Frame]:
    for i in range(start, stop, stride):
        yield self.get_frame(i)
```

All three methods should return `Frame` instances imported from `revelsMD.frame_sources`.

### 3. Optionally override `get_charges` and `get_masses`

The base class implementations raise `DataUnavailableError`. Override them if your backend has this data:

```python
def get_charges(self, atype: str) -> np.ndarray:
    indices = self.get_indices(atype)
    return self._charges[indices]

def get_masses(self, atype: str) -> np.ndarray:
    indices = self.get_indices(atype)
    return self._masses[indices]
```

### 4. Register in `__init__.py`

Add your class to `revelsMD/trajectories/__init__.py`:

```python
from .mybackend import MyTrajectory

__all__ = [
    ...,
    "MyTrajectory",
]
```

### 5. Write tests

Add a test file under `tests/`. At minimum, cover:

- Construction from valid inputs.
- `get_indices` returns the correct atom indices.
- `get_frame` returns a `Frame` with the expected positions and forces.
- `iter_frames` yields the correct number of frames in the correct order.
- `get_charges` and `get_masses` raise `DataUnavailableError` if not implemented, or return correct values if they are.
