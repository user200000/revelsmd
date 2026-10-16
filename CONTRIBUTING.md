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
from ._base import Trajectory, DataUnavailableError, check_finite


class MyTrajectory(Trajectory):
    def __init__(self, path: str, ..., units: str, temperature: float) -> None:
        super().__init__(units=units, temperature=temperature)
        # Base __init__ sets self.units, self.temperature, and self.beta.
        # You must also set:
        self.path = path
        self.frames: int = ...          # total number of frames
        self.cell_matrix: np.ndarray = ...  # shape (3, 3), rows = lattice vectors
```

Calling `super().__init__()` is required — it sets `units`, `temperature`, and `beta`. After that, your `__init__` must set `frames` (an `int`) and `cell_matrix` (a `(3, 3)` NumPy array with rows equal to the lattice vectors).

### 2. Implement the required abstract methods

**`get_indices(atype)`** — return the integer indices of all atoms of a given type, and raise `ValueError` if there are none:

```python
def get_indices(self, atype: str) -> np.ndarray:
    ...
```

**`_get_frame(index)`** — return a single frame by index. The public `get_frame()` checks the index and converts negative indices first, so `0 <= index < frames`:

```python
def _get_frame(self, index: int) -> Frame:
    positions = ...  # shape (n_atoms, 3)
    forces = ...     # shape (n_atoms, 3)
    check_finite(positions, forces, f"{self.path}, frame {index}")
    return Frame(positions=positions, forces=forces)
```

**`_iter_frames_impl(start, stop, stride)`** — yield frames over a normalised slice. This is called by the public `iter_frames()` method after bounds normalisation; `start`, `stop`, and `stride` are guaranteed to be non-negative integers:

```python
def _iter_frames_impl(
    self, start: int, stop: int, stride: int
) -> Iterator[Frame]:
    for i in range(start, stop, stride):
        yield self._get_frame(i)
```

`_get_frame` and `_iter_frames_impl` should return `Frame` instances imported from `revelsMD.frame_sources`, and raise `ValueError` naming the file and frame for a frame with NaN or infinite values (`check_finite` in `revelsMD.trajectories._base`) or with a cell other than the first frame's (`check_fixed_cell`). In `__init__`, validate the cell with `self._validate_cell_matrix(cell, where)`, naming the file, and issue warnings with `warn_user`, which attributes them to the user's code. A backend that reads several files should accept them through `as_paths`, which rejects unordered collections and repeated files.

A backend whose frames are byte ranges of files can extend `IndexedFileTrajectory` instead, as the LAMMPS and VASP backends do. Its `__init__` sets `self._frames` to a `FileFrames.join(...)` table and `self.frames = len(self._frames)`, and it implements `_where(index)` (the file and frame, for messages) and `_read_arrays(f, index)` (positions and forces from the open file). `_read_arrays` raises `ValueError` without the file or frame, which the base class adds; other exceptions pass through unchanged. The base class provides `_get_frame`, `_iter_frames_impl`, that error wrapping and the finite check; it does not check the cell.

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
- `get_frame` returns a `Frame` with the expected positions and forces, including for negative indices.
- `iter_frames` yields the correct number of frames in the correct order.
- `get_charges` and `get_masses` raise `DataUnavailableError` if not implemented, or return correct values if they are.
