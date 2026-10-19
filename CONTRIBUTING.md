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

### Error and warning messages

Exception and warning messages start with a lower-case letter and have no
trailing full stop, as Python's own do: `"temperature must be positive. Got: -1.0"`.
Identifiers keep their case. A message of several sentences puts full stops
between them, but not after the last.

### Branch workflow

Always work on a feature branch. Branch from `main` unless there is a specific reason to branch from elsewhere. Open a pull request against `main` when the work is ready for review.

---

## Adding a new trajectory backend

### 1. Subclass `Trajectory`

Create a new file in `revelsMD/trajectories/`. Subclass `Trajectory` from `revelsMD.trajectories._base`:

```python
from collections.abc import Generator

import numpy as np

from revelsMD.frame_sources import Frame
from ._base import Trajectory, DataUnavailableError, as_paths


class MyTrajectory(Trajectory):
    def __init__(self, trajectory_file, *, units: str, temperature: float) -> None:
        super().__init__(units=units, temperature=temperature)
        # Base __init__ sets self.units, self.temperature, and self.beta.
        # You must also set:
        self.path = as_paths(trajectory_file)[0]
        self.frames: int = ...          # total number of frames
        self.cell_matrix: np.ndarray = ...  # shape (3, 3), rows = lattice vectors
        self._validate_cell_matrix(self.cell_matrix, self.path)
```

The `...` are for your backend to fill in.

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
    return Frame(positions=positions, forces=forces)

def _where(self, index: int) -> str:
    return f"{self.path}, frame {index}"
```

**`_iter_frames_impl(start, stop, stride)`** — a generator of frames over a normalised slice. This is called by the public `iter_frames()` method after bounds normalisation; `start`, `stop`, and `stride` are guaranteed to be non-negative integers. The base class closes the generator when iteration ends, fails or is abandoned, so release open files in a `finally` block or `with` statement:

```python
def _iter_frames_impl(
    self, start: int, stop: int, stride: int
) -> Generator[Frame, None, None]:
    for i in range(start, stop, stride):
        yield self._get_frame(i)
```

`_get_frame` and `_iter_frames_impl` should return `Frame` instances imported from `revelsMD.frame_sources`, and raise `ValueError` naming the file and frame for a frame with a cell other than the first frame's (`check_fixed_cell` in `revelsMD.trajectories._base`). The base class checks every frame for NaN and infinite values, naming it with `_where(index)`; override `_where` to name the file. In `__init__`, validate the cell with `self._validate_cell_matrix(cell, where)`, naming the file, and issue warnings with `warn_user`, which attributes them to the user's code (`warn_cut_short` for a final frame a killed job left incomplete). A backend that reads several files should accept them through `as_paths`, and open MDAnalysis files with `open_universe`, which guesses nothing, and read charges and masses with `atom_values`, which takes each atom's value from those the user gave (checked with `given_values`), else from the topology (`topology_values`), and raises `DataUnavailableError` naming the labels with neither.

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
