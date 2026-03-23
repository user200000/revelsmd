# Adding a New Trajectory Backend

This guide walks through adding support for a new trajectory file format or data source.

## 1. Subclass `Trajectory`

Create a new module under `revelsMD/trajectories/` and subclass `Trajectory` from `revelsMD.trajectories._base`:

```python
from revelsMD.trajectories._base import Trajectory

class MyFormatTrajectory(Trajectory):
    ...
```

## 2. Call the base `__init__`

`Trajectory.__init__` requires `units` and `temperature` as keyword arguments and sets `self.units`, `self.temperature`, and `self.beta` for you. Call it before setting your own attributes:

```python
def __init__(self, filepath: str, *, units: str, temperature: float) -> None:
    super().__init__(units=units, temperature=temperature)
    # set self.frames, self.cell_matrix, etc.
```

You must also set `self.frames` (int) and `self.cell_matrix` (a `(3, 3)` numpy array whose rows are the lattice vectors).

## 3. Implement the required abstract methods

Two methods are abstract and must be implemented:

### `get_indices(atype: str) -> np.ndarray`

Return an array of integer atom indices for the given species or type name.

```python
def get_indices(self, atype: str) -> np.ndarray:
    ...
```

### `_iter_frames_impl(start, stop, stride) -> Iterator[Frame]`

Iterate over frames with pre-normalised (non-negative) bounds. This is called internally by the public `iter_frames()` method after bounds normalisation.

```python
def _iter_frames_impl(self, start: int, stop: int, stride: int) -> Iterator[Frame]:
    for i in range(start, stop, stride):
        yield self.get_frame(i)
```

### `get_frame(index: int) -> Frame`

Return a single frame by index. This method is abstract and must be implemented. It is also used by `interleaved_blocks` for random-access blocking.

```python
def get_frame(self, index: int) -> Frame:
    ...
```

## 4. Return `Frame` objects

Both `get_frame` and `_iter_frames_impl` yield `Frame` instances from `revelsMD.frame_sources`. Construct them with positional and force arrays:

```python
from revelsMD.frame_sources import Frame

frame = Frame(positions=positions, forces=forces)
```

`Frame` validates on construction:

- Both arrays must be numpy arrays.
- Both must have shape `(n_atoms, 3)`.
- `positions` and `forces` must have the same number of atoms.

## 5. Optional: override `get_charges` and `get_masses`

The base class provides default implementations of `get_charges` and `get_masses` that raise `DataUnavailableError`. If your format includes charge or mass data, override them:

```python
def get_charges(self, atype: str) -> np.ndarray:
    ...

def get_masses(self, atype: str) -> np.ndarray:
    ...
```

## 6. Register the backend

Add your class to `revelsMD/trajectories/__init__.py`:

```python
from .myformat import MyFormatTrajectory

__all__ = [
    ...,
    "MyFormatTrajectory",
]
```

## 7. Write tests

Place tests in `tests/trajectories/`. A minimal test suite should cover:

- Correct values for `frames`, `cell_matrix`, `units`, `temperature`, and `beta` after construction.
- `get_indices` returns the expected atom indices for a known species.
- `get_frame` returns a `Frame` with the correct shapes.
- `iter_frames` yields the expected number of frames and respects `start`, `stop`, and `stride`.
- If `get_charges` or `get_masses` are implemented, that they return arrays of the right length.
- That `get_charges`/`get_masses` raise `DataUnavailableError` if not implemented.

Use small, in-memory data where possible to keep tests fast and self-contained.
