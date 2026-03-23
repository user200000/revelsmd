# Performance tuning

## Computation backend

RevelsMD selects a backend for numerically intensive operations (grid
allocation, RDF pairwise loops) via the `REVELSMD_BACKEND` environment
variable. The backend must be set **before** importing `revelsMD`.

| Value | Description |
|-------|-------------|
| `numba` | JIT-compiled (default). Faster after the initial compilation. |
| `numpy` | Pure NumPy. No compilation step; useful for debugging or when Numba is unavailable. |

```bash
# Shell
export REVELSMD_BACKEND=numba   # default
export REVELSMD_BACKEND=numpy   # NumPy fallback
```

Or from Python before the first import:

```python
import os
os.environ['REVELSMD_BACKEND'] = 'numpy'

import revelsMD
```

## Parallel FFTs

The FFT step that converts the accumulated force grid to a density can
run on multiple threads via SciPy's FFT worker pool.

```bash
export REVELSMD_FFT_WORKERS=4    # use 4 threads
export REVELSMD_FFT_WORKERS=-1   # use all available cores
```

The default is `1` (single-threaded). Parallel FFTs are most beneficial
for large grids (high `nbins`). For small grids the threading overhead
may outweigh the gain.

Set this variable before importing `revelsMD`:

```python
import os
os.environ['REVELSMD_FFT_WORKERS'] = '4'

import revelsMD
```
