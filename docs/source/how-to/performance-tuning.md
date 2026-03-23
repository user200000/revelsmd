# Performance tuning

## Parallel FFTs

The FFT step that converts the accumulated force grid to a density can
run on multiple threads via SciPy's FFT worker pool.

```bash
export REVELSMD_FFT_WORKERS=4    # use 4 threads
export REVELSMD_FFT_WORKERS=-1   # use all available cores
```

The default is `1` (single-threaded). Set this variable before importing
`revelsMD`:

```python
import os
os.environ['REVELSMD_FFT_WORKERS'] = '4'

import revelsMD
```
