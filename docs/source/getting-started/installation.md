# Installation

## Requirements

RevelsMD requires Python 3.11 or later.

## Installing from PyPI

```bash
pip install revelsMD
```

## Installing from source

```bash
git clone https://github.com/user200000/revelsmd.git
cd revelsmd
pip install -e .
```

## Optional dependencies

VASP trajectory support requires [pymatgen](https://pymatgen.org/), available as an optional extra:

```bash
pip install revelsMD[vasp]
```

## Dependencies

The following packages are installed automatically:

- **NumPy** (>=2.2) — array operations and numerical computation
- **SciPy** (>=1.15) — scientific computing routines
- **MDAnalysis** (>=2.8.0) — trajectory file handling
- **Numba** (>=0.61.2) — JIT compilation for performance
- **tqdm** — progress bars
- **lxml** — XML parsing

## Numba acceleration

RevelsMD uses Numba for JIT-compiled numerical kernels. On first import, Numba will compile the kernels; subsequent imports are faster.

## Verifying the installation

```python
import revelsMD
print(revelsMD.__version__)
```
