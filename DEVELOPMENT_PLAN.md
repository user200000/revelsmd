# RevelsMD Development Plan

This document outlines the planned work to prepare RevelsMD for public release, based on a thorough code review conducted in January 2025.

## Current Status

RevelsMD is a scientifically sound Python package for calculating reduced-variance density fields and radial distribution functions from molecular dynamics simulations. The core algorithms are well-implemented and the package is now in a usable state.

**Completed since initial review:**

1. ~~Import-breaking syntax errors~~ - Fixed (PR merged)
2. ~~Dipole calculation bug~~ - Fixed (PR merged)
3. ~~File handle leaks~~ - Fixed (PR merged)
4. GitHub Actions CI workflow added
5. Unit test suite expanded
6. TrajectoryState ABC refactoring complete
7. RDF error handling improved (raises ValueError instead of print+return None)

**Remaining work:**

1. Code style inconsistencies and opportunities for simplification
2. Additional validation tests against analytical references
3. Performance optimisation opportunities

---

## Phase 1: Critical Fixes (Required Before Any Other Work)

### 1.1 Fix Import-Breaking Syntax Errors

**Priority:** Critical
**Status:** ✅ Complete

~~Commit `4065f42` introduced tab/space mixing that causes a `TabError` on import.~~

Fixed by restoring proper indentation structure.

### 1.2 Fix Early Return Bug in Dipole Calculation

**Priority:** High
**Status:** ✅ Complete

~~In `find_coms()`, the `return` statement was inside the loop causing only the first species to contribute to dipole.~~

Fixed by moving `return` statement outside the loop.

### 1.3 Close File Handles for LAMMPS Trajectories

**Priority:** Medium
**Status:** ✅ Complete

~~File handles opened in `make_force_grid` and `run_rdf` were never closed.~~

Fixed by using context managers (`with open(...) as f:`).

---

## Phase 2: Validation Test Suite

### 2.1 Analytical Reference Tests

Create tests using systems with mathematically known results:

| Test System | Known Property | Validates |
|-------------|---------------|-----------|
| Single atom at known position | Density peak at exact location | Grid binning, coordinate handling |
| Uniform random gas | g(r) = 1, flat density | Normalization, FFT pipeline |
| Two atoms at fixed separation | g(r) peak at separation distance | Minimum image, RDF calculation |
| Crystal lattice (FCC/BCC) | Peaks at lattice spacings | Periodic boundaries, averaging |
| Neutral dipole pair | Known dipole moment | Charge/polarisation density |

### 2.2 Internal Consistency Tests

- Same trajectory loaded via different backends produces identical results
- Forward and backward RDF integration converge to same answer
- Different grid resolutions produce same bulk density
- Force conservation for rigid molecules

### 2.3 Regression Tests

Once the code is validated:
- Generate reference outputs for representative systems
- Store as `.npz` files in `tests/reference_data/`
- Compare future runs against stored references

### 2.4 Test Data Requirements

Obtain or generate minimal test trajectories:

| System | Format | Purpose |
|--------|--------|---------|
| LJ fluid | LAMMPS dump | RDF, 3D number density |
| LJ fluid | NumPy arrays | Backend comparison |
| Rigid water | GROMACS .trr/.tpr | Charge density, polarisation, rigid molecules |
| Ionic solid | VASP vasprun.xml | AIMD forces, charge density |

---

## Phase 3: Code Quality Improvements

### 3.1 Extract Common Utilities

**Minimum image convention** is implemented 4 times with slight variations. Extract to a single utility function:

```python
def minimum_image(displacement, box):
    """Apply minimum image convention to displacement vectors."""
    return displacement - np.round(displacement / box) * box
```

**Files affected:**
- `revelsMD/revels_rdf.py` - lines 109-111, 205-207
- `revelsMD/revels_3D.py` - lines 910-912, 927-929

### 3.2 Simplify NumPy Operations

Several operations use unnecessarily verbose patterns:

**Pairwise distances** (`revels_rdf.py:100-106`):
```python
# Current: explicit loop
for x in range(ns):
    rx[x, :] = pos_ang[:, 0] - pos_ang[x, 0]
    # ... etc

# Proposed: broadcasting
r_vec = pos_ang[np.newaxis, :, :] - pos_ang[:, np.newaxis, :]
```

**k-vector broadcasting** (`revels_3D.py:377-382`):
```python
# Current: explicit repeat
xrep_3d = np.repeat(np.repeat(xrep[:, None, None], self.nbinsy, axis=1), self.nbinsz, axis=2)

# Proposed: automatic broadcasting
ksquared = xrep[:, None, None]**2 + yrep[None, :, None]**2 + zrep[None, None, :]**2
```

**Bin accumulation** (`revels_rdf.py:130-135`):
```python
# Current: explicit loop
for l in range(n_bins - 2, -1, -1):
    mask = digtized_array == l
    if np.any(mask):
        storage_array[l] = np.sum(dp[mask])

# Proposed: bincount
storage_array = np.bincount(digitized_array, weights=dp, minlength=n_bins)
```

### 3.3 Consolidate x/y/z Arrays

Throughout the codebase, x/y/z components are handled as separate arrays:
- `forceX`, `forceY`, `forceZ`
- `box_x`, `box_y`, `box_z`
- `rx`, `ry`, `rz`

Consider consolidating to arrays with shape `(..., 3)`:
- Enables vectorized operations over components
- Reduces code duplication
- Simplifies function signatures

**Note:** This is a larger refactoring that should only be done after the test suite is in place.

### 3.4 Deprecate Spelling Aliases

**Status:** ✅ Partially complete

| Location | Alias | Correct Name |
|----------|-------|--------------|
| `MDATrajectoryState` | `get_indicies` | `get_indices` |
| `NumpyTrajectoryState` | `get_indicies` | `get_indices` |
| `LammpsTrajectoryState` | `get_indicies` | `get_indices` |
| `revels_3D.SelectionState` | `get_indicies` | `get_indices` |

Progress:
1. ✅ Deprecation warnings added to TrajectoryState ABC (inherited by subclasses)
2. [ ] Update any documentation referencing the old spelling
3. [ ] Remove aliases in a future major version

### 3.5 Consistent Error Handling

**Status:** ✅ Partially complete

~~`run_rdf` returns `None` and prints to stdout on error.~~

Progress:
- ✅ `run_rdf` and `run_rdf_lambda` now raise `ValueError` for invalid frame bounds (PR #17)
- [ ] Review other functions for similar patterns

---

## Phase 4: Performance Optimisation (Future)

### 4.1 Profile Before Optimising

Identify actual bottlenecks using:
```python
import cProfile
cProfile.run('RevelsRDF.run_rdf(...)', sort='cumtime')
```

Likely candidates:
- `triangular_allocation` - called once per frame, deposits to 8 voxels per particle
- `single_frame_rdf_like/unlike` - O(N²) pairwise operations

### 4.2 Numba JIT Compilation

The explicit loop-unrolled style in `triangular_allocation` and the RDF pairwise loops are good candidates for Numba:

```python
from numba import jit

@jit(nopython=True)
def triangular_allocation_numba(...):
    # Existing implementation works with minimal changes
```

### 4.3 Consider Existing Libraries

For particle-mesh operations:
- `pmesh` - particle-mesh operations with CIC support
- `scipy.ndimage` - for some grid operations

---

## Phase 5: Release Preparation

### 5.1 Infrastructure

- [x] Add GitHub Actions CI workflow for automated testing
- [ ] Add `CONTRIBUTING.md` with development guidelines
- [ ] Add `CHANGELOG.md`

### 5.2 Code Quality Tooling

- [ ] Add linting configuration (ruff or flake8)
- [ ] Add formatting configuration (black, isort)
- [ ] Add type checking configuration (mypy)
- [ ] Add pre-commit hooks

### 5.3 Documentation

- [ ] Review and update docstrings
- [ ] Ensure notebooks run with current code
- [ ] Add installation instructions to README
- [ ] Verify ReadTheDocs builds correctly

### 5.4 Package Distribution

- [ ] Verify `pyproject.toml` is complete
- [ ] Test installation via pip
- [ ] Consider PyPI publication

---

## Phase 6: Future Development Targets

These targets were identified during recent refactoring work.

### 6.1 Modern Type Hints

**Priority:** Low
**Status:** Not started

Update type annotations to Python 3.11+ syntax:
- `list[str]` instead of `List[str]`
- `dict[str, int]` instead of `Dict[str, int]`
- `X | None` instead of `Optional[X]`
- Remove `from __future__ import annotations` where no longer needed

### 6.2 Non-Orthorhombic Cell Support

**Priority:** Medium
**Status:** Not started

Currently the code only supports orthorhombic (rectangular) simulation cells. Supporting triclinic cells would enable analysis of a wider range of simulations.

**Required changes:**
- Store full lattice matrix instead of scalar box dimensions (`box_x`, `box_y`, `box_z`)
- Update minimum image convention to use lattice vectors
- Calculate volume from lattice determinant
- Calculate `rmax` from inscribed sphere radius
- Update grid binning for non-orthogonal coordinates

**Files affected:**
- `revelsMD/trajectory_states.py` - box storage
- `revelsMD/revels_3D.py` - minimum image, grid operations
- `revelsMD/revels_rdf.py` - minimum image, distance calculations

### 6.3 Boltzmann Factors in Trajectory States

**Priority:** Low
**Status:** Not started

Temperature (and hence kT for Boltzmann weighting) should be stored/looked up directly in the trajectory state rather than passed as a separate parameter. This would simplify the API and reduce the chance of inconsistent temperature values.

### 6.4 Estimator API Simplification

**Priority:** Medium
**Status:** Not started

The current 10 "estimator" functions are actually the same algorithm with different preprocessing. Consider consolidating to a unified `compute_density()` API as outlined in `DESIGN_NOTES.md`.

**Benefits:**
- Single entry point for all density calculations
- Preprocessing logic (positions, forces, weights) in separate testable functions
- Eliminates code duplication and reduces bug risk

---

## Appendix: Detailed Bug Descriptions

### A.1 TabError in `revels_3D.py`

**Commit:** `4065f42`
**Error:** `TabError: inconsistent use of tabs and spaces in indentation`

The commit attempted to add validation that rigid molecule selections have matching atom counts (e.g., same number of O and H atoms for water). However, the changes used tabs while the rest of the file uses spaces, and the control flow structure was scrambled.

**Affected code blocks:**
1. `make_force_grid` (lines 175-186): Atom name validation block uses tabs
2. `SelectionState.__init__` (lines 658-695): Mixed tabs/spaces, `if/else` structure broken

### A.2 Early Return in `find_coms`

**Location:** `revels_3D.py`, approximately line 930

```python
if calc_dipoles:
    charges = GS.SS.charges[0]
    charges_cumulant = charges[:, np.newaxis] * (positons[SS.indices[0]] - coms)
    for species_index in range(1, len(SS.indices)):
        seperation = (positons[SS.indices[species_index]] - coms)
        # ... minimum image correction ...
        charges_cumulant += charges[species_index] * seperation
        molecular_dipole = charges_cumulant
        return coms, molecular_dipole  # BUG: returns inside loop!
```

The `return` statement is inside the `for` loop, so for a water molecule with species `['O', 'H', 'H']`, only the first H contributes to the dipole before returning.

### A.3 Unclosed File Handles

**Locations:**
- `revels_3D.py:255` - `f = open(TS.trajectory_file)` in `make_force_grid`
- `revels_3D.py:497` - `f = open(TS.trajectory_file)` in `get_lambda`
- `revels_rdf.py:318` - `f = open(TS.trajectory_file)` in `run_rdf`
- `revels_rdf.py:445` - `f = open(TS.trajectory_file)` in `run_rdf_lambda`

None of these file handles are closed after use.

---

## Appendix: Test Coverage Gaps

Based on review of `tests/` directory:

| Module | Current Coverage | Key Gaps |
|--------|------------------|----------|
| `revels_3D.py` | Basic pipeline only | No direct estimator tests, no kernel validation, no lambda method validation |
| `revels_rdf.py` | Shape/finiteness checks | No numerical correctness tests, no cross-validation |
| `trajectory_states.py` | Mock-based init tests | No real file I/O tests |
| `lammps_parser.py` | Basic parsing | No malformed file handling |
| `vasp_parser.py` | Basic parsing | No multi-file tests |
| `conversion_factors.py` | Complete | - |

---

*Document created: January 2025*
*Last updated: January 2025*

---

## Appendix: Completed Work Summary

| Item | PR/Commit | Date |
|------|-----------|------|
| TabError fix | Merged | Jan 2025 |
| Dipole early return bug | Merged | Jan 2025 |
| Unclosed file handles | Merged | Jan 2025 |
| GitHub Actions CI | Merged | Jan 2025 |
| TrajectoryState ABC | PR #16 | Jan 2025 |
| RDF error handling | PR #17 | Jan 2025 |
| Unit test expansion | Multiple PRs | Jan 2025 |
