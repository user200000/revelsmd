# Integration Test Suite

This directory contains integration and regression tests for RevelsMD. All
tests run everywhere, including CI: the trajectory data they need is
committed to the repository, no test skips on missing data, and a green run
verifies every committed baseline.

## Test Organisation

| File | Purpose |
|------|---------|
| `test_regression.py` | Value-pinning tests against committed reference data (five baseline families) plus reference-data integrity checks |
| `test_analytical_reference.py` | Tests against known mathematical results using synthetic data, including the exact harmonic-pair g(r) validation of both estimators |
| `test_cross_backend_consistency.py` | Verifies the same data produces consistent results across trajectory backends, plus resolution-consistency (bulk density across grid sizes) and kernel-consistency (triangular vs box) checks |
| `test_pipeline_rigid_water.py` | Loader-property tests on the real water topology (charge availability and neutrality) |

The whole suite runs in well under a minute.

## Running Tests

```bash
# All integration tests
pytest tests/integration/ -v

# Only regression tests
pytest tests/integration/test_regression.py -v

# Only analytical tests
pytest tests/integration/test_analytical_reference.py -v

# Tests for a specific backend
pytest tests/integration/ -v -k "lammps"
```

## Test Data

Trimmed trajectory subsets (about 8 MB total) are committed in `tests/data/`:

| Dataset | Location | Contents |
|---------|----------|----------|
| Example 1 (LAMMPS LJ) | `tests/data/example_1_LJ/` | First 10 frames of the LJ fluid dump plus topology |
| Example 2 (LAMMPS LJ 3D) | `tests/data/example_2_LJ_3D/` | First 10 frames plus topology |
| Example 3 (VASP BaSnF4) | `tests/data/example_3_vasp/` | First 10 calculation steps of vasprun.xml |
| Example 4 (GROMACS water) | `tests/data/example_4_water/` | First 10 frames of the rigid-water TRR plus TPR |
| Synthetic | generated at runtime | Analytical and synthetic-baseline tests |

Missing data files fail tests loudly: absence means a broken checkout, not a
skippable condition. Full-length trajectories are not needed by anything in
the repository; they remain useful only for ad-hoc high-statistics
validation during estimator development.

## Reference Data

Baselines for the regression tests are committed in `tests/reference_data/`:

```
tests/reference_data/
    lammps_example1/     # LAMMPS RDF (forward/backward/lambda/strided/g_count) and density (triangular/box) references
    lammps_example2/     # Density lambda-combination references
    mda_example4/        # Water RDF (like and unlike pairs), number (Ow and rigid), charge and polarisation density references
    vasp_example3/       # BaSnF4 RDF and density references
    synthetic/           # Synthetic trajectory references, including rho_lambda and lambda_weights
```

Regeneration (`python scripts/generate_reference_data.py`) is a reviewed
change: commit new baselines together with the change that motivated them
and explain the numerical shift in the pull request. Never regenerate simply
to make a red test pass. See the module docstring of `test_regression.py`
for the tolerance tiers.

## Test Markers

| Marker | Description |
|--------|-------------|
| `@pytest.mark.integration` | All integration tests |
| `@pytest.mark.analytical` | Tests against known mathematical results |
| `@pytest.mark.regression` | Tests against committed reference data |
| `@pytest.mark.requires_example1` | Uses the committed Example 1 subset |
| `@pytest.mark.requires_example2` | Uses the committed Example 2 subset |
| `@pytest.mark.requires_example4` | Uses the committed Example 4 subset |
| `@pytest.mark.requires_vasp` | Uses the committed VASP subset |

## Adding New Tests

1. **Analytical tests**: add to `test_analytical_reference.py` using
   synthetic `NumpyTrajectory` fixtures. Prefer these when an exact or
   tightly-bounded expected result exists.
2. **Regression tests**: add the computation to
   `scripts/generate_reference_data.py`, regenerate, commit the new
   baseline, and add the pinning test to `test_regression.py` at the
   documented tolerance tier.
3. Avoid finiteness-only smoke tests: if a code path is worth an
   integration test, it is worth pinning a value or asserting an invariant.
