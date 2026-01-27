# Integration Test Suite

This directory contains integration and regression tests for RevelsMD.

## Test Organisation

| File | Purpose | Runtime |
|------|---------|---------|
| `test_analytical_reference.py` | Validates against known mathematical results using synthetic data | ~5s |
| `test_loader_analysis_matrix.py` | Tests all loader x analysis combinations | ~30s |
| `test_cross_backend_consistency.py` | Verifies same data produces identical results across backends | ~45s |
| `test_regression.py` | Compares against stored reference data | ~45s |
| `test_pipeline_*.py` | End-to-end workflow tests for each example | varies |

## Running Tests

```bash
# Run all integration tests (excluding slow tests)
pytest tests/integration/ -v

# Run only regression tests
pytest tests/integration/test_regression.py -v

# Run only analytical tests (no external data needed)
pytest tests/integration/test_analytical_reference.py -v

# Include slow tests
pytest tests/integration/ -v --run-slow

# Run tests for a specific backend
pytest tests/integration/ -v -k "lammps"
pytest tests/integration/ -v -k "mda"
pytest tests/integration/ -v -k "vasp"
```

## Test Data Requirements

| Dataset | Size | Location | Required For |
|---------|------|----------|--------------|
| Example 1 (LAMMPS LJ) | 12MB | `examples/example_1_LJ/` | LAMMPS tests |
| Example 3 subset (VASP BaSnF4) | 2.5MB | `tests/test_data/example_3_vasp_subset/` | VASP tests |
| Example 4 subset (GROMACS water) | 22MB | `tests/test_data/example_4_subset/` | MDA tests |
| Synthetic | 0MB | Generated at runtime | Analytical tests |

Tests skip gracefully when data is unavailable.

### Canonical Example Data

Full canonical example datasets are in `examples/` (large, not needed for routine testing):

| Example | Size | Contents |
|---------|------|----------|
| Example 1 (LJ RDF) | 12MB | LAMMPS LJ fluid for RDF |
| Example 2 (LJ 3D) | 563MB | LAMMPS 3D density |
| Example 3 (BaSnF4) | 3.6GB | VASP AIMD solid electrolyte (10 replicas) |
| Example 4 (Water) | 1.8GB | GROMACS rigid water |

## Reference Data

Reference data for regression tests is stored in `tests/reference_data/` (git-ignored):

```
tests/reference_data/
    lammps_example1/     # LAMMPS RDF and density references
    mda_example4/        # MDA/GROMACS water references
    vasp_example3/       # VASP BaSnF4 references
    synthetic/           # Synthetic trajectory references
```

To regenerate reference data from the current code:
```bash
python scripts/generate_reference_data.py
```

**Warning**: Only regenerate references when you are confident the current code is correct, as this establishes the baseline for future regression detection.

## Known Issues

### MDA RDF Bug (revels_rdf.py:331)

**Status**: Documented, not fixed

`RevelsRDF.run_rdf()` fails with MDA trajectories due to incorrect attribute access:
```python
# Current (broken):
TS.mdanalysis_universe.trajectory.atoms.positions

# Should be:
TS.mdanalysis_universe.atoms.positions
```

**Workaround**: Use `run_rdf_lambda()` which has the correct implementation.

**Affected tests**: Marked with `@pytest.mark.xfail`
- `TestMDAAnalysisMatrix::test_rdf_like`
- `TestMDAAnalysisMatrix::test_rdf_unlike`
- `TestMDAVsNumpyConsistency::test_rdf_identical`

### Non-Pythonic stop=-1 Handling

**Status**: Documented, fix planned in ABC PR

Frame selection uses `stop % frames` which means `stop=-1` processes frames `0` to `frames-2` (loses the last frame), rather than the Pythonic behaviour of processing all frames.

**Workaround**: Use explicit frame counts rather than `stop=-1` in tests.

### RDF Performance

**Status**: Known limitation

RDF calculation is O(N^2) with a Python loop in `single_frame_rdf_like`, taking ~2 seconds per frame for 2304 atoms. This makes integration tests slow but is acceptable for the force-sampling method's intended use cases.

### Rigid Molecule Unequal Atom Counts

**Status**: Documented, issue #10

When using `rigid=True` mode with species that have unequal atom counts, the triangular allocation can fail due to index misalignment.

**Affected tests**: Marked with `pytest.skip` or `@pytest.mark.xfail`

## Test Markers

| Marker | Description |
|--------|-------------|
| `@pytest.mark.integration` | All integration tests |
| `@pytest.mark.analytical` | Tests against known mathematical results |
| `@pytest.mark.regression` | Tests against stored reference data |
| `@pytest.mark.slow` | Long-running tests (skipped by default) |
| `@pytest.mark.requires_example1` | Requires Example 1 LAMMPS data |
| `@pytest.mark.requires_example4` | Requires Example 4 GROMACS data |
| `@pytest.mark.requires_vasp` | Requires VASP vasprun.xml |

## Adding New Tests

1. **Analytical tests**: Add to `test_analytical_reference.py` using synthetic `NumpyTrajectoryState` fixtures
2. **Loader/analysis combinations**: Add to `test_loader_analysis_matrix.py`
3. **Regression tests**: Add computation to `scripts/generate_reference_data.py`, then add test to `test_regression.py`

## Validation Plots

For visual inspection of results:
```bash
python scripts/generate_validation_plots.py
```

Output is saved to `tests/validation_plots/` (git-ignored).
