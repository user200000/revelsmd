# Claude Code Guidelines

## Development Process

Develop software through methodical, collaborative, and incremental approaches that prioritize careful planning. Only produce code when asked for it.

Follow a pair programming and TDD process:
- Do not provide code unless specifically asked for it
- Only provide the parts of code that are needed rather than entire updated files

## Testing Philosophy

- Tests should be as simple as possible while testing the desired behaviour
- Tests should be well isolated where possible
- Tests should test the intended desired behaviour, not legacy behaviour that is targeted for removal

## Additional Instructions

- Do not acknowledge Claude in commits, pull requests, etc.
- Do not include "Test plan" sections in pull request descriptions
- Use British spellings in commit messages and pull request descriptions
- Do not use Unicode characters (e.g., subscripts like H₂O) in commits, PRs, or issues - use plain text (e.g., H2O)

## Known Issues

### MDA RDF Bug (revels_rdf.py:331)
`RevelsRDF.run_rdf()` fails with MDA trajectories. Uses `.trajectory.atoms` instead of `.atoms`.
Workaround: Use `run_rdf_lambda()` which has the correct implementation.

### Non-Pythonic stop=-1 Handling
Frame selection uses `stop % frames`, so `stop=-1` processes frames 0 to frames-2 (loses last frame).
Fix planned in ABC PR.

### RDF Performance
RDF calculation is O(N^2) with Python loop (~2s/frame for 2304 atoms). Acceptable for intended use cases.

### Rigid Molecule Unequal Atom Counts (issue #10)
`rigid=True` mode fails when species have unequal atom counts due to index misalignment in triangular allocation.

## Integration Tests

See `tests/integration/README.md` for details on the integration test suite.

Run tests: `pytest tests/integration/ -v`
Generate reference data: `python scripts/generate_reference_data.py`
Generate validation plots: `python scripts/generate_validation_plots.py`
