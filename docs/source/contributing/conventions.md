# Conventions

## Testing philosophy

Tests should be as simple as possible while covering the intended behaviour. Aim for well-isolated tests that do not depend on unrelated parts of the codebase. Test what the code is meant to do, not incidental legacy behaviour that is due to be removed.

## Branch workflow

Always work in a feature branch. Create a branch from `main` (or from another branch if you are building on work-in-progress), make your changes, and open a pull request.

```bash
git switch -c my-feature main
```

Avoid committing directly to `main`.

## Commit style

- Use British spellings in commit messages and pull request descriptions (e.g. "initialise", "behaviour", "recognise").
- Do not use Unicode characters such as subscripts or superscripts in commit messages or PR descriptions. Use plain ASCII equivalents instead (e.g. write `H2O`, not `H₂O`).
- Keep the first line of a commit message short and descriptive. Use the imperative mood ("Add support for ...", "Fix incorrect ...").
