# Conventions

## Testing philosophy

Tests should be as simple as possible while covering the intended behaviour. Aim for well-isolated tests that do not depend on unrelated parts of the codebase. Test what the code is meant to do, not incidental legacy behaviour that is due to be removed.

## Branch workflow

Always work in a feature branch. Create a branch from `main` (or from another branch if you are building on work-in-progress), make your changes, and open a pull request.

```bash
git switch -c my-feature main
```

Avoid committing directly to `main`.
