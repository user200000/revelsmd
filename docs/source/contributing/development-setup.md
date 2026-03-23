# Development Setup

## Clone the repository

```bash
git clone https://github.com/user200000/revelsmd.git
cd revelsmd
```

## Install in development mode

Install the package with the `test` extras, which include pytest and related tools:

```bash
pip install -e ".[test]"
```

To also build the documentation locally, add the `docs` extras:

```bash
pip install -e ".[test,docs]"
```

## Running tests

```bash
pytest
```

Some tests are marked with additional markers for slow or data-intensive runs.
To skip slow tests:

```bash
pytest -m "not slow"
```

## Building the documentation

```bash
cd docs && make html
```

The built HTML will be in `docs/build/html/`.
