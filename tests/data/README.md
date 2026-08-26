# Committed test data

Trimmed trajectory subsets used by the integration and regression test
suites. These files are the canonical test data: everything in the
repository (tests, reference-data generation) reads from here.

These committed files are primary artefacts, not derived caches. They are
the fixed ground truth the test suite and reference baselines are built
on, and they do not track their original sources: if a full-length source
trajectory ever changes, the committed subsets deliberately stay as they
are. The sha256 checksums below pin their integrity — verify with
`shasum -a 256` if in doubt.

The sections below record how each subset was originally constructed.
`scripts/create_test_subsets.py` documents that construction executably
and is the tool for deliberately building NEW subsets (more frames, a new
system) — replacing a committed file is a reviewed change, made together
with regenerated reference baselines and a PR explanation, never a routine
"sync". The full-length sources are NOT in the repository: `examples/` and
`tests/test_data/` are local-only gitignored paths.

## example_1_LJ

Lennard-Jones fluid, 2880 atoms (2304 of type 1, 576 of type 2), 10 frames.

- Source: `examples/example_1_LJ/` (full dump: 50 frames)
- `dump.nh.lammps`: first 10 frames, split on `ITEM: TIMESTEP`
  (byte-identical head of the source dump)
- `data.fin.nh.data`: verbatim copy
- Construction command: `python scripts/create_test_subsets.py`
  (or `--example1-dir PATH` to point at a different source)

| File | sha256 |
| --- | --- |
| `dump.nh.lammps` | `29f07f0ddd4c7fce2b0e02f6c8fdac5789dcb69600f4d58a1b9350311ad1a299` |
| `data.fin.nh.data` | `b352029323dc70214f737a8a749d4c84f0a213855de4549c1e2f68331c362931` |

## example_2_LJ_3D

Lennard-Jones 3D density example (frozen central particle plus solvating LJ
spheres), 2880 atoms, 10 frames.

- Source: `examples/example_2_LJ_3D/` (full dump: ~584 MB)
- `dump.nh.lammps`: first 10 frames, split on `ITEM: TIMESTEP`
  (byte-identical head of the source dump)
- `data.fin.nh.data`: verbatim copy
- Construction command: `python scripts/create_test_subsets.py`
  (or `--example2-dir PATH`)

| File | sha256 |
| --- | --- |
| `dump.nh.lammps` | `12f7bb9250a594aa555e6a22e30d94db482a09e51f5fa2cc01e1bb6761dba56e` |
| `data.fin.nh.data` | `2f07dce4b64f2eccf6e6a002070c799613a50257201e34af79c58dc4fb83f6bb` |

## example_3_vasp

BaSnF4 solid electrolyte AIMD (VASP), 324 atoms (54 Ba, 54 Sn, 216 F),
10 calculation steps at 600 K.

- Source: `examples/example_3_BaSnF4/r1/vasprun.xml` (3001 calculation
  blocks, ~134 MB)
- `vasprun.xml`: XML header, first 10 `<calculation>` blocks (re-serialised
  with single-newline separators), and the closing finalpos/footer
- Construction command: `python scripts/create_test_subsets.py`
  (or `--vasp-xml PATH`)

| File | sha256 |
| --- | --- |
| `vasprun.xml` | `8600665f68ad290bbfc1f86570d22979d0e2f198c9a703e1c98b72741c530e0e` |

## example_4_water

SPC/E rigid water (GROMACS), 6339 atoms (2113 molecules), 10 frames at
300 K.

- Source: `tests/test_data/example_4_subset/` (prod_100frames.trr, itself
  cut from the full `examples/example_4_rigid_water/prod.trr`)
- `prod.trr`: first 10 frames rewritten via MDAnalysis (positions,
  velocities and forces preserved; verified by reload-and-compare)
- `prod.tpr`: verbatim copy
- Construction command: `python scripts/create_test_subsets.py`
  (or `--water-dir PATH`)

| File | sha256 |
| --- | --- |
| `prod.trr` | `6ef57d66179c59e391d30af6df517168812ba1c82727fa7253d1fa66a20f542e` |
| `prod.tpr` | `617704b93199215ceb400e429743b310d92bede8c889b2e4783331fec0285871` |
