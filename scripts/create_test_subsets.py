#!/usr/bin/env python3
"""
Regenerate the committed trajectory subsets in tests/data/ from full sources.

The committed subsets are the canonical test data; this script documents and
reproduces how each was cut from its full-length source:

- LAMMPS Example 1 and 2: first 10 frames of the dump file, split on
  "ITEM: TIMESTEP" so the subset is a byte-identical head of the source;
  the LAMMPS data file is copied verbatim.
- Water (Example 4): first 10 TRR frames rewritten via MDAnalysis
  (positions, velocities and forces preserved); the tpr is copied verbatim.
- VASP (Example 3, BaSnF4): the vasprun.xml header, the first 10
  <calculation> blocks (re-serialised with single-newline separators), and
  the closing finalpos/footer.

The default source paths point at the full-length datasets, which are NOT in
the repository: `examples/` and `tests/test_data/` are local-only gitignored
symlinks (or directories) holding the original full trajectories. Anyone
without them can still trust the committed subsets via the sha256 checksums
recorded in tests/data/README.md.

Each written file is verified: text formats are byte-compared against the
head of their source (or copied verbatim and byte-compared in full), and the
rewritten TRR is reloaded and its positions/forces compared against the
source frames.

Usage:
    python scripts/create_test_subsets.py [--n-frames 10]
        [--example1-dir PATH] [--example2-dir PATH]
        [--vasp-xml PATH] [--water-dir PATH]
"""

import argparse
import hashlib
import re
import shutil
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parents[1]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "data"


def sha256(path: Path) -> str:
    """Return the sha256 hex digest of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def report(path: Path) -> None:
    """Print a written file with its size and checksum."""
    size = path.stat().st_size
    print(f"  Wrote {path} ({size} bytes)")
    print(f"    sha256: {sha256(path)}")


def copy_verbatim(src: Path, dst: Path) -> None:
    """Copy a file unchanged and verify the copy byte-identically."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    if dst.read_bytes() != src.read_bytes():
        raise RuntimeError(f"Verbatim copy differs from source: {dst}")
    report(dst)


def create_lammps_subset(src_dir: Path, dst_dir: Path, n_frames: int) -> None:
    """Cut the first n_frames of a LAMMPS dump; copy the data file verbatim.

    Frames are delimited by "ITEM: TIMESTEP" lines, so the subset is a
    byte-identical head of the source dump.
    """
    src_dump = src_dir / "dump.nh.lammps"
    dst_dump = dst_dir / "dump.nh.lammps"
    dst_dir.mkdir(parents=True, exist_ok=True)

    marker = b"ITEM: TIMESTEP"
    boundaries = []
    offset = 0
    with open(src_dump, "rb") as f:
        for line in f:
            if line.startswith(marker):
                boundaries.append(offset)
                if len(boundaries) > n_frames:
                    break
            offset += len(line)
    if len(boundaries) < n_frames:
        raise RuntimeError(
            f"{src_dump} holds only {len(boundaries)} frames; "
            f"{n_frames} requested."
        )
    # End of frame n_frames: the start of frame n_frames+1, or EOF.
    end = boundaries[n_frames] if len(boundaries) > n_frames else src_dump.stat().st_size

    with open(src_dump, "rb") as f:
        head = f.read(end)
    dst_dump.write_bytes(head)

    # Verify: the subset must be a byte-identical head of the source.
    with open(src_dump, "rb") as f:
        if f.read(end) != dst_dump.read_bytes():
            raise RuntimeError(f"Subset is not a byte-identical head: {dst_dump}")
    report(dst_dump)

    copy_verbatim(src_dir / "data.fin.nh.data", dst_dir / "data.fin.nh.data")


def create_vasp_subset(src_xml: Path, dst_xml: Path, n_frames: int) -> None:
    """Extract the header, first n_frames <calculation> blocks, and footer."""
    content = src_xml.read_text()
    first_calc = content.find("<calculation>")
    if first_calc == -1:
        raise RuntimeError(f"No <calculation> blocks found in {src_xml}")
    header = content[:first_calc]

    calcs = re.findall(r"<calculation>.*?</calculation>", content, re.DOTALL)
    if len(calcs) < n_frames:
        raise RuntimeError(
            f"{src_xml} holds only {len(calcs)} calculation blocks; "
            f"{n_frames} requested."
        )

    last_calc_end = content.rfind("</calculation>") + len("</calculation>")
    footer = content[last_calc_end:]

    subset = header + "".join(c + "\n" for c in calcs[:n_frames]) + "\n" + footer
    dst_xml.parent.mkdir(parents=True, exist_ok=True)
    dst_xml.write_text(subset)

    # Verify: header plus calculation blocks must byte-match the head of the
    # source. In the source, blocks are separated by "\n " (the space is part
    # of the next block's indentation, stripped by the regex); the subset
    # re-serialises them with plain newlines, so the comparison reinstates
    # the source separator.
    source_head = header + "\n ".join(calcs[:n_frames])
    if not content.startswith(source_head):
        raise RuntimeError(
            f"Extracted calculation region does not byte-match the head of {src_xml}"
        )
    if dst_xml.read_text() != subset:
        raise RuntimeError(f"Written subset does not match extraction: {dst_xml}")
    report(dst_xml)


def create_water_subset(src_dir: Path, dst_dir: Path, n_frames: int) -> None:
    """Rewrite the first n_frames TRR frames via MDAnalysis; copy the tpr."""
    import MDAnalysis as mda

    src_tpr = src_dir / "prod.tpr"
    src_trr = src_dir / "prod_100frames.trr"
    dst_tpr = dst_dir / "prod.tpr"
    dst_trr = dst_dir / "prod.trr"
    dst_dir.mkdir(parents=True, exist_ok=True)

    copy_verbatim(src_tpr, dst_tpr)

    universe = mda.Universe(str(src_tpr), str(src_trr))
    with mda.Writer(str(dst_trr), n_atoms=len(universe.atoms)) as writer:
        for i, _ in enumerate(universe.trajectory):
            if i >= n_frames:
                break
            writer.write(universe.atoms)

    # Verify: reload the written file and compare positions and forces
    # against the source frames (TRR round-trips these exactly).
    check = mda.Universe(str(dst_tpr), str(dst_trr))
    if len(check.trajectory) != n_frames:
        raise RuntimeError(
            f"{dst_trr} holds {len(check.trajectory)} frames, expected {n_frames}"
        )
    for i, _ in enumerate(check.trajectory):
        universe.trajectory[i]
        if not np.array_equal(check.atoms.positions, universe.atoms.positions):
            raise RuntimeError(f"Positions differ at frame {i} in {dst_trr}")
        if not np.array_equal(check.atoms.forces, universe.atoms.forces):
            raise RuntimeError(f"Forces differ at frame {i} in {dst_trr}")
    report(dst_trr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate the committed trajectory subsets in tests/data/."
    )
    parser.add_argument(
        "--n-frames", type=int, default=10,
        help="frames/blocks per subset (default: 10, matching the committed data)",
    )
    parser.add_argument(
        "--example1-dir", type=Path,
        default=PROJECT_ROOT / "examples" / "example_1_LJ",
        help="directory holding the full Example 1 dump.nh.lammps and data.fin.nh.data",
    )
    parser.add_argument(
        "--example2-dir", type=Path,
        default=PROJECT_ROOT / "examples" / "example_2_LJ_3D",
        help="directory holding the full Example 2 dump.nh.lammps and data.fin.nh.data",
    )
    parser.add_argument(
        "--vasp-xml", type=Path,
        default=PROJECT_ROOT / "examples" / "example_3_BaSnF4" / "r1" / "vasprun.xml",
        help="full BaSnF4 vasprun.xml",
    )
    parser.add_argument(
        "--water-dir", type=Path,
        default=PROJECT_ROOT / "tests" / "test_data" / "example_4_subset",
        help="directory holding prod_100frames.trr and prod.tpr",
    )
    args = parser.parse_args()

    datasets = [
        ("example_1_LJ",
         lambda: create_lammps_subset(
             args.example1_dir, OUTPUT_DIR / "example_1_LJ", args.n_frames)),
        ("example_2_LJ_3D",
         lambda: create_lammps_subset(
             args.example2_dir, OUTPUT_DIR / "example_2_LJ_3D", args.n_frames)),
        ("example_3_vasp",
         lambda: create_vasp_subset(
             args.vasp_xml, OUTPUT_DIR / "example_3_vasp" / "vasprun.xml",
             args.n_frames)),
        ("example_4_water",
         lambda: create_water_subset(
             args.water_dir, OUTPUT_DIR / "example_4_water", args.n_frames)),
    ]

    failures = []
    for name, build in datasets:
        print(f"{name}:")
        try:
            build()
        except (OSError, RuntimeError) as exc:
            failures.append(name)
            print(f"  FAILED: {exc}")
        print()

    if failures:
        print(f"ERROR: failed datasets: {', '.join(failures)}")
        sys.exit(1)
    print(f"All subsets written to {OUTPUT_DIR} and verified.")


if __name__ == "__main__":
    main()
