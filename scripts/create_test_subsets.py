#!/usr/bin/env python3
"""
Construct trajectory subsets for tests/data/ from full-length sources.

The committed subsets are primary artefacts -- the fixed ground truth the
test suite and reference baselines are built on -- and are not kept in sync
with their sources. This script documents their original construction
executably, and is the tool for deliberately building NEW subsets;
replacing a committed file is a reviewed change made together with
regenerated reference baselines (see tests/data/README.md). Construction
recipes:

- LAMMPS Example 1 and 2: first 10 frames of the dump file, split on
  "ITEM: TIMESTEP" so the subset is a byte-identical head of the source;
  the LAMMPS data file is copied verbatim.
- Water (Example 4): first 10 TRR frames rewritten via MDAnalysis
  (positions and forces verified against the source); the tpr is copied
  verbatim.
- VASP (Example 3, BaSnF4): the vasprun.xml header, the first 10
  <calculation> blocks, and the closing finalpos/footer.

The default source paths point at the full-length datasets, which are NOT in
the repository: `examples/` and `tests/test_data/` are local-only gitignored
symlinks (or directories) holding the original full trajectories. Anyone
without them can still trust the committed subsets via the sha256 checksums
recorded in tests/data/README.md.

Every file is written to a temporary path, verified, and only then moved
atomically over the target; a failed verification deletes the temporary
file and leaves any existing target untouched. Verification: text formats
are byte-compared against the head of their source (or copied verbatim and
byte-compared in full), the VASP subset is compared block-by-block against
the parsed source (agnostic to the whitespace between <calculation>
blocks, so a file written by this script verifies against itself), and the
rewritten TRR is reloaded and its positions/forces compared against the
source frames.

Existing target files are never overwritten unless --force is passed;
without it they are reported and skipped.

Usage:
    python scripts/create_test_subsets.py [--force] [--n-frames 10]
        [--example1-dir PATH] [--example2-dir PATH]
        [--vasp-xml PATH] [--water-dir PATH]
"""

import argparse
import hashlib
import os
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


def temp_path(dst: Path) -> Path:
    """Temporary sibling path for dst (same filesystem, keeps the suffix).

    The suffix is preserved because format-sniffing writers/readers
    (MDAnalysis) select the file format from it.
    """
    return dst.with_name(f"{dst.stem}.tmp-{os.getpid()}{dst.suffix}")


def write_verified(dst: Path, force: bool, write, verify) -> str:
    """Write dst via a temporary file, verifying before the atomic replace.

    ``write(tmp)`` produces the candidate file at the temporary path and
    ``verify(tmp)`` raises on any mismatch. Only a verified candidate is
    moved over dst (atomically, same filesystem); on any failure the
    temporary file is removed and an existing dst is left untouched.

    Returns "written", or "skipped" when dst exists and force is False.
    """
    if dst.exists() and not force:
        print(f"  {dst}: exists, skipping (use --force)")
        return "skipped"
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = temp_path(dst)
    try:
        write(tmp)
        verify(tmp)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, dst)
    report(dst)
    return "written"


def copy_verbatim(src: Path, dst: Path, force: bool) -> str:
    """Copy a file unchanged, verifying the copy byte-identically."""

    def verify(tmp: Path) -> None:
        if tmp.read_bytes() != src.read_bytes():
            raise RuntimeError(f"Verbatim copy differs from source: {dst}")

    return write_verified(
        dst, force, lambda tmp: shutil.copyfile(src, tmp), verify
    )


def create_lammps_subset(
    src_dir: Path, dst_dir: Path, n_frames: int, force: bool
) -> list[str]:
    """Cut the first n_frames of a LAMMPS dump; copy the data file verbatim.

    Frames are delimited by "ITEM: TIMESTEP" lines, so the subset is a
    byte-identical head of the source dump.
    """
    src_dump = src_dir / "dump.nh.lammps"
    dst_dump = dst_dir / "dump.nh.lammps"

    def write(tmp: Path) -> None:
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
        end = (
            boundaries[n_frames]
            if len(boundaries) > n_frames
            else src_dump.stat().st_size
        )
        with open(src_dump, "rb") as f:
            tmp.write_bytes(f.read(end))

    def verify(tmp: Path) -> None:
        # The subset must be a byte-identical head of the source.
        subset = tmp.read_bytes()
        with open(src_dump, "rb") as f:
            if f.read(len(subset)) != subset:
                raise RuntimeError(
                    f"Subset is not a byte-identical head: {dst_dump}"
                )

    return [
        write_verified(dst_dump, force, write, verify),
        copy_verbatim(
            src_dir / "data.fin.nh.data", dst_dir / "data.fin.nh.data", force
        ),
    ]


def _parse_vasp(content: str) -> tuple[str, list[str], str]:
    """Split a vasprun.xml into header, <calculation> blocks, and footer."""
    first_calc = content.find("<calculation>")
    if first_calc == -1:
        raise RuntimeError("No <calculation> blocks found")
    calcs = re.findall(r"<calculation>.*?</calculation>", content, re.DOTALL)
    last_calc_end = content.rfind("</calculation>") + len("</calculation>")
    return content[:first_calc], calcs, content[last_calc_end:]


def create_vasp_subset(
    src_xml: Path, dst_xml: Path, n_frames: int, force: bool
) -> list[str]:
    """Extract the header, first n_frames <calculation> blocks, and footer."""
    content = src_xml.read_text()
    try:
        header, calcs, footer = _parse_vasp(content)
    except RuntimeError as exc:
        raise RuntimeError(f"{exc} in {src_xml}") from None
    if len(calcs) < n_frames:
        raise RuntimeError(
            f"{src_xml} holds only {len(calcs)} calculation blocks; "
            f"{n_frames} requested."
        )

    def write(tmp: Path) -> None:
        subset = (
            header + "".join(c + "\n" for c in calcs[:n_frames]) + "\n" + footer
        )
        tmp.write_text(subset)

    def verify(tmp: Path) -> None:
        # Compare the parsed structure, not raw bytes: the whitespace
        # separating <calculation> blocks differs between an original
        # vasprun.xml ("\n " continuation indentation) and a file written
        # by this script (plain newlines), and carries no content. This
        # keeps verification meaningful whichever kind of source is used.
        sub_header, sub_calcs, sub_footer = _parse_vasp(tmp.read_text())
        if sub_header != header:
            raise RuntimeError(f"Header does not match {src_xml}")
        if sub_calcs != calcs[:n_frames]:
            raise RuntimeError(
                f"Calculation blocks do not match the first {n_frames} "
                f"blocks of {src_xml}"
            )
        if sub_footer.strip() != footer.strip():
            raise RuntimeError(f"Footer does not match {src_xml}")

    return [write_verified(dst_xml, force, write, verify)]


def create_water_subset(
    src_dir: Path, dst_dir: Path, n_frames: int, force: bool
) -> list[str]:
    """Rewrite the first n_frames TRR frames via MDAnalysis; copy the tpr."""
    import MDAnalysis as mda

    src_tpr = src_dir / "prod.tpr"
    src_trr = src_dir / "prod_100frames.trr"
    dst_tpr = dst_dir / "prod.tpr"
    dst_trr = dst_dir / "prod.trr"

    statuses = [copy_verbatim(src_tpr, dst_tpr, force)]

    universe = mda.Universe(str(src_tpr), str(src_trr))

    def write(tmp: Path) -> None:
        with mda.Writer(str(tmp), n_atoms=len(universe.atoms)) as writer:
            for i, _ in enumerate(universe.trajectory):
                if i >= n_frames:
                    break
                writer.write(universe.atoms)

    def verify(tmp: Path) -> None:
        # Reload the written file and compare positions and forces against
        # the source frames (TRR round-trips these exactly). The source tpr
        # supplies the topology so verification does not depend on whether
        # the tpr copy above was skipped.
        check = mda.Universe(str(src_tpr), str(tmp))
        if len(check.trajectory) != n_frames:
            raise RuntimeError(
                f"{dst_trr} holds {len(check.trajectory)} frames, "
                f"expected {n_frames}"
            )
        for i, _ in enumerate(check.trajectory):
            universe.trajectory[i]
            if not np.array_equal(
                check.atoms.positions, universe.atoms.positions
            ):
                raise RuntimeError(f"Positions differ at frame {i} in {dst_trr}")
            if not np.array_equal(check.atoms.forces, universe.atoms.forces):
                raise RuntimeError(f"Forces differ at frame {i} in {dst_trr}")

    tmp_trr = temp_path(dst_trr)
    try:
        statuses.append(write_verified(dst_trr, force, write, verify))
    finally:
        # MDAnalysis caches frame offsets for the temporary TRR in hidden
        # sibling files; remove them so no per-pid junk accumulates.
        for suffix in ("_offsets.npz", "_offsets.lock"):
            (tmp_trr.parent / f".{tmp_trr.name}{suffix}").unlink(
                missing_ok=True
            )
    return statuses


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Construct trajectory subsets in tests/data/ from full-length "
            "sources. The committed subsets are primary artefacts; replacing "
            "one is a deliberate, reviewed change (see tests/data/README.md), "
            "so existing files are skipped unless --force is passed."
        )
    )
    parser.add_argument(
        "--force", action="store_true",
        help="replace existing target files (default: report and skip them)",
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
             args.example1_dir, OUTPUT_DIR / "example_1_LJ", args.n_frames,
             args.force)),
        ("example_2_LJ_3D",
         lambda: create_lammps_subset(
             args.example2_dir, OUTPUT_DIR / "example_2_LJ_3D", args.n_frames,
             args.force)),
        ("example_3_vasp",
         lambda: create_vasp_subset(
             args.vasp_xml, OUTPUT_DIR / "example_3_vasp" / "vasprun.xml",
             args.n_frames, args.force)),
        ("example_4_water",
         lambda: create_water_subset(
             args.water_dir, OUTPUT_DIR / "example_4_water", args.n_frames,
             args.force)),
    ]

    failures = []
    written = 0
    skipped = 0
    for name, build in datasets:
        print(f"{name}:")
        try:
            statuses = build()
        except Exception as exc:
            failures.append(name)
            print(f"  FAILED: {type(exc).__name__}: {exc}")
        else:
            written += statuses.count("written")
            skipped += statuses.count("skipped")
        print()

    if failures:
        print(f"ERROR: failed datasets: {', '.join(failures)}")
        sys.exit(1)
    print(
        f"Done: {written} file(s) written and verified, "
        f"{skipped} existing file(s) skipped, in {OUTPUT_DIR}."
    )


if __name__ == "__main__":
    main()
