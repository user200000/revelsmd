"""Integrity checks for the committed trajectory subsets in tests/data/.

The committed subsets are primary artefacts pinned by the sha256 table in
tests/data/README.md. These tests make those checksums load-bearing: a
drift in either a data file or the README fails the suite, so replacing a
subset forces the README (and the reviewed-change process it documents) to
be updated in the same commit.

Only committed files are read, so these tests run everywhere including CI.
"""

import hashlib
import re
from pathlib import Path

import pytest

DATA_DIR = Path(__file__).parent.parent / "data"
README = DATA_DIR / "README.md"

# A checksum table row: | `filename` | `64-hex-digit digest` |
# Tolerant of surrounding whitespace and column padding, strict on the
# digest format itself.
_ROW_PATTERN = re.compile(
    r"^\s*\|\s*`(?P<name>[^`]+)`\s*\|\s*`(?P<digest>[0-9a-f]{64})`\s*\|\s*$"
)
_HEADING_PATTERN = re.compile(r"^##\s+(?P<section>\S+)")


def checksum_table():
    """Parse (relative path, digest) rows from the README checksum tables.

    Filenames repeat across datasets (both LAMMPS examples ship a
    ``dump.nh.lammps``), so each row is resolved against the nearest
    preceding ``##`` section heading, which names the dataset directory.
    """
    rows = []
    section = None
    for line in README.read_text().splitlines():
        heading = _HEADING_PATTERN.match(line)
        if heading:
            section = heading.group("section")
            continue
        row = _ROW_PATTERN.match(line)
        if row:
            assert section is not None, (
                f"Checksum row before any section heading: {line!r}"
            )
            rows.append((Path(section) / row.group("name"),
                         row.group("digest")))
    return rows


@pytest.mark.integration
class TestCommittedDataChecksums:
    """The sha256 table in tests/data/README.md matches the files on disk."""

    def test_readme_lists_checksums(self):
        """The README parses to a non-empty checksum table."""
        rows = checksum_table()
        assert rows, f"No checksum rows parsed from {README}"
        # One row per committed data file: nothing listed but missing, and
        # nothing committed but unlisted.
        listed = {rel for rel, _ in rows}
        # Hidden files are runtime caches (e.g. MDAnalysis .trr_offsets),
        # gitignored rather than committed, so they are not listed.
        on_disk = {
            path.relative_to(DATA_DIR)
            for path in DATA_DIR.rglob("*")
            if path.is_file()
            and path.name != "README.md"
            and not path.name.startswith(".")
        }
        assert listed == on_disk, (
            f"README table and tests/data/ disagree: "
            f"listed-only={sorted(map(str, listed - on_disk))}, "
            f"unlisted={sorted(map(str, on_disk - listed))}"
        )

    def test_checksums_match(self):
        """Every listed file's sha256 equals the digest recorded for it."""
        mismatches = []
        for rel_path, expected in checksum_table():
            actual = hashlib.sha256(
                (DATA_DIR / rel_path).read_bytes()
            ).hexdigest()
            if actual != expected:
                mismatches.append(f"{rel_path}: README {expected}, file {actual}")
        assert not mismatches, (
            "Committed data files do not match the README checksum table "
            "(a subset replacement must update tests/data/README.md in the "
            "same reviewed change):\n" + "\n".join(mismatches)
        )
