"""Variants of tests/data/example_3_vasp/vasprun.xml made by byte edits."""

import xml.etree.ElementTree as ET
from pathlib import Path

from lxml import etree
import numpy as np

SOURCE = Path(__file__).parent / "data" / "example_3_vasp" / "vasprun.xml"
OPEN, CLOSE = b"<calculation>", b"</calculation>"
FORCES = b'<varray name="forces"'
BASIS = b'<varray name="basis"'


def split(text: bytes) -> tuple[bytes, list[bytes], bytes]:
    """Return (text before the first calculation, the calculations, text after the last)."""
    first = text.find(OPEN)
    calculations = []
    position = first
    end = first
    while position != -1:
        end = text.find(CLOSE, position) + len(CLOSE)
        calculations.append(text[position:end])
        position = text.find(OPEN, end)
    return text[:first], calculations, text[end:]


def join(head: bytes, calculations: list[bytes], tail: bytes) -> bytes:
    return head + b"\n ".join(calculations) + tail


def without_forces(calculation: bytes) -> bytes:
    start = calculation.find(FORCES)
    end = calculation.find(b"</varray>", start) + len(b"</varray>")
    return calculation[:start] + calculation[end:]


def without_structure(calculation: bytes) -> bytes:
    """Remove the calculation's <structure> element."""
    start = calculation.find(b"<structure>")
    end = calculation.find(b"</structure>", start) + len(b"</structure>")
    return calculation[:start] + calculation[end:]


def without_basis(calculation: bytes) -> bytes:
    """Remove the lattice basis varray from the calculation's structure."""
    start = calculation.find(BASIS, calculation.find(b"<structure>"))
    end = calculation.find(b"</varray>", start) + len(b"</varray>")
    return calculation[:start] + calculation[end:]


def with_basis(calculation: bytes, transform) -> bytes:
    """Replace the structure's basis with transform(basis)."""
    start = calculation.find(BASIS, calculation.find(b"<structure>"))
    end = calculation.find(b"</varray>", start) + len(b"</varray>")
    element = etree.fromstring(calculation[start:end])
    basis = np.array([v.text.split() for v in element.findall("v")], dtype=float)
    rows = "".join(f"     <v> {float(a)!r} {float(b)!r} {float(c)!r} </v>\n" for a, b, c in transform(basis))
    replacement = f'<varray name="basis" >\n{rows}    </varray>'.encode()
    return calculation[:start] + replacement + calculation[end:]


def write(path: Path, data: bytes) -> str:
    path.write_bytes(data)
    return str(path)


def truncated(path: Path) -> str:
    """Cut off partway through the last calculation, after its structure."""
    text = SOURCE.read_bytes()
    last = text.rfind(OPEN)
    cut = text.find(b"</structure>", last) + len(b"</structure>") + 20
    return write(path, text[:cut])


def _floats(elements):
    """Rows of numbers from <v> elements, converted with float()."""
    return np.array([[float(x) for x in v.text.split()] for v in elements])


def independent_frames(path=SOURCE):
    """Positions (fractional @ basis) and forces of each ionic step, read with ElementTree.

    This parser shares no code with the loader, so the loader is compared
    with an independent reading of the file.
    """
    frames = []
    for calculation in ET.parse(path).getroot().findall("calculation"):
        structure = calculation.find("structure")
        basis = _floats(structure.find("crystal").findall("varray[@name='basis']/v"))
        fractional = _floats(structure.findall("varray[@name='positions']/v"))
        forces = _floats(calculation.findall("varray[@name='forces']/v"))
        frames.append((fractional @ basis, forces))
    return frames

