"""Write small LAMMPS dumps for tests.

The atoms are those of tests/data/lammps_small: ids {3, 7, 12, 20, 21, 40},
type 1 for ids 3, 12, 21 and type 2 for the rest. In a frame written with
shift s, atom i is at (i + s, i + 0.5 + s, i + 0.25 + s) with force
(-i, timestep, i), and rows are written in the order 20, 3, 40, 12, 21, 7.
"""

import numpy as np

IDS = (20, 3, 40, 12, 21, 7)
IDS_SORTED = np.array(sorted(IDS))
ORTHO_BOUNDS = "0.0 10.0\n0.0 10.0\n0.0 10.0\n"
COLUMNS = "id type x y z fx fy fz"


def atom_type(i: int) -> int:
    return 1 if i in (3, 12, 21) else 2


def rows(timestep: int, shift: float = 0.0) -> list[str]:
    return [
        f"{i} {atom_type(i)} {i + shift!r} {i + 0.5 + shift!r} {i + 0.25 + shift!r} "
        f"{-i} {timestep} {i}"
        for i in IDS
    ]


def frame(
    timestep: int,
    shift: float = 0.0,
    *,
    box_item: str = "pp pp pp",
    bounds: str = ORTHO_BOUNDS,
    columns: str = COLUMNS,
    atom_rows: list[str] | None = None,
    n_atoms: int = 6,
) -> str:
    body = rows(timestep, shift) if atom_rows is None else atom_rows
    return (
        f"ITEM: TIMESTEP\n{timestep}\nITEM: NUMBER OF ATOMS\n{n_atoms}\n"
        f"ITEM: BOX BOUNDS {box_item}\n{bounds}ITEM: ATOMS {columns}\n"
        + "".join(row + "\n" for row in body)
    )


def positions(shift: float = 0.0) -> np.ndarray:
    """Expected positions of a frame, rows in ascending id order."""
    return np.column_stack([IDS_SORTED, IDS_SORTED + 0.5, IDS_SORTED + 0.25]) + shift


def forces(timestep: int) -> np.ndarray:
    """Expected forces of a frame, rows in ascending id order."""
    return np.column_stack([-IDS_SORTED, np.full(6, timestep), IDS_SORTED]).astype(float)


def write(path, *frames: str) -> str:
    path.write_text("".join(frames))
    return str(path)
