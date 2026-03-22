"""Tests for the standalone cube file writer."""

from __future__ import annotations

import numpy as np
import pytest

from revelsMD.density.writers.cube import ANGSTROM_TO_BOHR, write_cube


def _parse_cube_header(path):
    """Parse a cube file header, returning (natoms, origin, voxels, data_lines)."""
    with open(path) as f:
        lines = f.readlines()

    comment1 = lines[0]
    comment2 = lines[1]

    parts = lines[2].split()
    natoms = int(parts[0])
    origin = [float(x) for x in parts[1:4]]

    voxels = []
    for i in range(3):
        parts = lines[3 + i].split()
        n = int(parts[0])
        vec = [float(x) for x in parts[1:4]]
        voxels.append((n, vec))

    # 2 comment lines + natoms line + 3 voxel lines + atom lines
    data_start = 6 + natoms
    data_text = "".join(lines[data_start:])
    data_values = [float(x) for x in data_text.split()]

    return {
        "comment1": comment1,
        "comment2": comment2,
        "natoms": natoms,
        "origin": origin,
        "voxels": voxels,
        "data_values": data_values,
    }


class TestWriteCube:
    """Tests for write_cube."""

    def test_creates_non_empty_file(self, tmp_path):
        grid = np.ones((3, 4, 5))
        cell = np.diag([10.0, 10.0, 10.0])
        path = tmp_path / "test.cube"

        write_cube(path, grid, cell)

        assert path.exists()
        assert path.stat().st_size > 0

    def test_header_has_zero_atoms(self, tmp_path):
        grid = np.ones((3, 4, 5))
        cell = np.diag([10.0, 10.0, 10.0])
        path = tmp_path / "test.cube"

        write_cube(path, grid, cell)
        header = _parse_cube_header(path)

        assert header["natoms"] == 0
        assert header["origin"] == [0.0, 0.0, 0.0]

    def test_voxel_vectors_correct(self, tmp_path):
        grid = np.ones((4, 6, 8))
        cell = np.array([
            [12.0, 0.0, 0.0],
            [0.0, 18.0, 0.0],
            [0.0, 0.0, 24.0],
        ])
        path = tmp_path / "test.cube"

        write_cube(path, grid, cell)
        header = _parse_cube_header(path)

        for i, (n, vec) in enumerate(header["voxels"]):
            expected_n = grid.shape[i]
            expected_vec = cell[i] / expected_n * ANGSTROM_TO_BOHR
            assert n == expected_n
            np.testing.assert_allclose(vec, expected_vec, atol=1e-5)

    def test_data_section_has_correct_count(self, tmp_path):
        grid = np.ones((3, 4, 5))
        cell = np.diag([10.0, 10.0, 10.0])
        path = tmp_path / "test.cube"

        write_cube(path, grid, cell)
        header = _parse_cube_header(path)

        assert len(header["data_values"]) == 3 * 4 * 5

    def test_data_values_roundtrip(self, tmp_path):
        rng = np.random.default_rng(42)
        grid = rng.standard_normal((3, 4, 5))
        cell = np.diag([10.0, 10.0, 10.0])
        path = tmp_path / "test.cube"

        write_cube(path, grid, cell)
        header = _parse_cube_header(path)

        recovered = np.array(header["data_values"]).reshape(grid.shape)
        np.testing.assert_allclose(recovered, grid, rtol=1e-5)

    def test_data_ordering_c_contiguous(self, tmp_path):
        """Verify data is written in C (row-major) order by checking specific positions."""
        grid = np.zeros((2, 3, 4))
        grid[0, 0, 0] = 1.0
        grid[0, 0, 1] = 2.0
        grid[1, 0, 0] = 3.0
        cell = np.diag([10.0, 10.0, 10.0])
        path = tmp_path / "test.cube"

        write_cube(path, grid, cell)
        header = _parse_cube_header(path)
        values = header["data_values"]

        # C order: [0,0,0]=1st, [0,0,1]=2nd, [1,0,0] at index 3*4=12th
        assert values[0] == pytest.approx(1.0)
        assert values[1] == pytest.approx(2.0)
        assert values[12] == pytest.approx(3.0)

    def test_fortran_order_array_written_correctly(self, tmp_path):
        """Fortran-order input produces the same output as C-order."""
        rng = np.random.default_rng(99)
        grid_c = rng.standard_normal((3, 4, 5))
        grid_f = np.asfortranarray(grid_c)
        cell = np.diag([10.0, 10.0, 10.0])

        path_c = tmp_path / "c_order.cube"
        path_f = tmp_path / "f_order.cube"
        write_cube(path_c, grid_c, cell)
        write_cube(path_f, grid_f, cell)

        header_c = _parse_cube_header(path_c)
        header_f = _parse_cube_header(path_f)
        np.testing.assert_allclose(header_f["data_values"], header_c["data_values"])

    def test_triclinic_cell(self, tmp_path):
        grid = np.ones((3, 4, 5))
        cell = np.array([
            [10.0, 0.0, 0.0],
            [2.0, 9.0, 0.0],
            [1.0, 1.0, 8.0],
        ])
        path = tmp_path / "test.cube"

        write_cube(path, grid, cell)
        header = _parse_cube_header(path)

        for i, (n, vec) in enumerate(header["voxels"]):
            expected_vec = cell[i] / n * ANGSTROM_TO_BOHR
            np.testing.assert_allclose(vec, expected_vec, atol=1e-5)

    def test_invalid_path_raises(self, tmp_path):
        grid = np.ones((3, 4, 5))
        cell = np.diag([10.0, 10.0, 10.0])

        with pytest.raises(FileNotFoundError):
            write_cube("/nonexistent/path/test.cube", grid, cell)

    def test_custom_comment(self, tmp_path):
        grid = np.ones((2, 2, 2))
        cell = np.diag([5.0, 5.0, 5.0])
        path = tmp_path / "test.cube"

        write_cube(path, grid, cell, comment="test comment")
        header = _parse_cube_header(path)

        assert "test comment" in header["comment1"]

    def test_2d_grid_raises(self, tmp_path):
        grid = np.ones((3, 4))
        cell = np.diag([10.0, 10.0, 10.0])

        with pytest.raises(ValueError, match="3D array"):
            write_cube(tmp_path / "test.cube", grid, cell)

    def test_wrong_cell_matrix_shape_raises(self, tmp_path):
        grid = np.ones((3, 4, 5))
        cell = np.diag([10.0, 10.0])  # 2x2

        with pytest.raises(ValueError, match=r"\(3, 3\)"):
            write_cube(tmp_path / "test.cube", grid, cell)
