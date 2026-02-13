"""Tests for revelsMD.density package and its module structure."""

import warnings

import numpy as np
import pytest
from ase import Atoms

from revelsMD.density import DensityGrid, Selection


# ---------------------------------------------------------------------------
# DensityGrid Initialisation
# ---------------------------------------------------------------------------

def test_densitygrid_initialisation(ts):
    gs = DensityGrid(ts, density_type="number", nbins=4)
    assert gs.nbinsx == 4
    assert gs.lx == pytest.approx(ts.box_x / 4)
    assert gs.voxel_volume > 0
    assert np.all(gs.force_x == 0)
    assert gs.count == 0  # No data accumulated yet


def test_densitygrid_uses_trajectory_beta(ts):
    """DensityGrid should use beta from the trajectory object."""
    gs = DensityGrid(ts, density_type="number", nbins=4)
    assert gs.beta == ts.beta


def test_densitygrid_invalid_box(ts):
    ts.box_x = -10.0
    with pytest.raises(ValueError):
        DensityGrid(ts, "number")


def test_densitygrid_invalid_bins(ts):
    with pytest.raises(ValueError):
        DensityGrid(ts, "number", nbins=(0, 4, 4))


def test_densitygrid_stores_cell_matrix(ts):
    """DensityGrid should store cell_matrix from the trajectory."""
    gs = DensityGrid(ts, density_type="number", nbins=4)
    expected = np.diag([10.0, 10.0, 10.0])
    np.testing.assert_allclose(gs.cell_matrix, expected)


def test_densitygrid_stores_cell_inverse(ts):
    """DensityGrid should store the inverse cell matrix."""
    gs = DensityGrid(ts, density_type="number", nbins=4)
    expected_inv = np.linalg.inv(np.diag([10.0, 10.0, 10.0]))
    np.testing.assert_allclose(gs.cell_inverse, expected_inv)


def test_densitygrid_is_orthorhombic(ts):
    """DensityGrid should flag the cell as orthorhombic."""
    gs = DensityGrid(ts, density_type="number", nbins=4)
    assert gs.is_orthorhombic is True


def test_densitygrid_voxel_volume_from_cell(ts):
    """Voxel volume should equal det(cell_matrix) / (nbins^3)."""
    gs = DensityGrid(ts, density_type="number", nbins=4)
    expected = abs(np.linalg.det(np.diag([10.0, 10.0, 10.0]))) / (4 * 4 * 4)
    assert gs.voxel_volume == pytest.approx(expected)


def test_densitygrid_orthorhombic_regression(ts):
    """Orthorhombic DensityGrid should produce identical bin edges and voxel sizes."""
    gs = DensityGrid(ts, density_type="number", nbins=4)
    # Bin edges should be Cartesian
    np.testing.assert_allclose(gs.binsx, np.arange(0, 10.0 + 2.5, 2.5))
    np.testing.assert_allclose(gs.binsy, np.arange(0, 10.0 + 2.5, 2.5))
    np.testing.assert_allclose(gs.binsz, np.arange(0, 10.0 + 2.5, 2.5))
    # Voxel sizes should be Cartesian
    assert gs.lx == pytest.approx(2.5)
    assert gs.ly == pytest.approx(2.5)
    assert gs.lz == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# k-vectors and FFT utilities
# ---------------------------------------------------------------------------

def test_kvectors_ksquared_shapes(ts):
    gs = DensityGrid(ts, "number", nbins=4)
    kx, ky, kz = gs.get_kvectors()
    assert kx.shape[0] == gs.nbinsx
    ks = gs.get_ksquared()
    assert ks.shape == (gs.nbinsx, gs.nbinsy, gs.nbinsz)
    assert np.all(ks >= 0)


def test_build_kvectors_3d_shape():
    """_build_kvectors_3d should return (nbins, nbins, nbins, 3) k-vectors
    and (nbins, nbins, nbins) ksquared."""
    from revelsMD.trajectories.numpy import NumpyTrajectory

    cell = np.diag([10.0, 8.0, 6.0])
    traj = NumpyTrajectory(
        positions=np.zeros((2, 3, 3)),
        forces=np.zeros((2, 3, 3)),
        cell_matrix=cell,
        species_list=["A", "A", "A"],
        temperature=300.0, units="real",
    )
    gs = DensityGrid(traj, density_type="number", nbins=4)
    k_vectors, ksquared = gs._build_kvectors_3d()
    assert k_vectors.shape == (4, 4, 4, 3)
    assert ksquared.shape == (4, 4, 4)
    # ksquared should equal the sum of squares of k components
    np.testing.assert_allclose(ksquared, np.sum(k_vectors ** 2, axis=-1))


def test_build_kvectors_3d_orthorhombic_equivalence():
    """For an orthorhombic cell, _build_kvectors_3d should give k-vectors
    equivalent to the existing per-axis get_kvectors method."""
    from revelsMD.trajectories.numpy import NumpyTrajectory

    cell = np.diag([10.0, 8.0, 6.0])
    traj = NumpyTrajectory(
        positions=np.zeros((2, 3, 3)),
        forces=np.zeros((2, 3, 3)),
        cell_matrix=cell,
        species_list=["A", "A", "A"],
        temperature=300.0, units="real",
    )
    gs = DensityGrid(traj, density_type="number", nbins=4)

    # Get the existing 1D k-vectors
    kx_1d, ky_1d, kz_1d = gs.get_kvectors()

    # Build full 3D k-vectors
    k_vectors, ksquared = gs._build_kvectors_3d()

    # For orthorhombic cells, k_x[i,j,k] should equal kx_1d[i] etc.
    for i in range(4):
        for j in range(4):
            for k in range(4):
                np.testing.assert_allclose(
                    k_vectors[i, j, k],
                    [kx_1d[i], ky_1d[j], kz_1d[k]],
                    atol=1e-12,
                )


def test_build_kvectors_3d_triclinic():
    """For a triclinic cell, verify k-vectors match 2*pi * inv(M)^T @ m."""
    from revelsMD.trajectories.numpy import NumpyTrajectory

    cell = np.array([
        [10.0, 0.0, 0.0],
        [3.0, 9.0, 0.0],
        [0.0, 0.0, 8.0],
    ])
    nbins = 4
    traj = NumpyTrajectory(
        positions=np.zeros((2, 3, 3)),
        forces=np.zeros((2, 3, 3)),
        cell_matrix=cell,
        species_list=["A", "A", "A"],
        temperature=300.0, units="real",
    )
    gs = DensityGrid(traj, density_type="number", nbins=nbins)

    k_vectors, _ = gs._build_kvectors_3d()

    # Expected: k = 2*pi * inv(M)^T @ [m1, m2, m3]^T
    M_inv_T = np.linalg.inv(cell).T
    miller = np.fft.fftfreq(nbins, d=1.0 / nbins)
    for i, m1 in enumerate(miller):
        for j, m2 in enumerate(miller):
            for k_idx, m3 in enumerate(miller):
                expected = 2 * np.pi * M_inv_T @ np.array([m1, m2, m3])
                np.testing.assert_allclose(
                    k_vectors[i, j, k_idx], expected, atol=1e-12,
                )


# ---------------------------------------------------------------------------
# DensityGrid._process_frame: Box & Triangular kernels
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kernel", ["box", "triangular"])
def test_process_frame_kernels(ts, kernel):
    """_process_frame deposits positions/forces to grid using specified kernel."""
    gs = DensityGrid(ts, "number", nbins=4)
    pos = np.array([[1.0, 2.0, 3.0]])
    frc = np.array([[0.5, 0.0, 0.0]])
    gs._process_frame(pos, frc, weight=1.0, kernel=kernel)
    assert np.any(gs.force_x != 0)
    assert np.any(gs.counter != 0)


def test_process_frame_increments_count(ts):
    """_process_frame increments the frame count."""
    gs = DensityGrid(ts, "number", nbins=4)
    assert gs.count == 0
    gs._process_frame(np.array([[1.0, 2.0, 3.0]]), np.array([[0.5, 0.0, 0.0]]))
    assert gs.count == 1
    gs._process_frame(np.array([[2.0, 3.0, 4.0]]), np.array([[0.0, 0.5, 0.0]]))
    assert gs.count == 2


def test_process_frame_invalid_kernel(ts):
    """_process_frame raises ValueError for unknown kernel."""
    gs = DensityGrid(ts, "number", nbins=4)
    with pytest.raises(ValueError, match="Unsupported kernel"):
        gs._process_frame(np.array([[1.0, 2.0, 3.0]]), np.array([[0.5, 0.0, 0.0]]), kernel="invalid")


@pytest.mark.parametrize("kernel", ["box", "triangular"])
def test_process_frame_triclinic_deposits(kernel):
    """_process_frame should deposit to grid for triclinic cells."""
    from revelsMD.trajectories.numpy import NumpyTrajectory

    cell = np.array([
        [10.0, 0.0, 0.0],
        [3.0, 9.0, 0.0],
        [0.0, 0.0, 8.0],
    ])
    traj = NumpyTrajectory(
        positions=np.zeros((2, 3, 3)),
        forces=np.zeros((2, 3, 3)),
        cell_matrix=cell,
        species_list=["A", "A", "A"],
        temperature=300.0, units="real",
    )
    gs = DensityGrid(traj, density_type="number", nbins=4)
    # Position at (5, 4.5, 4) should be inside the cell
    pos = np.array([[5.0, 4.5, 4.0]])
    frc = np.array([[0.5, 0.0, 0.0]])
    gs._process_frame(pos, frc, weight=1.0, kernel=kernel)
    assert np.any(gs.force_x != 0)
    assert np.any(gs.counter != 0)


def test_process_frame_triclinic_boundary_particles():
    """Particles at fractional coordinate boundaries should not crash."""
    from revelsMD.trajectories.numpy import NumpyTrajectory

    cell = np.array([
        [10.0, 0.0, 0.0],
        [3.0, 9.0, 0.0],
        [0.0, 0.0, 8.0],
    ])
    traj = NumpyTrajectory(
        positions=np.zeros((2, 3, 3)),
        forces=np.zeros((2, 3, 3)),
        cell_matrix=cell,
        species_list=["A", "A", "A"],
        temperature=300.0, units="real",
    )
    gs = DensityGrid(traj, density_type="number", nbins=4)
    # Origin and near-edge positions in Cartesian
    pos = np.array([[0.0, 0.0, 0.0], [9.99, 8.99, 7.99]])
    frc = np.array([[0.1, 0.0, 0.0], [0.0, 0.1, 0.0]])
    gs._process_frame(pos, frc, weight=1.0, kernel="triangular")
    # Should not crash and should deposit something
    assert np.any(gs.counter != 0)


# ---------------------------------------------------------------------------
# DensityGrid.deposit (basic tests)
# ---------------------------------------------------------------------------

def test_deposit_single_array(ts):
    """deposit with single array deposits once."""
    gs = DensityGrid(ts, "number", nbins=4)
    pos = np.array([[1.0, 2.0, 3.0]])
    frc = np.array([[0.5, 0.0, 0.0]])
    gs.deposit(pos, frc, weights=1.0, kernel="triangular")

    assert gs.count == 1
    assert np.any(gs.counter != 0)
    assert np.any(gs.force_x != 0)


def test_deposit_list_of_arrays(ts):
    """deposit with list of arrays deposits each separately."""
    gs = DensityGrid(ts, "number", nbins=4)
    pos_list = [np.array([[1.0, 2.0, 3.0]]), np.array([[4.0, 5.0, 6.0]])]
    frc_list = [np.array([[0.5, 0.0, 0.0]]), np.array([[0.0, 0.5, 0.0]])]
    gs.deposit(pos_list, frc_list, weights=1.0, kernel="box")

    assert gs.count == 2
    assert np.any(gs.counter != 0)


def test_deposit_broadcasts_scalar_weight(ts):
    """deposit broadcasts scalar weight to all position arrays."""
    gs = DensityGrid(ts, "number", nbins=4)
    pos_list = [np.array([[1.0, 2.0, 3.0]]), np.array([[4.0, 5.0, 6.0]])]
    frc_list = [np.array([[0.5, 0.0, 0.0]]), np.array([[0.0, 0.5, 0.0]])]
    gs.deposit(pos_list, frc_list, weights=1.0, kernel="triangular")

    # Both depositions should have been made
    assert gs.count == 2
    assert np.any(gs.counter != 0)


def test_deposit_rejects_list_weights_with_single_positions(ts):
    """deposit raises TypeError if weights is a list but positions is a single array."""
    gs = DensityGrid(ts, "number", nbins=4)
    pos = np.array([[1.0, 2.0, 3.0]])
    frc = np.array([[0.5, 0.0, 0.0]])
    weights_list = [np.array([1.0])]

    with pytest.raises(TypeError, match="weights cannot be a list"):
        gs.deposit(pos, frc, weights=weights_list)


def test_deposit_rejects_list_forces_with_single_positions(ts):
    """deposit raises TypeError if forces is a list but positions is a single array."""
    gs = DensityGrid(ts, "number", nbins=4)
    pos = np.array([[1.0, 2.0, 3.0]])
    frc_list = [np.array([[0.5, 0.0, 0.0]])]

    with pytest.raises(TypeError, match="positions and forces must both be lists or both be arrays"):
        gs.deposit(pos, frc_list, weights=1.0)


# ---------------------------------------------------------------------------
# Selection (basic tests)
# ---------------------------------------------------------------------------

def test_selection_single_species(ts):
    ss = Selection(ts, "H", centre_location=True)
    assert ss.single_species
    assert isinstance(ss.indices, np.ndarray)


def test_selection_single_species_with_charges(ts):
    ss = Selection(ts, "H", centre_location=True, density_type='charge')
    assert np.all(ss.charges == 0.1)


def test_selection_single_species_with_polarisation(ts):
    ss = Selection(ts, "H", centre_location=True, density_type='polarisation')
    assert np.all(ss.charges == 0.1)
    assert np.all(ss.masses == 1.0)


def test_selection_rigid_molecule(ts):
    ss = Selection(ts, ["H", "O"], centre_location=True)
    assert not ss.single_species
    assert isinstance(ss.indices, list)
    assert len(ss.indices) == 2


def test_selection_rigid_with_polarisation(ts):
    ss = Selection(ts, ["H", "O"], centre_location=True, density_type='polarisation')
    assert len(ss.masses) == 2
    assert len(ss.charges) == 2


def test_selection_invalid_centre_location(ts):
    with pytest.raises(ValueError):
        Selection(ts, ["H", "O"], centre_location="invalid")


def test_selection_rigid_water():
    """Rigid molecules require unique labels for each atom in the molecule.

    Since topology data is not used, atoms are identified by unique labels
    (e.g. Ow, Hw1, Hw2 for water) rather than element symbols.
    """
    class WaterTSMock:
        box_x = box_y = box_z = 10.0
        cell_matrix = np.diag([10.0, 10.0, 10.0])
        units = "real"
        species = ["Ow", "Hw1", "Hw2"]
        # 2 water molecules: Ow, Hw1, Hw2 each = 2 atoms per species
        _ids = {"Ow": np.array([0, 3]), "Hw1": np.array([1, 4]), "Hw2": np.array([2, 5])}
        _charges = {"Ow": np.array([-0.8, -0.8]), "Hw1": np.array([0.4, 0.4]), "Hw2": np.array([0.4, 0.4])}
        _masses = {"Ow": np.array([16.0, 16.0]), "Hw1": np.array([1.0, 1.0]), "Hw2": np.array([1.0, 1.0])}

        def get_indices(self, atype):
            return self._ids[atype]

        def get_charges(self, atype):
            return self._charges[atype]

        def get_masses(self, atype):
            return self._masses[atype]

    ts_water = WaterTSMock()
    ss = Selection(ts_water, ["Ow", "Hw1", "Hw2"], centre_location=True, rigid=True)
    assert len(ss.indices) == 3
    assert len(ss.indices[0]) == 2  # 2 Ow atoms
    assert len(ss.indices[1]) == 2  # 2 Hw1 atoms
    assert len(ss.indices[2]) == 2  # 2 Hw2 atoms


# ---------------------------------------------------------------------------
# Full pipeline tests
# ---------------------------------------------------------------------------

def test_full_number_density_pipeline(tmp_path, ts):
    gs = DensityGrid(ts, "number", nbins=4)
    gs.accumulate(ts, atom_names="H", rigid=False)
    assert gs.count > 0  # Data has been accumulated

    gs.get_real_density()
    assert hasattr(gs, "rho_force")
    assert gs.rho_force.shape == (gs.nbinsx, gs.nbinsy, gs.nbinsz)

    cube_file = tmp_path / "density.cube"
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1]])
    gs.write_to_cube(atoms, gs.rho_force, cube_file)
    assert cube_file.exists()


def test_get_lambda_basic(ts):
    """Test basic get_lambda functionality."""
    gs = DensityGrid(ts, "number", nbins=4)
    gs.accumulate(ts, atom_names="H", rigid=False)
    # get_lambda() re-accumulates internally, so no need to call get_real_density()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        gs.get_lambda(ts, sections=2)
    assert gs.rho_lambda is not None
    assert gs.rho_lambda.shape == gs.rho_force.shape


def test_get_lambda_emits_deprecation_warning(ts):
    """get_lambda should emit a DeprecationWarning."""
    gs = DensityGrid(ts, "number", nbins=4)
    gs.accumulate(ts, atom_names="H", rigid=False)
    # get_lambda() re-accumulates internally, so no need to call get_real_density()

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        gs.get_lambda(ts, sections=2)

    # Check that at least one DeprecationWarning with expected message was emitted
    dep_warnings = [warn for warn in w if issubclass(warn.category, DeprecationWarning)]
    assert dep_warnings, "Expected at least one DeprecationWarning from get_lambda"
    assert any("compute_lambda=True" in str(warn.message) for warn in dep_warnings)


# ---------------------------------------------------------------------------
# compute_lambda parameter tests
# ---------------------------------------------------------------------------


class TestAccumulateComputeLambda:
    """Tests for accumulate() with compute_lambda parameter."""

    @pytest.fixture
    def multi_frame_trajectory(self):
        """Create a trajectory with enough frames for sectioned lambda estimation."""
        class MultiFrameTrajectory:
            def __init__(self):
                self.box_x = self.box_y = self.box_z = 10.0
                self.cell_matrix = np.diag([10.0, 10.0, 10.0])
                self.units = 'real'
                self.temperature = 300.0
                from revelsMD.trajectories._base import compute_beta
                self.beta = compute_beta(self.units, self.temperature)
                self.frames = 10

                # 3 atoms per frame, 10 frames
                np.random.seed(42)
                self._positions = [
                    np.random.rand(3, 3) * 10 for _ in range(self.frames)
                ]
                self._forces = [
                    np.random.randn(3, 3) * 0.1 for _ in range(self.frames)
                ]

                self._ids = {"H": np.array([0, 1, 2])}
                self._charges = {"H": np.array([0.1, 0.1, 0.1])}
                self._masses = {"H": np.array([1.0, 1.0, 1.0])}

            def get_indices(self, atype):
                return self._ids[atype]

            def get_charges(self, atype):
                return self._charges[atype]

            def get_masses(self, atype):
                return self._masses[atype]

            def iter_frames(self, start=0, stop=None, stride=1):
                if stop is None:
                    stop = self.frames
                for i in range(start, stop, stride):
                    yield self._positions[i], self._forces[i]

            def get_frame(self, index):
                return self._positions[index], self._forces[index]

        return MultiFrameTrajectory()

    def test_accumulate_without_compute_lambda_no_welford(self, multi_frame_trajectory):
        """accumulate() without compute_lambda does not create Welford accumulator."""
        gs = DensityGrid(multi_frame_trajectory, "number", nbins=4)
        gs.accumulate(multi_frame_trajectory, atom_names="H")

        assert gs._welford is None
        assert gs.rho_lambda is None

    def test_accumulate_with_compute_lambda_creates_welford(self, multi_frame_trajectory):
        """accumulate() with compute_lambda=True creates Welford accumulator."""
        gs = DensityGrid(multi_frame_trajectory, "number", nbins=4)
        gs.accumulate(multi_frame_trajectory, atom_names="H", compute_lambda=True)

        assert gs._welford is not None
        assert gs._welford.has_data

    def test_accumulate_compute_lambda_default_sections(self, multi_frame_trajectory):
        """accumulate() with compute_lambda=True defaults to one section per frame."""
        gs = DensityGrid(multi_frame_trajectory, "number", nbins=4)
        gs.accumulate(multi_frame_trajectory, atom_names="H", compute_lambda=True)

        # Default is one section per frame (10 frames = 10 sections)
        assert gs._welford.count == 10

    def test_accumulate_compute_lambda_custom_sections(self, multi_frame_trajectory):
        """accumulate() with compute_lambda=True accepts custom sections."""
        gs = DensityGrid(multi_frame_trajectory, "number", nbins=4)
        gs.accumulate(
            multi_frame_trajectory, atom_names="H",
            compute_lambda=True, sections=5
        )

        assert gs._welford.count == 5

    def test_rho_lambda_available_after_compute_lambda(self, multi_frame_trajectory):
        """rho_lambda is available after accumulate with compute_lambda=True."""
        gs = DensityGrid(multi_frame_trajectory, "number", nbins=4)
        gs.accumulate(multi_frame_trajectory, atom_names="H", compute_lambda=True)

        # Access triggers lazy finalisation
        assert gs.rho_lambda is not None
        assert gs.rho_lambda.shape == (4, 4, 4)

    def test_lambda_weights_available_after_compute_lambda(self, multi_frame_trajectory):
        """lambda_weights is available after accumulate with compute_lambda=True."""
        gs = DensityGrid(multi_frame_trajectory, "number", nbins=4)
        gs.accumulate(multi_frame_trajectory, atom_names="H", compute_lambda=True)

        assert gs.lambda_weights is not None
        assert gs.lambda_weights.shape == (4, 4, 4)

    def test_rho_force_still_available(self, multi_frame_trajectory):
        """rho_force is still available after compute_lambda accumulation."""
        gs = DensityGrid(multi_frame_trajectory, "number", nbins=4)
        gs.accumulate(multi_frame_trajectory, atom_names="H", compute_lambda=True)

        # Access rho_lambda to trigger finalisation (which also computes rho_force)
        _ = gs.rho_lambda

        assert gs.rho_force is not None
        # Should have non-trivial values (after finalisation)
        assert np.any(gs.rho_force != 0)

    def test_multiple_accumulate_calls_update_welford(self, multi_frame_trajectory):
        """Multiple accumulate() calls with compute_lambda continue building stats."""
        gs = DensityGrid(multi_frame_trajectory, "number", nbins=4)

        # First accumulation
        gs.accumulate(multi_frame_trajectory, atom_names="H",
                     compute_lambda=True, sections=3, start=0, stop=5)
        first_count = gs._welford.count

        # Second accumulation
        gs.accumulate(multi_frame_trajectory, atom_names="H",
                     compute_lambda=True, sections=3, start=5, stop=10)
        second_count = gs._welford.count

        # Welford count should increase
        assert second_count > first_count

    def test_rho_lambda_returns_none_without_compute_lambda(self, multi_frame_trajectory):
        """rho_lambda returns None when compute_lambda was not used."""
        gs = DensityGrid(multi_frame_trajectory, "number", nbins=4)
        gs.accumulate(multi_frame_trajectory, atom_names="H")

        # Without compute_lambda, rho_lambda should be None
        assert gs.rho_lambda is None
        assert gs.lambda_weights is None

    def test_rho_force_none_before_access(self, multi_frame_trajectory):
        """rho_force is None (not computed) until accessed."""
        gs = DensityGrid(multi_frame_trajectory, "number", nbins=4)
        gs.accumulate(multi_frame_trajectory, atom_names="H", compute_lambda=True)

        # Before accessing rho_force or rho_lambda, internal state is None
        assert gs._rho_force is None

        # After accessing rho_lambda, rho_force should be computed
        _ = gs.rho_lambda
        assert gs._rho_force is not None
        assert np.any(gs._rho_force != 0)

    def test_deprecated_get_lambda_uses_internal_method(
        self, multi_frame_trajectory
    ):
        """Deprecated get_lambda() delegates to _accumulate_with_sections()."""
        gs = DensityGrid(multi_frame_trajectory, "number", nbins=4)
        gs.accumulate(multi_frame_trajectory, atom_names="H")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            gs.get_lambda(multi_frame_trajectory, sections=5)

        # Verify it used the Welford accumulator internally
        assert gs._welford is not None
        assert gs._welford.count == 5
        assert gs._rho_lambda is not None  # Lambda was computed

    def test_multi_trajectory_lambda_accumulation(self, multi_frame_trajectory):
        """Lambda statistics accumulate across multiple accumulate() calls."""
        gs = DensityGrid(multi_frame_trajectory, "number", nbins=4)

        # First accumulation with 5 sections
        gs.accumulate(
            multi_frame_trajectory, atom_names="H",
            compute_lambda=True, sections=5, start=0, stop=5
        )
        assert gs._welford.count == 5

        # Second accumulation adds more sections
        gs.accumulate(
            multi_frame_trajectory, atom_names="H",
            compute_lambda=True, sections=5, start=5, stop=10
        )
        assert gs._welford.count == 10  # Combined from both calls

        # Lambda uses combined statistics
        rho = gs.rho_lambda
        assert rho is not None

    def test_frames_processed_accumulates_across_calls(self, multi_frame_trajectory):
        """frames_processed should accumulate total frames across multiple accumulate() calls."""
        gs = DensityGrid(multi_frame_trajectory, "number", nbins=4)

        # First accumulation: frames 0-4 (5 frames)
        gs.accumulate(
            multi_frame_trajectory, atom_names="H", start=0, stop=5
        )
        assert gs.frames_processed == 5

        # Second accumulation: frames 5-9 (5 frames)
        gs.accumulate(
            multi_frame_trajectory, atom_names="H", start=5, stop=10
        )
        assert gs.frames_processed == 10  # Total from both calls

    def test_compute_lambda_false_clears_welford_with_warning(self, multi_frame_trajectory):
        """accumulate(compute_lambda=False) clears existing Welford state with warning."""
        gs = DensityGrid(multi_frame_trajectory, "number", nbins=4)

        # Accumulate with lambda
        gs.accumulate(
            multi_frame_trajectory, atom_names="H", compute_lambda=True
        )
        assert gs._welford is not None

        # Accumulate without lambda clears the Welford and warns
        with pytest.warns(UserWarning, match="discards existing lambda statistics"):
            gs.accumulate(
                multi_frame_trajectory, atom_names="H", compute_lambda=False
            )
        assert gs._welford is None
        assert gs._rho_lambda is None

    def test_compute_lambda_false_no_warning_when_no_welford(self, multi_frame_trajectory):
        """accumulate(compute_lambda=False) does not warn if no Welford exists."""
        gs = DensityGrid(multi_frame_trajectory, "number", nbins=4)

        # First accumulate without lambda - no warning expected
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # Turn warnings into errors
            gs.accumulate(
                multi_frame_trajectory, atom_names="H", compute_lambda=False
            )
        assert gs._welford is None

    def test_rho_lambda_raises_with_insufficient_sections(self, multi_frame_trajectory):
        """Accessing rho_lambda with < 2 sections raises ValueError."""
        gs = DensityGrid(multi_frame_trajectory, "number", nbins=4)
        gs.accumulate(
            multi_frame_trajectory, atom_names="H",
            compute_lambda=True, sections=1, start=0, stop=1
        )

        with pytest.raises(ValueError, match="fewer than 2 sections"):
            _ = gs.rho_lambda

    def test_rho_lambda_works_with_one_section_per_trajectory(
        self, multi_frame_trajectory
    ):
        """sections=1 works if accumulated across multiple trajectories."""
        gs = DensityGrid(multi_frame_trajectory, "number", nbins=4)

        # First trajectory with sections=1
        gs.accumulate(
            multi_frame_trajectory, atom_names="H",
            compute_lambda=True, sections=1, start=0, stop=5
        )
        assert gs._welford.count == 1

        # Second trajectory with sections=1
        gs.accumulate(
            multi_frame_trajectory, atom_names="H",
            compute_lambda=True, sections=1, start=5, stop=10
        )
        assert gs._welford.count == 2  # Now have 2 total sections

        # Lambda now works
        rho = gs.rho_lambda
        assert rho is not None

    def test_sections_exceeds_frames_raises_error(self, multi_frame_trajectory):
        """Requesting more sections than frames should raise ValueError."""
        gs = DensityGrid(multi_frame_trajectory, "number", nbins=4)

        # multi_frame_trajectory has 10 frames, request 20 sections
        with pytest.raises(ValueError, match="sections.*exceeds.*frames"):
            gs.accumulate(
                multi_frame_trajectory, atom_names="H",
                compute_lambda=True, sections=20
            )


# ---------------------------------------------------------------------------
# Mock trajectory for Selection tests (water molecules)
# ---------------------------------------------------------------------------

class MockTrajectory:
    """Mock trajectory for testing Selection methods."""

    def __init__(self):
        self.box_x = self.box_y = self.box_z = 10.0
        self.cell_matrix = np.diag([10.0, 10.0, 10.0])
        self.units = 'real'
        self.beta = 1.0 / (300.0 * 0.0019872041)  # 1/(kB*T) for T=300K in real units

    def get_indices(self, atom_name):
        # 3 molecules, each with atoms O, H1, H2
        indices = {
            'O': np.array([0, 3, 6]),
            'H1': np.array([1, 4, 7]),
            'H2': np.array([2, 5, 8]),
        }
        return indices[atom_name]

    def get_masses(self, atom_name):
        masses = {
            'O': np.array([16.0, 16.0, 16.0]),
            'H1': np.array([1.0, 1.0, 1.0]),
            'H2': np.array([1.0, 1.0, 1.0]),
        }
        return masses[atom_name]

    def get_charges(self, atom_name):
        charges = {
            'O': np.array([-0.8, -0.8, -0.8]),
            'H1': np.array([0.4, 0.4, 0.4]),
            'H2': np.array([0.4, 0.4, 0.4]),
        }
        return charges[atom_name]


class IterableMockTrajectory(MockTrajectory):
    """MockTrajectory extended with frame iteration support."""

    def __init__(self):
        super().__init__()
        self.frames = 2
        self._positions = [
            np.array([
                [1.0, 5.0, 5.0], [1.5, 5.0, 5.0], [0.5, 5.0, 5.0],
                [4.0, 5.0, 5.0], [4.5, 5.0, 5.0], [3.5, 5.0, 5.0],
                [7.0, 5.0, 5.0], [7.5, 5.0, 5.0], [6.5, 5.0, 5.0],
            ], dtype=float),
            np.array([
                [1.1, 5.1, 5.0], [1.6, 5.1, 5.0], [0.6, 5.1, 5.0],
                [4.1, 5.1, 5.0], [4.6, 5.1, 5.0], [3.6, 5.1, 5.0],
                [7.1, 5.1, 5.0], [7.6, 5.1, 5.0], [6.6, 5.1, 5.0],
            ], dtype=float),
        ]
        self._forces = [
            np.array([
                [1.0, 0.1, 0.0], [0.5, 0.05, 0.0], [0.5, 0.05, 0.0],
                [2.0, 0.2, 0.0], [1.0, 0.1, 0.0], [1.0, 0.1, 0.0],
                [3.0, 0.3, 0.0], [1.5, 0.15, 0.0], [1.5, 0.15, 0.0],
            ], dtype=float),
            np.array([
                [1.1, 0.11, 0.0], [0.55, 0.055, 0.0], [0.55, 0.055, 0.0],
                [2.2, 0.22, 0.0], [1.1, 0.11, 0.0], [1.1, 0.11, 0.0],
                [3.3, 0.33, 0.0], [1.65, 0.165, 0.0], [1.65, 0.165, 0.0],
            ], dtype=float),
        ]

    def iter_frames(self, start, stop, period):
        for i in range(start, stop or self.frames, period):
            yield self._positions[i], self._forces[i]


# ---------------------------------------------------------------------------
# Selection.get_positions() tests
# ---------------------------------------------------------------------------

class TestDeposit:
    """Tests for DensityGrid.deposit with Selection."""

    @pytest.fixture
    def trajectory(self):
        return MockTrajectory()

    @pytest.fixture
    def positions(self):
        """9 atoms: 3 water molecules."""
        return np.array([
            [1.0, 5.0, 5.0],   # O (index 0)
            [1.5, 5.0, 5.0],   # H1 (index 1)
            [0.5, 5.0, 5.0],   # H2 (index 2)
            [4.0, 5.0, 5.0],   # O (index 3)
            [4.5, 5.0, 5.0],   # H1 (index 4)
            [3.5, 5.0, 5.0],   # H2 (index 5)
            [7.0, 5.0, 5.0],   # O (index 6)
            [7.5, 5.0, 5.0],   # H1 (index 7)
            [6.5, 5.0, 5.0],   # H2 (index 8)
        ], dtype=float)

    @pytest.fixture
    def forces(self):
        """Forces for 9 atoms."""
        return np.array([
            [1.0, 0.1, 0.0],   # O
            [0.5, 0.05, 0.0],  # H1
            [0.5, 0.05, 0.0],  # H2
            [2.0, 0.2, 0.0],   # O
            [1.0, 0.1, 0.0],   # H1
            [1.0, 0.1, 0.0],   # H2
            [3.0, 0.3, 0.0],   # O
            [1.5, 0.15, 0.0],  # H1
            [1.5, 0.15, 0.0],  # H2
        ], dtype=float)

    def test_deposit_single_species_number_density(self, trajectory, positions, forces):
        """deposit with single species populates grid."""

        gs = DensityGrid(trajectory, "number", nbins=5)
        ss = Selection(trajectory, 'O', centre_location=True, rigid=False, density_type='number')
        gs.deposit(ss.get_positions(positions), ss.get_forces(forces), ss.get_weights(), kernel="triangular")

        assert np.any(gs.force_x != 0)
        assert np.any(gs.counter != 0)
        assert gs.count == 1

    def test_deposit_multi_species_non_rigid(self, trajectory, positions, forces):
        """deposit with multi-species non-rigid deposits each species."""

        gs = DensityGrid(trajectory, "number", nbins=5)
        ss = Selection(trajectory, ['O', 'H1', 'H2'], centre_location=True, rigid=False, density_type='number')
        gs.deposit(ss.get_positions(positions), ss.get_forces(forces), ss.get_weights(), kernel="triangular")

        # 3 species deposited = 3 calls to _process_frame
        assert gs.count == 3
        assert np.any(gs.counter != 0)

    def test_deposit_rigid_com_number_density(self, trajectory, positions, forces):
        """deposit with rigid molecule at COM populates grid."""

        gs = DensityGrid(trajectory, "number", nbins=5)
        ss = Selection(trajectory, ['O', 'H1', 'H2'], centre_location=True, rigid=True, density_type='number')
        gs.deposit(ss.get_positions(positions), ss.get_forces(forces), ss.get_weights(), kernel="triangular")

        # Rigid deposits once per molecule group
        assert gs.count == 1
        assert np.any(gs.counter != 0)
        assert np.any(gs.force_x != 0)

    def test_deposit_charge_density_single_species(self, trajectory, positions, forces):
        """deposit with charge density uses charge weights."""

        gs = DensityGrid(trajectory, "charge", nbins=5)
        ss = Selection(trajectory, 'O', centre_location=True, rigid=False, density_type='charge')
        gs.deposit(ss.get_positions(positions), ss.get_forces(forces), ss.get_weights(), kernel="triangular")

        # O has negative charge, so counter contributions are negative
        assert np.any(gs.counter != 0)
        assert gs.count == 1

    def test_deposit_list_positions_array_forces_raises(self, trajectory, positions, forces):
        """deposit raises TypeError when positions is list but forces is array."""
        gs = DensityGrid(trajectory, "number", nbins=5)

        # positions as list of arrays, forces as single array
        positions_list = [positions[:3], positions[3:6], positions[6:]]
        with pytest.raises(TypeError, match="positions and forces must both be lists or both be arrays"):
            gs.deposit(positions_list, forces, weights=1.0)


def test_accumulate_mismatched_cell_raises(ts):
    """Accumulating a trajectory with a different cell should raise ValueError."""
    gs = DensityGrid(ts, density_type="number", nbins=4)

    class MismatchedTrajectory:
        def __init__(self):
            self.box_x = self.box_y = self.box_z = 12.0
            self.cell_matrix = np.diag([12.0, 12.0, 12.0])
            self.units = "real"
            self.temperature = 300.0
            self.frames = 2
            from revelsMD.trajectories._base import compute_beta
            self.beta = compute_beta(self.units, self.temperature)
        def get_indices(self, atype):
            return np.array([0])
        def iter_frames(self, start=0, stop=None, stride=1):
            yield np.array([[1.0, 2.0, 3.0]]), np.array([[0.1, 0.0, 0.0]])

    with pytest.raises(ValueError, match="[Cc]ell"):
        gs.accumulate(MismatchedTrajectory(), atom_names="H")


class TestMakeForceGridUnified:
    """Test that accumulate using unified approach gives same results."""

    @pytest.fixture
    def trajectory(self):
        return IterableMockTrajectory()

    def test_accumulate_single_species_number(self, trajectory):
        """accumulate with single species number density produces correct grid."""

        gs = DensityGrid(trajectory, "number", nbins=5)
        gs.accumulate(trajectory, atom_names="O", rigid=False, start=0, stop=2)

        # Verify grid was populated
        assert gs.count == 2
        assert gs.counter.sum() > 0
        assert gs.count > 0  # Data has been accumulated


class TestSelectionGetWeights:
    """Tests for Selection.get_weights() method."""

    @pytest.fixture
    def trajectory(self):
        return MockTrajectory()

    @pytest.fixture
    def positions(self):
        """9 atoms: 3 water molecules arranged along x-axis."""
        return np.array([
            # Molecule 0: O at x=0, H1 at x=1, H2 at x=-1
            [0.0, 5.0, 5.0],   # O (index 0)
            [1.0, 5.0, 5.0],   # H1 (index 1)
            [-1.0, 5.0, 5.0],  # H2 (index 2)
            # Molecule 1: O at x=3, H1 at x=4, H2 at x=2
            [3.0, 5.0, 5.0],   # O (index 3)
            [4.0, 5.0, 5.0],   # H1 (index 4)
            [2.0, 5.0, 5.0],   # H2 (index 5)
            # Molecule 2: O at x=6, H1 at x=7, H2 at x=5
            [6.0, 5.0, 5.0],   # O (index 6)
            [7.0, 5.0, 5.0],   # H1 (index 7)
            [5.0, 5.0, 5.0],   # H2 (index 8)
        ], dtype=float)

    def test_number_density_returns_one(self, trajectory):
        """Number density should return weight of 1.0."""

        ss = Selection(trajectory, 'O', centre_location=True, rigid=False, density_type='number')
        result = ss.get_weights()

        assert result == 1.0

    def test_charge_single_species_returns_charges(self, trajectory):
        """Charge density for single species should return charge array."""

        ss = Selection(trajectory, 'O', centre_location=True, rigid=False, density_type='charge')
        result = ss.get_weights()

        expected = np.array([-0.8, -0.8, -0.8])  # O charges
        np.testing.assert_array_equal(result, expected)

    def test_charge_multi_species_not_rigid_returns_list(self, trajectory):
        """Charge density for multi-species non-rigid should return list of charge arrays."""

        ss = Selection(trajectory, ['O', 'H1', 'H2'], centre_location=True, rigid=False, density_type='charge')
        result = ss.get_weights()

        assert isinstance(result, list)
        assert len(result) == 3
        np.testing.assert_array_equal(result[0], np.array([-0.8, -0.8, -0.8]))  # O
        np.testing.assert_array_equal(result[1], np.array([0.4, 0.4, 0.4]))    # H1
        np.testing.assert_array_equal(result[2], np.array([0.4, 0.4, 0.4]))    # H2

    def test_charge_rigid_returns_summed_charges(self, trajectory):
        """Charge density for rigid should return total charge per molecule."""

        ss = Selection(trajectory, ['O', 'H1', 'H2'], centre_location=True, rigid=True, density_type='charge')
        result = ss.get_weights()

        # Total charge per molecule: -0.8 + 0.4 + 0.4 = 0.0
        expected = np.array([0.0, 0.0, 0.0])
        np.testing.assert_allclose(result, expected)

    def test_polarisation_returns_dipole_projection(self, trajectory, positions):
        """Polarisation density should return dipole projected along axis."""

        ss = Selection(
            trajectory, ['O', 'H1', 'H2'], centre_location=True, rigid=True,
            density_type='polarisation', polarisation_axis=0
        )
        result = ss.get_weights(positions)

        # COM for each molecule is at x = (16*x_O + 1*x_H1 + 1*x_H2) / 18
        # Molecule 0: COM_x = (16*0 + 1*1 + 1*(-1)) / 18 = 0
        # Dipole_x = q_O*(x_O - COM_x) + q_H1*(x_H1 - COM_x) + q_H2*(x_H2 - COM_x)
        #          = -0.8*(0 - 0) + 0.4*(1 - 0) + 0.4*(-1 - 0)
        #          = 0 + 0.4 - 0.4 = 0.0
        # Similarly for molecules 1 and 2
        expected = np.array([0.0, 0.0, 0.0])
        np.testing.assert_allclose(result, expected, atol=1e-10)


class TestSelectionValidation:
    """Tests for Selection input validation."""

    @pytest.fixture
    def trajectory(self):
        return MockTrajectory()

    def test_density_type_validation_called(self, trajectory, mocker):
        """Selection should call validate_density_type with the provided value."""
        mock_validate = mocker.patch(
            'revelsMD.density.selection.validate_density_type',
            return_value='number'
        )

        Selection(
            trajectory,
            atom_names='O',
            centre_location=True,
            density_type='NUMBER',
        )

        mock_validate.assert_called_once_with('NUMBER')


class TestSelectionGetForces:
    """Tests for Selection.get_forces() method."""

    @pytest.fixture
    def trajectory(self):
        return MockTrajectory()

    @pytest.fixture
    def forces(self):
        """Forces for 9 atoms (3 water molecules)."""
        return np.array([
            # Molecule 0
            [1.0, 0.0, 0.0],   # O (index 0)
            [0.5, 0.0, 0.0],   # H1 (index 1)
            [0.5, 0.0, 0.0],   # H2 (index 2)
            # Molecule 1
            [2.0, 0.0, 0.0],   # O (index 3)
            [1.0, 0.0, 0.0],   # H1 (index 4)
            [1.0, 0.0, 0.0],   # H2 (index 5)
            # Molecule 2
            [3.0, 0.0, 0.0],   # O (index 6)
            [1.5, 0.0, 0.0],   # H1 (index 7)
            [1.5, 0.0, 0.0],   # H2 (index 8)
        ], dtype=float)

    def test_single_species_returns_forces_at_indices(self, trajectory, forces):
        """Single species should return forces at selected indices."""

        ss = Selection(trajectory, 'O', centre_location=True, rigid=False)
        result = ss.get_forces(forces)

        expected = forces[[0, 3, 6], :]  # O atoms
        np.testing.assert_array_equal(result, expected)

    def test_multi_species_not_rigid_returns_list(self, trajectory, forces):
        """Multi-species, non-rigid should return list of force arrays."""

        ss = Selection(trajectory, ['O', 'H1', 'H2'], centre_location=True, rigid=False)
        result = ss.get_forces(forces)

        assert isinstance(result, list)
        assert len(result) == 3
        np.testing.assert_array_equal(result[0], forces[[0, 3, 6], :])  # O
        np.testing.assert_array_equal(result[1], forces[[1, 4, 7], :])  # H1
        np.testing.assert_array_equal(result[2], forces[[2, 5, 8], :])  # H2

    def test_rigid_sums_forces_across_molecule(self, trajectory, forces):
        """Rigid molecule should sum forces across all atoms in molecule."""

        ss = Selection(trajectory, ['O', 'H1', 'H2'], centre_location=True, rigid=True)
        result = ss.get_forces(forces)

        # Sum forces for each molecule
        # Molecule 0: [1,0,0] + [0.5,0,0] + [0.5,0,0] = [2,0,0]
        # Molecule 1: [2,0,0] + [1,0,0] + [1,0,0] = [4,0,0]
        # Molecule 2: [3,0,0] + [1.5,0,0] + [1.5,0,0] = [6,0,0]
        assert result.shape == (3, 3)
        np.testing.assert_allclose(result[:, 0], [2.0, 4.0, 6.0])
        np.testing.assert_allclose(result[:, 1], [0.0, 0.0, 0.0])


class TestSelectionGetPositionsPeriodicBoundary:
    """Tests for Selection.get_positions() with molecules spanning periodic boundaries."""

    @pytest.fixture
    def trajectory(self):
        """Single molecule trajectory."""
        class SingleMolTrajectory:
            def __init__(self):
                self.box_x = self.box_y = self.box_z = 10.0
                self.cell_matrix = np.diag([10.0, 10.0, 10.0])
                self.units = 'real'

            def get_indices(self, atom_name):
                return {'O': np.array([0]), 'H1': np.array([1]), 'H2': np.array([2])}[atom_name]

            def get_masses(self, atom_name):
                return {'O': np.array([16.0]), 'H1': np.array([1.0]), 'H2': np.array([1.0])}[atom_name]

            def get_charges(self, atom_name):
                return {'O': np.array([-0.8]), 'H1': np.array([0.4]), 'H2': np.array([0.4])}[atom_name]

        return SingleMolTrajectory()

    @pytest.fixture
    def positions_across_boundary(self):
        """Molecule spanning periodic boundary: O at x=9.5, H1 at x=0.3, H2 at x=0.5."""
        return np.array([
            [9.5, 5.0, 5.0],   # O (index 0) - near right edge
            [0.3, 5.0, 5.0],   # H1 (index 1) - wrapped to left edge
            [0.5, 5.0, 5.0],   # H2 (index 2) - wrapped to left edge
        ], dtype=float)

    def test_com_with_molecule_spanning_periodic_boundary(self, trajectory, positions_across_boundary):
        """COM should handle molecules that span periodic boundaries correctly."""

        # Box is 10x10x10, molecule spans boundary in x
        # O at x=9.5, H1 at x=0.3 (really at x=10.3, i.e. 0.8 from O)
        # H2 at x=0.5 (really at x=10.5, i.e. 1.0 from O)
        ss = Selection(
            trajectory, ['O', 'H1', 'H2'], centre_location=True, rigid=True
        )
        result = ss.get_positions(positions_across_boundary)

        # With minimum image: H1 is at x=10.3, H2 is at x=10.5 relative to O
        # COM_x = (16*9.5 + 1*10.3 + 1*10.5) / 18 = (152 + 10.3 + 10.5) / 18 = 172.8 / 18 = 9.6
        # Then wrapped to box: 9.6 (already in box)
        #
        # Without minimum image (BUG):
        # COM_x = (16*9.5 + 1*0.3 + 1*0.5) / 18 = (152 + 0.8) / 18 = 8.49 (WRONG!)

        # The COM should be near x=9.6, not x=8.49
        assert result[0, 0] > 9.0, f"COM x={result[0, 0]} should be > 9.0 (near the O atom)"

    def test_dipole_with_molecule_spanning_periodic_boundary(self, trajectory, positions_across_boundary):
        """Dipole calculation should handle molecules that span periodic boundaries."""

        ss = Selection(
            trajectory, ['O', 'H1', 'H2'], centre_location=True, rigid=True,
            density_type='polarisation', polarisation_axis=0
        )
        result = ss.get_weights(positions_across_boundary)

        # The molecule is symmetric around COM in y and z, but asymmetric in x
        # H1 is at x=10.3 (0.7 from COM at 9.6), H2 is at x=10.5 (0.9 from COM)
        # O is at x=9.5 (-0.1 from COM)
        # Dipole_x = q_O*(x_O - COM_x) + q_H1*(x_H1 - COM_x) + q_H2*(x_H2 - COM_x)
        #          = -0.8*(-0.1) + 0.4*(0.7) + 0.4*(0.9)
        #          = 0.08 + 0.28 + 0.36 = 0.72
        #
        # Without minimum image (BUG), H atoms would appear far from COM:
        # Dipole_x = -0.8*(-0.1) + 0.4*(0.3-9.6) + 0.4*(0.5-9.6)
        #          = 0.08 + 0.4*(-9.3) + 0.4*(-9.1) = 0.08 - 3.72 - 3.64 = -7.28

        # The dipole should be small and positive, not large and negative
        assert result[0] > 0, f"Dipole x={result[0]} should be > 0"
        assert result[0] < 1.0, f"Dipole x={result[0]} should be < 1.0 (small molecule)"


class TestSelectionExtract:
    """Tests for Selection.extract() method."""

    def test_extract_returns_tuple_of_three(self):
        """extract() should return a tuple of (positions, forces, weights)."""
        trajectory = MockTrajectory()
        positions = np.zeros((9, 3))
        forces = np.zeros((9, 3))

        ss = Selection(trajectory, 'O', centre_location=True, rigid=False, density_type='number')
        result = ss.extract(positions, forces)

        assert isinstance(result, tuple)
        assert len(result) == 3


class TestSelectionGetPositions:
    """Tests for Selection.get_positions() method."""

    @pytest.fixture
    def trajectory(self):
        return MockTrajectory()

    @pytest.fixture
    def positions(self):
        """9 atoms: 3 water molecules arranged along x-axis."""
        return np.array([
            # Molecule 0: O at x=0, H1 at x=0.5, H2 at x=-0.5
            [0.0, 5.0, 5.0],   # O (index 0)
            [0.5, 5.0, 5.0],   # H1 (index 1)
            [-0.5, 5.0, 5.0],  # H2 (index 2)
            # Molecule 1: O at x=3, H1 at x=3.5, H2 at x=2.5
            [3.0, 5.0, 5.0],   # O (index 3)
            [3.5, 5.0, 5.0],   # H1 (index 4)
            [2.5, 5.0, 5.0],   # H2 (index 5)
            # Molecule 2: O at x=6, H1 at x=6.5, H2 at x=5.5
            [6.0, 5.0, 5.0],   # O (index 6)
            [6.5, 5.0, 5.0],   # H1 (index 7)
            [5.5, 5.0, 5.0],   # H2 (index 8)
        ], dtype=float)

    def test_single_species_returns_positions_at_indices(self, trajectory, positions):
        """Single species should return positions at selected indices."""

        ss = Selection(trajectory, 'O', centre_location=True, rigid=False)
        result = ss.get_positions(positions)

        expected = positions[[0, 3, 6], :]  # O atoms
        np.testing.assert_array_equal(result, expected)

    def test_multi_species_not_rigid_returns_list(self, trajectory, positions):
        """Multi-species, non-rigid should return list of position arrays."""

        ss = Selection(trajectory, ['O', 'H1', 'H2'], centre_location=True, rigid=False)
        result = ss.get_positions(positions)

        assert isinstance(result, list)
        assert len(result) == 3
        np.testing.assert_array_equal(result[0], positions[[0, 3, 6], :])  # O
        np.testing.assert_array_equal(result[1], positions[[1, 4, 7], :])  # H1
        np.testing.assert_array_equal(result[2], positions[[2, 5, 8], :])  # H2

    def test_rigid_com_returns_center_of_mass(self, trajectory, positions):
        """Rigid molecule with COM should return mass-weighted center positions."""

        ss = Selection(trajectory, ['O', 'H1', 'H2'], centre_location=True, rigid=True)
        result = ss.get_positions(positions)

        # COM for each molecule (masses: O=16, H1=1, H2=1, total=18)
        # Molecule 0: (16*0 + 1*0.5 + 1*(-0.5)) / 18 = 0/18 = 0.0
        # Molecule 1: (16*3 + 1*3.5 + 1*2.5) / 18 = 54/18 = 3.0
        # Molecule 2: (16*6 + 1*6.5 + 1*5.5) / 18 = 108/18 = 6.0
        assert result.shape == (3, 3)
        np.testing.assert_allclose(result[:, 0], [0.0, 3.0, 6.0], rtol=1e-10)
        np.testing.assert_allclose(result[:, 1], [5.0, 5.0, 5.0], rtol=1e-10)

    def test_rigid_specific_atom_returns_that_atoms_positions(self, trajectory, positions):
        """Rigid molecule with specific atom index should return that atom's positions."""

        # centre_location=1 means use H1 positions
        ss = Selection(trajectory, ['O', 'H1', 'H2'], centre_location=1, rigid=True)
        result = ss.get_positions(positions)

        expected = positions[[1, 4, 7], :]  # H1 atoms
        np.testing.assert_array_equal(result, expected)


# ---------------------------------------------------------------------------
# Triclinic FFT validation tests
# ---------------------------------------------------------------------------

class TestTriclinicFFT:
    """Tests for the Borgis density formula with triclinic cells."""

    def test_ideal_gas_triclinic_flat_density(self):
        """Uniform random positions + zero forces -> flat density."""
        from revelsMD.trajectories.numpy import NumpyTrajectory

        cell = np.array([
            [10.0, 0.0, 0.0],
            [3.0, 9.0, 0.0],
            [0.0, 0.0, 8.0],
        ])
        n_atoms = 50
        n_frames = 10
        nbins = 8
        rng = np.random.default_rng(42)

        # Random positions in fractional coordinates, then convert to Cartesian
        frac = rng.random((n_frames, n_atoms, 3))
        positions = np.einsum('fai,ij->faj', frac, cell)
        forces = np.zeros((n_frames, n_atoms, 3))

        traj = NumpyTrajectory(
            positions=positions, forces=forces,
            cell_matrix=cell,
            species_list=["A"] * n_atoms,
            temperature=300.0, units="real",
        )
        gs = DensityGrid(traj, density_type="number", nbins=nbins)
        for i in range(n_frames):
            gs._process_frame(positions[i], forces[i], weight=1.0)

        # Force-based density should be flat (all perturbation is zero)
        rho_force, rho_count, _, _ = gs._fft_force_to_density(
            gs.force_x, gs.force_y, gs.force_z, gs.counter, gs.count
        )
        # rho_force should equal mean(rho_count) everywhere
        np.testing.assert_allclose(
            rho_force, np.mean(rho_count), rtol=0.3,
            err_msg="Ideal gas (zero forces) should produce approximately flat density",
        )

    def test_sinusoidal_force_triclinic(self):
        """Sinusoidal force at a reciprocal lattice vector should produce
        the expected density perturbation in a triclinic cell."""
        from revelsMD.trajectories.numpy import NumpyTrajectory

        cell = np.array([
            [10.0, 0.0, 0.0],
            [3.0, 9.0, 0.0],
            [0.0, 0.0, 8.0],
        ])
        n_atoms = 500
        n_frames = 20
        nbins = 16
        rng = np.random.default_rng(99)

        # Use a reciprocal lattice vector: k = 2*pi * inv(M)^T @ [1,0,0]
        M_inv_T = np.linalg.inv(cell).T
        k0 = 2 * np.pi * M_inv_T @ np.array([1.0, 0.0, 0.0])
        F0 = 0.1  # force amplitude

        # Uniformly distributed positions
        frac = rng.random((n_frames, n_atoms, 3))
        positions = np.einsum('fai,ij->faj', frac, cell)

        # Force = F0 * sin(k0 . r) in the x-direction
        # k0 . r for each frame/atom
        k_dot_r = np.einsum('fai,i->fa', positions, k0)
        forces = np.zeros((n_frames, n_atoms, 3))
        forces[:, :, 0] = F0 * np.sin(k_dot_r)

        traj = NumpyTrajectory(
            positions=positions, forces=forces,
            cell_matrix=cell,
            species_list=["A"] * n_atoms,
            temperature=300.0, units="real",
        )
        gs = DensityGrid(traj, density_type="number", nbins=nbins)
        for i in range(n_frames):
            gs._process_frame(positions[i], forces[i], weight=1.0)

        rho_force, rho_count, del_rho_k, _ = gs._fft_force_to_density(
            gs.force_x, gs.force_y, gs.force_z, gs.counter, gs.count
        )

        # The density perturbation should be non-zero (force is non-trivial)
        assert np.max(np.abs(rho_force - np.mean(rho_force))) > 1e-6, \
            "Sinusoidal force should produce non-trivial density perturbation"

        # The del_rho_k should have dominant peaks at the Miller indices [1,0,0]
        # and [-1,0,0] (the applied k-vector and its conjugate)
        del_rho_k_abs = np.abs(del_rho_k)
        # Zero the DC component
        del_rho_k_abs[0, 0, 0] = 0
        # The peak should be at index [1,0,0] or [-1,0,0] = [nbins-1,0,0]
        peak_pos = np.unravel_index(np.argmax(del_rho_k_abs), del_rho_k_abs.shape)
        assert peak_pos[1] == 0 and peak_pos[2] == 0, \
            f"Peak should be at [*,0,0] Miller indices, got {peak_pos}"
        assert peak_pos[0] in (1, nbins - 1), \
            f"Peak should be at Miller index m1=1 or {nbins-1}, got {peak_pos[0]}"

    def test_orthorhombic_triclinic_paths_agree(self):
        """For a diagonal cell, orthorhombic and triclinic FFT paths should
        produce identical results."""
        from revelsMD.trajectories.numpy import NumpyTrajectory

        cell = np.diag([10.0, 8.0, 6.0])
        n_atoms = 20
        n_frames = 5
        nbins = 8
        rng = np.random.default_rng(123)

        positions = rng.random((n_frames, n_atoms, 3)) * np.array([10, 8, 6])
        forces = rng.standard_normal((n_frames, n_atoms, 3))

        traj = NumpyTrajectory(
            positions=positions, forces=forces,
            cell_matrix=cell,
            species_list=["A"] * n_atoms,
            temperature=300.0, units="real",
        )

        # Build grid (will use orthorhombic path since cell is diagonal)
        gs_ortho = DensityGrid(traj, density_type="number", nbins=nbins)
        assert gs_ortho.is_orthorhombic is True
        for i in range(n_frames):
            gs_ortho._process_frame(positions[i], forces[i], weight=1.0)

        # Compute density using orthorhombic path
        rho_ortho, _, _, _ = gs_ortho._fft_force_to_density(
            gs_ortho.force_x, gs_ortho.force_y, gs_ortho.force_z,
            gs_ortho.counter, gs_ortho.count
        )

        # Now force the triclinic path by building k-vectors and using them
        gs_tri = DensityGrid(traj, density_type="number", nbins=nbins)
        for i in range(n_frames):
            gs_tri._process_frame(positions[i], forces[i], weight=1.0)

        # Manually invoke triclinic FFT path
        k_vectors, ksquared = gs_tri._build_kvectors_3d()
        # Verify k-vectors match the orthorhombic ones
        kx_1d, ky_1d, kz_1d = gs_tri.get_kvectors()
        ksq_ortho = gs_tri.get_ksquared()
        np.testing.assert_allclose(ksquared, ksq_ortho, atol=1e-10)

        # Both grids have the same accumulated data
        np.testing.assert_allclose(gs_ortho.force_x, gs_tri.force_x)

        # The orthorhombic rho_force and a manually-computed triclinic rho_force
        # should agree
        rho_tri, _, _, _ = gs_tri._fft_force_to_density(
            gs_tri.force_x, gs_tri.force_y, gs_tri.force_z,
            gs_tri.counter, gs_tri.count
        )
        np.testing.assert_allclose(rho_ortho, rho_tri, atol=1e-10)


# ---------------------------------------------------------------------------
# compute_density() convenience function
# ---------------------------------------------------------------------------

class TestComputeDensity:
    """Tests for compute_density() convenience function."""

    @pytest.fixture
    def trajectory(self):
        return IterableMockTrajectory()

    def test_compute_density_returns_densitygrid(self, trajectory):
        """compute_density should return a DensityGrid with computed density."""
        from revelsMD.density import compute_density

        result = compute_density(trajectory, atom_names='O', nbins=5)

        assert isinstance(result, DensityGrid)
        assert hasattr(result, 'rho_force')
        assert result.rho_force.shape == (5, 5, 5)
        assert np.all(np.isfinite(result.rho_force))

    def test_compute_density_with_rigid_molecules(self, trajectory):
        """compute_density should work with rigid molecules."""
        from revelsMD.density import compute_density

        result = compute_density(
            trajectory,
            atom_names=['O', 'H1', 'H2'],
            rigid=True,
            nbins=5
        )

        assert hasattr(result, 'rho_force')
        assert result.rho_force.shape == (5, 5, 5)

    def test_compute_density_importable_from_density(self):
        """compute_density should be importable from revelsMD.density."""
        from revelsMD.density import compute_density
        assert compute_density is not None

    @pytest.fixture
    def trajectory_with_get_frame(self):
        """Create mock trajectory that supports both iteration and get_frame."""
        class IterableMockTrajectoryWithGetFrame(MockTrajectory):
            def __init__(self):
                super().__init__()
                self.frames = 4
                # 9 atoms: 3 water molecules (O, H1, H2 each)
                self._positions = [
                    np.array([
                        [1.0, 5.0, 5.0], [1.5, 5.0, 5.0], [0.5, 5.0, 5.0],
                        [4.0, 5.0, 5.0], [4.5, 5.0, 5.0], [3.5, 5.0, 5.0],
                        [7.0, 5.0, 5.0], [7.5, 5.0, 5.0], [6.5, 5.0, 5.0],
                    ], dtype=float),
                    np.array([
                        [1.1, 5.1, 5.0], [1.6, 5.1, 5.0], [0.6, 5.1, 5.0],
                        [4.1, 5.1, 5.0], [4.6, 5.1, 5.0], [3.6, 5.1, 5.0],
                        [7.1, 5.1, 5.0], [7.6, 5.1, 5.0], [6.6, 5.1, 5.0],
                    ], dtype=float),
                    np.array([
                        [1.2, 5.2, 5.0], [1.7, 5.2, 5.0], [0.7, 5.2, 5.0],
                        [4.2, 5.2, 5.0], [4.7, 5.2, 5.0], [3.7, 5.2, 5.0],
                        [7.2, 5.2, 5.0], [7.7, 5.2, 5.0], [6.7, 5.2, 5.0],
                    ], dtype=float),
                    np.array([
                        [1.3, 5.3, 5.0], [1.8, 5.3, 5.0], [0.8, 5.3, 5.0],
                        [4.3, 5.3, 5.0], [4.8, 5.3, 5.0], [3.8, 5.3, 5.0],
                        [7.3, 5.3, 5.0], [7.8, 5.3, 5.0], [6.8, 5.3, 5.0],
                    ], dtype=float),
                ]
                self._forces = [
                    np.array([
                        [1.0, 0.1, 0.0], [0.5, 0.05, 0.0], [0.5, 0.05, 0.0],
                        [2.0, 0.2, 0.0], [1.0, 0.1, 0.0], [1.0, 0.1, 0.0],
                        [3.0, 0.3, 0.0], [1.5, 0.15, 0.0], [1.5, 0.15, 0.0],
                    ], dtype=float),
                    np.array([
                        [1.1, 0.11, 0.0], [0.55, 0.055, 0.0], [0.55, 0.055, 0.0],
                        [2.2, 0.22, 0.0], [1.1, 0.11, 0.0], [1.1, 0.11, 0.0],
                        [3.3, 0.33, 0.0], [1.65, 0.165, 0.0], [1.65, 0.165, 0.0],
                    ], dtype=float),
                    np.array([
                        [1.2, 0.12, 0.0], [0.6, 0.06, 0.0], [0.6, 0.06, 0.0],
                        [2.4, 0.24, 0.0], [1.2, 0.12, 0.0], [1.2, 0.12, 0.0],
                        [3.6, 0.36, 0.0], [1.8, 0.18, 0.0], [1.8, 0.18, 0.0],
                    ], dtype=float),
                    np.array([
                        [1.3, 0.13, 0.0], [0.65, 0.065, 0.0], [0.65, 0.065, 0.0],
                        [2.6, 0.26, 0.0], [1.3, 0.13, 0.0], [1.3, 0.13, 0.0],
                        [3.9, 0.39, 0.0], [1.95, 0.195, 0.0], [1.95, 0.195, 0.0],
                    ], dtype=float),
                ]

            def iter_frames(self, start, stop, period):
                for i in range(start, stop or self.frames, period):
                    yield self._positions[i], self._forces[i]

            def get_frame(self, idx):
                return self._positions[idx], self._forces[idx]

        return IterableMockTrajectoryWithGetFrame()

    def test_compute_density_compute_lambda(self, trajectory_with_get_frame):
        """compute_density with compute_lambda=True populates rho_lambda."""
        from revelsMD.density import compute_density

        grid = compute_density(
            trajectory_with_get_frame,
            atom_names='O',
            nbins=5,
            compute_lambda=True,
            sections=2,
        )

        assert grid.rho_lambda is not None
        assert grid.rho_lambda.shape == (5, 5, 5)

    def test_compute_density_standard_default(self, trajectory):
        """Default compute_lambda=False behaves as before."""
        from revelsMD.density import compute_density

        grid = compute_density(trajectory, atom_names='O', nbins=5)

        assert grid.rho_force is not None
        assert grid.rho_lambda is None
        assert grid.count > 0  # Data has been accumulated

    def test_compute_density_integration_deprecated(self, trajectory_with_get_frame):
        """integration='lambda' still works but emits DeprecationWarning."""
        from revelsMD.density import compute_density

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            grid = compute_density(
                trajectory_with_get_frame,
                atom_names='O',
                nbins=5,
                integration='lambda',
                sections=2,
            )

        # Check that at least one DeprecationWarning with expected message was emitted
        assert any(
            issubclass(warn.category, DeprecationWarning)
            and "compute_lambda=True" in str(warn.message)
            for warn in w
        )
        assert grid.rho_lambda is not None

    def test_compute_density_invalid_integration(self, trajectory):
        """Invalid integration raises ValueError."""
        from revelsMD.density import compute_density

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            with pytest.raises(ValueError, match="integration"):
                compute_density(trajectory, atom_names='O', nbins=5, integration='invalid')


# ---------------------------------------------------------------------------
# DensityGrid.get_lambda() edge case tests
# ---------------------------------------------------------------------------

class TestDensityGridGetLambdaEdgeCases:
    """Tests for edge case handling in DensityGrid.get_lambda()."""

    def test_get_lambda_produces_finite_output(self):
        """get_lambda produces finite combination and optimal_density values.

        This test verifies the fix for the zero-variance edge case bug where
        division by zero could produce NaN/Inf in the output.
        """
        from revelsMD.density import DensityGrid, Selection

        # Create a minimal trajectory with very few frames
        # This increases the chance of zero-variance voxels
        class MinimalTrajectory:
            def __init__(self):
                self.box_x = self.box_y = self.box_z = 10.0
                self.cell_matrix = np.diag([10.0, 10.0, 10.0])
                self.units = 'real'
                self.frames = 2
                self.beta = 1.0 / (300.0 * 0.0019872041)

            def get_indices(self, atom_name):
                return np.array([0, 1])

            def get_masses(self, atom_name):
                return np.array([1.0, 1.0])

            def get_frame(self, idx):
                # Return identical positions for all frames to create zero variance
                positions = np.array([[2.0, 5.0, 5.0], [8.0, 5.0, 5.0]])
                forces = np.array([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]])
                return positions, forces

            def iter_frames(self, start, stop, period):
                for i in range(start, stop or self.frames, period):
                    yield self.get_frame(i)

        traj = MinimalTrajectory()
        gs = DensityGrid(traj, "number", nbins=3)
        ss = Selection(traj, 'H', centre_location=True, rigid=False, density_type='number')
        gs._selection = ss
        gs.kernel = "triangular"
        gs.to_run = list(range(traj.frames))

        # Manually deposit frames
        for positions, forces in traj.iter_frames(0, traj.frames, 1):
            gs.deposit(
                ss.get_positions(positions),
                ss.get_forces(forces),
                ss.get_weights(),
                kernel="triangular"
            )

        gs.get_lambda(traj, sections=2)

        # The key assertion: no NaN or Inf values
        assert np.all(np.isfinite(gs.lambda_weights)), "lambda_weights contains NaN/Inf"
        assert np.all(np.isfinite(gs.rho_lambda)), "rho_lambda contains NaN/Inf"


# ---------------------------------------------------------------------------
# Deprecated property alias tests
# ---------------------------------------------------------------------------

class TestDeprecatedPropertyAliases:
    """Tests for deprecated property aliases on DensityGrid."""

    def test_optimal_density_emits_deprecation_warning(self, ts):
        """optimal_density should emit DeprecationWarning."""
        gs = DensityGrid(ts, "number", nbins=4)
        gs.accumulate(ts, atom_names="H", compute_lambda=True, sections=2)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = gs.optimal_density

        dep_warnings = [warn for warn in w if issubclass(warn.category, DeprecationWarning)]
        assert dep_warnings, "Expected DeprecationWarning from optimal_density"
        assert any("optimal_density" in str(warn.message) for warn in dep_warnings)

    def test_combination_emits_deprecation_warning(self, ts):
        """combination should emit DeprecationWarning."""
        gs = DensityGrid(ts, "number", nbins=4)
        gs.accumulate(ts, atom_names="H", compute_lambda=True, sections=2)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = gs.combination

        dep_warnings = [warn for warn in w if issubclass(warn.category, DeprecationWarning)]
        assert dep_warnings, "Expected DeprecationWarning from combination"
        assert any("combination" in str(warn.message) for warn in dep_warnings)

    def test_optimal_density_returns_same_as_rho_lambda(self, ts):
        """optimal_density should return the same value as rho_lambda."""
        gs = DensityGrid(ts, "number", nbins=4)
        gs.accumulate(ts, atom_names="H", compute_lambda=True, sections=2)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            # Access rho_lambda first to trigger finalisation
            rho_lambda = gs.rho_lambda
            assert gs.optimal_density is rho_lambda

    def test_combination_returns_same_as_lambda_weights(self, ts):
        """combination should return the same value as lambda_weights."""
        gs = DensityGrid(ts, "number", nbins=4)
        gs.accumulate(ts, atom_names="H", compute_lambda=True, sections=2)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            # Access lambda_weights first to trigger finalisation
            lambda_weights = gs.lambda_weights
            assert gs.combination is lambda_weights


# ---------------------------------------------------------------------------
# DensityGrid.write_to_cube() tests
# ---------------------------------------------------------------------------

class TestWriteToCube:
    """Tests for DensityGrid.write_to_cube() method."""

    def test_write_to_cube_creates_file(self, tmp_path, ts):
        """write_to_cube creates a cube file."""
        gs = DensityGrid(ts, "number", nbins=4)
        gs.accumulate(ts, atom_names="H", rigid=False)
        gs.get_real_density()

        cube_file = tmp_path / "test.cube"
        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1]])
        gs.write_to_cube(atoms, gs.rho_force, cube_file)

        assert cube_file.exists()

    def test_write_to_cube_with_pymatgen_structure(self, tmp_path, ts):
        """write_to_cube handles pymatgen Structure input."""
        from pymatgen.core import Structure, Lattice

        gs = DensityGrid(ts, "number", nbins=4)
        gs.accumulate(ts, atom_names="H", rigid=False)
        gs.get_real_density()

        structure = Structure(
            Lattice.cubic(10.0),
            ["H", "H"],
            [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
        )

        cube_file = tmp_path / "test_pymatgen.cube"
        gs.write_to_cube(structure, gs.rho_force, cube_file)

        assert cube_file.exists()

    def test_write_to_cube_invalid_path_raises(self, ts):
        """write_to_cube with invalid path raises appropriate error."""
        gs = DensityGrid(ts, "number", nbins=4)
        gs.accumulate(ts, atom_names="H", rigid=False)
        gs.get_real_density()

        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1]])

        with pytest.raises((OSError, FileNotFoundError)):
            gs.write_to_cube(atoms, gs.rho_force, "/nonexistent/path/test.cube")


# ---------------------------------------------------------------------------
# Compute-on-demand tests for rho_force / rho_count
# ---------------------------------------------------------------------------

class TestComputeOnDemand:
    """Tests for compute-on-demand behaviour of rho_force and rho_count."""

    @pytest.mark.parametrize("attr", ["rho_force", "rho_count"])
    def test_density_computes_on_demand(self, ts, attr):
        """rho_force/rho_count should compute automatically without calling get_real_density()."""
        gs = DensityGrid(ts, "number", nbins=4)
        gs.accumulate(ts, atom_names="H", rigid=False)

        result = getattr(gs, attr)

        assert result is not None
        assert result.shape == (4, 4, 4)
        assert np.any(result != 0)

    @pytest.mark.parametrize("attr", ["rho_force", "rho_count"])
    def test_density_is_cached(self, ts, attr):
        """rho_force/rho_count should return the same cached object on repeated access."""
        gs = DensityGrid(ts, "number", nbins=4)
        gs.accumulate(ts, atom_names="H", rigid=False)

        first_access = getattr(gs, attr)
        second_access = getattr(gs, attr)

        assert first_access is second_access

    def test_accumulate_clears_cached_densities(self, ts):
        """accumulate() should clear cached rho_force/rho_count."""
        gs = DensityGrid(ts, "number", nbins=4)
        gs.accumulate(ts, atom_names="H", rigid=False)

        # Access to cache the result
        first_rho_force = gs.rho_force
        assert first_rho_force is not None

        # Accumulate again
        gs.accumulate(ts, atom_names="H", rigid=False)

        # Should have cleared the cache and recomputed
        second_rho_force = gs.rho_force
        assert second_rho_force is not first_rho_force

    @pytest.mark.parametrize("attr", ["rho_force", "rho_count"])
    def test_density_returns_none_before_accumulate(self, ts, attr):
        """rho_force/rho_count should return None before any accumulation."""
        gs = DensityGrid(ts, "number", nbins=4)

        assert getattr(gs, attr) is None

    def test_get_real_density_emits_deprecation_warning(self, ts):
        """get_real_density() should emit a DeprecationWarning."""
        gs = DensityGrid(ts, "number", nbins=4)
        gs.accumulate(ts, atom_names="H", rigid=False)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            gs.get_real_density()

        dep_warnings = [warn for warn in w if issubclass(warn.category, DeprecationWarning)]
        assert dep_warnings, "Expected at least one DeprecationWarning"
        assert any("get_real_density" in str(warn.message) for warn in dep_warnings)

    def test_get_real_density_still_works(self, ts):
        """get_real_density() should still work for backward compatibility."""
        gs = DensityGrid(ts, "number", nbins=4)
        gs.accumulate(ts, atom_names="H", rigid=False)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            gs.get_real_density()

        # Should have populated the densities
        assert gs.rho_force is not None
        assert gs.rho_count is not None
        assert np.any(gs.rho_force != 0)

    def test_rho_lambda_uses_cached_densities(self, ts):
        """rho_lambda finalisation should use cached rho_force if available."""
        gs = DensityGrid(ts, "number", nbins=4)
        gs.accumulate(ts, atom_names="H", compute_lambda=True, sections=2)

        # Access rho_force first (caches it)
        rho_force_cached = gs.rho_force

        # Now access rho_lambda (should use cached rho_force)
        _ = gs.rho_lambda

        # The cached rho_force should still be the same object
        assert gs.rho_force is rho_force_cached

    def test_fft_density_calculation_consistent(self, ts):
        """FFT density calculation should be consistent between main and array methods."""
        gs = DensityGrid(ts, "number", nbins=4)
        gs.accumulate(ts, atom_names="H", rigid=False)

        # Compute via the main method (uses self.force_x, etc.)
        rho_force_main = gs.rho_force
        rho_count_main = gs.rho_count

        # Compute via the array method with the same data
        rho_force_array, rho_count_array = gs._compute_densities_from_arrays(
            gs.force_x, gs.force_y, gs.force_z, gs.counter, gs.count
        )

        # Results should be identical
        np.testing.assert_array_equal(rho_force_main, rho_force_array)
        np.testing.assert_array_equal(rho_count_main, rho_count_array)


# ---------------------------------------------------------------------------
# rho_hybrid tests
# ---------------------------------------------------------------------------

class TestHybridDensity:
    """Tests for DensityGrid.rho_hybrid()."""

    def test_selects_force_above_and_count_below_threshold(self, ts):
        """Result should exactly match np.where(rho_count >= threshold, rho_force, rho_count)."""
        gs = DensityGrid(ts, "number", nbins=4)
        gs.accumulate(ts, atom_names="H", rigid=False)

        threshold = 0.01
        expected = np.where(gs.rho_count >= threshold, gs.rho_force, gs.rho_count)

        np.testing.assert_array_equal(gs.rho_hybrid(threshold), expected)

    def test_zero_threshold_returns_force_everywhere(self, ts):
        """Threshold of zero means rho_count >= 0 always, so result equals rho_force."""
        gs = DensityGrid(ts, "number", nbins=4)
        gs.accumulate(ts, atom_names="H", rigid=False)

        np.testing.assert_array_equal(gs.rho_hybrid(0.0), gs.rho_force)

    def test_large_threshold_returns_count_everywhere(self, ts):
        """Threshold above max rho_count means all voxels use rho_count."""
        gs = DensityGrid(ts, "number", nbins=4)
        gs.accumulate(ts, atom_names="H", rigid=False)

        large_threshold = gs.rho_count.max() + 1.0

        np.testing.assert_array_equal(gs.rho_hybrid(large_threshold), gs.rho_count)

    def test_negative_threshold_raises_value_error(self, ts):
        """Negative threshold should raise ValueError."""
        gs = DensityGrid(ts, "number", nbins=4)
        gs.accumulate(ts, atom_names="H", rigid=False)

        with pytest.raises(ValueError, match="threshold must be non-negative"):
            gs.rho_hybrid(-0.1)

    def test_before_accumulate_raises_runtime_error(self, ts):
        """Calling rho_hybrid before accumulate should raise RuntimeError."""
        gs = DensityGrid(ts, "number", nbins=4)

        with pytest.raises(RuntimeError, match="No density data available"):
            gs.rho_hybrid(0.01)
