import pytest
import numpy as np
from revelsMD.revels_rdf import RevelsRDF


class TSMock:
    """Minimal trajectory-state mock with numpy variety for testing RDF calculations."""
    def __init__(self):
        self.box_x = 10.0
        self.box_y = 10.0
        self.box_z = 10.0
        self.units = "real"
        self.frames = 3
        self.variety = "numpy"
        self.charge_and_mass = True

        # 3 frames × 3 atoms × 3 coordinates
        self.positions = np.array([
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[1.1, 2.1, 3.1], [4.1, 5.1, 6.1], [7.1, 8.1, 9.1]],
            [[0.9, 1.9, 2.9], [3.9, 4.9, 5.9], [6.9, 7.9, 8.9]]
        ])
        self.forces = np.array([
            [[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]],
            [[0.2, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 0.2]],
            [[0.3, 0.0, 0.0], [0.0, 0.3, 0.0], [0.0, 0.0, 0.3]],
        ])

        self.species = ["H", "O", "H"]
        self._ids = {"H": np.array([0, 2]), "O": np.array([1])}
        self._charges = {"H": np.array([0.1, 0.1]), "O": np.array([-0.2])}
        self._masses = {"H": np.array([1.0, 1.0]), "O": np.array([16.0])}

    def get_indicies(self, atype):
        return self._ids[atype]

    def get_charges(self, atype):
        return self._charges[atype]

    def get_masses(self, atype):
        return self._masses[atype]


@pytest.fixture
def ts():
    """Provide a reusable trajectory-state mock."""
    return TSMock()


# -------------------------------
# single_frame_rdf_like
# -------------------------------

def test_single_frame_rdf_like(ts):
    bins = np.linspace(0, 5, 10)
    indicies = ts.get_indicies("H")
    result = RevelsRDF.single_frame_rdf_like(
        ts.positions[0],
        ts.forces[0],
        indicies,
        ts.box_x,
        ts.box_y,
        ts.box_z,
        bins,
    )
    assert result.shape == bins.shape
    assert np.isfinite(result).all()
    assert np.all(np.isreal(result))


# -------------------------------
# single_frame_rdf_unlike
# -------------------------------

def test_single_frame_rdf_unlike(ts):
    bins = np.linspace(0, 5, 10)
    indicies = [ts.get_indicies("H"), ts.get_indicies("O")]
    result = RevelsRDF.single_frame_rdf_unlike(
        ts.positions[0],
        ts.forces[0],
        indicies,
        ts.box_x,
        ts.box_y,
        ts.box_z,
        bins,
    )
    assert result.shape == bins.shape
    assert np.isfinite(result).all()
    assert np.all(np.isreal(result))


# -------------------------------
# run_rdf (like pairs)
# -------------------------------

def test_run_rdf_like_pairs(ts):
    result = RevelsRDF.run_rdf(
        ts,
        atom_a="H",
        atom_b="H",
        temp=300,
        delr=1.0,
        start=0,
        stop=2,
        period=1,
        rmax=True,
        from_zero=True,
    )
    # Result should be shape (2, n)
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == 2
    assert np.all(np.isfinite(result))


# -------------------------------
# run_rdf (unlike pairs)
# -------------------------------

def test_run_rdf_unlike_pairs(ts):
    result = RevelsRDF.run_rdf(
        ts,
        atom_a="H",
        atom_b="O",
        temp=300,
        delr=1.0,
        start=0,
        stop=2,
        period=1,
        rmax=False,
        from_zero=False,
    )
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == 2
    assert np.all(np.isfinite(result))


# -------------------------------
# run_rdf edge conditions
# -------------------------------

def test_run_rdf_invalid_frame(ts, capsys):
    out = RevelsRDF.run_rdf(ts, "H", "O", 300, start=10, stop=2)
    captured = capsys.readouterr()
    assert "First frame index" in captured.out or out is None


# -------------------------------
# run_rdf_lambda
# -------------------------------

def test_run_rdf_lambda_like(ts):
    result = RevelsRDF.run_rdf_lambda(
        ts,
        atom_a="H",
        atom_b="H",
        temp=300,
        delr=1.0,
        start=0,
        stop=2,
        period=1,
        rmax=True,
    )
    assert isinstance(result, np.ndarray)
    # Three columns: [r, combined RDF, λ]
    assert result.shape[1] == 3
    assert np.all(np.isfinite(result))
    assert np.all((result[:, 2] >= -1) & (result[:, 2] <= 2))  # λ values roughly bounded

