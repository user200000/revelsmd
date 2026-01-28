"""
pytest test suite for the generate_boltzmann() function.
"""

import pytest
import scipy.constants as constants
from revelsMD.utils import generate_boltzmann


def test_generate_boltzmann_lj():
    """Lennard-Jones reduced units should return exactly 1."""
    assert generate_boltzmann('lj') == 1.0


def test_generate_boltzmann_real():
    """Verify 'real' units match R / (calorie * 1000)."""
    expected = constants.physical_constants['molar gas constant'][0] / constants.calorie / 1000
    assert pytest.approx(generate_boltzmann('real'), rel=1e-12) == expected


def test_generate_boltzmann_metal():
    """Verify 'metal' units match Boltzmann constant in eV/K."""
    expected = constants.physical_constants['Boltzmann constant in eV/K'][0]
    assert pytest.approx(generate_boltzmann('metal'), rel=1e-12) == expected


def test_generate_boltzmann_mda():
    """Verify 'mda' units match R / 1000."""
    expected = constants.physical_constants['molar gas constant'][0] / 1000
    assert pytest.approx(generate_boltzmann('mda'), rel=1e-12) == expected


def test_generate_boltzmann_invalid_unit():
    """Unsupported unit names should raise ValueError."""
    with pytest.raises(ValueError):
        generate_boltzmann('quantum-donut')

