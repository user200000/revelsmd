"""Tests for revelsMD.density.constants module."""

import pytest

from revelsMD.density.constants import VALID_DENSITY_TYPES, validate_density_type


class TestValidateDensityType:
    """Tests for validate_density_type function."""

    def test_valid_types_accepted(self):
        """All valid density types should be accepted."""
        for density_type in VALID_DENSITY_TYPES:
            result = validate_density_type(density_type)
            assert result == density_type

    def test_invalid_type_raises(self):
        """Invalid density type should raise ValueError."""
        with pytest.raises(ValueError, match="density_type must be one of"):
            validate_density_type("invalid")

    def test_normalises_to_lowercase(self):
        """Should normalise to lowercase."""
        assert validate_density_type("NUMBER") == "number"
        assert validate_density_type("Charge") == "charge"
        assert validate_density_type("POLARISATION") == "polarisation"

    def test_strips_whitespace(self):
        """Should strip leading/trailing whitespace."""
        assert validate_density_type(" number ") == "number"
        assert validate_density_type("\tcharge\n") == "charge"

    def test_error_message_shows_original_value(self):
        """Error message should show the original (unnormalised) value."""
        with pytest.raises(ValueError, match="'INVALID'"):
            validate_density_type("INVALID")
