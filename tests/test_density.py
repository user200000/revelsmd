"""Tests for revelsMD.density package and its module structure."""

import pytest
import warnings


def test_selectionstate_importable_from_density():
    """SelectionState should be importable from revelsMD.density."""
    from revelsMD.density import SelectionState
    assert SelectionState is not None


def test_selectionstate_importable_from_submodule():
    """SelectionState should be importable from revelsMD.density.selection_state."""
    from revelsMD.density.selection_state import SelectionState
    assert SelectionState is not None


def test_selectionstate_backward_compatible_via_revels3d():
    """Revels3D.SelectionState should still work but emit deprecation warning."""
    from revelsMD.revels_3D import Revels3D
    from revelsMD.density import SelectionState
    with pytest.warns(DeprecationWarning, match="Revels3D.SelectionState is deprecated"):
        assert Revels3D.SelectionState is SelectionState


def test_helperfunctions_importable_from_density():
    """HelperFunctions should be importable from revelsMD.density."""
    from revelsMD.density import HelperFunctions
    assert HelperFunctions is not None


def test_helperfunctions_importable_from_submodule():
    """HelperFunctions should be importable from revelsMD.density.helper_functions."""
    from revelsMD.density.helper_functions import HelperFunctions
    assert HelperFunctions is not None


def test_helperfunctions_backward_compatible_via_revels3d():
    """Revels3D.HelperFunctions should still work but emit deprecation warning."""
    from revelsMD.revels_3D import Revels3D
    from revelsMD.density import HelperFunctions
    with pytest.warns(DeprecationWarning, match="Revels3D.HelperFunctions is deprecated"):
        assert Revels3D.HelperFunctions is HelperFunctions


def test_estimators_importable_from_density():
    """Estimators should be importable from revelsMD.density."""
    from revelsMD.density import Estimators
    assert Estimators is not None


def test_estimators_importable_from_submodule():
    """Estimators should be importable from revelsMD.density.estimators."""
    from revelsMD.density.estimators import Estimators
    assert Estimators is not None


def test_estimators_backward_compatible_via_revels3d():
    """Revels3D.Estimators should still work but emit deprecation warning."""
    from revelsMD.revels_3D import Revels3D
    from revelsMD.density import Estimators
    with pytest.warns(DeprecationWarning, match="Revels3D.Estimators is deprecated"):
        assert Revels3D.Estimators is Estimators


def test_gridstate_importable_from_density():
    """GridState should be importable from revelsMD.density."""
    from revelsMD.density import GridState
    assert GridState is not None


def test_gridstate_importable_from_submodule():
    """GridState should be importable from revelsMD.density.grid_state."""
    from revelsMD.density.grid_state import GridState
    assert GridState is not None


def test_gridstate_backward_compatible_via_revels3d():
    """Revels3D.GridState should still work but emit deprecation warning."""
    from revelsMD.revels_3D import Revels3D
    from revelsMD.density import GridState
    with pytest.warns(DeprecationWarning, match="Revels3D.GridState is deprecated"):
        assert Revels3D.GridState is GridState
