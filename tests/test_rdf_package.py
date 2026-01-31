"""Tests for the revelsMD.rdf package structure and deprecation warnings."""

import warnings

import pytest


class TestRDFPackageImports:
    """Test that RDF functions are importable from the new package location."""

    def test_run_rdf_importable_from_rdf_package(self):
        """run_rdf should be importable from revelsMD.rdf."""
        from revelsMD.rdf import run_rdf

        assert callable(run_rdf)

    def test_run_rdf_lambda_importable_from_rdf_package(self):
        """run_rdf_lambda should be importable from revelsMD.rdf."""
        from revelsMD.rdf import run_rdf_lambda

        assert callable(run_rdf_lambda)

    def test_single_frame_rdf_importable_from_rdf_package(self):
        """single_frame_rdf should be importable from revelsMD.rdf."""
        from revelsMD.rdf import single_frame_rdf

        assert callable(single_frame_rdf)


class TestRevelsRDFDeprecationWarnings:
    """Test that accessing RevelsRDF methods emits deprecation warnings."""

    def test_revelsrdf_run_rdf_emits_deprecation_warning(self):
        """RevelsRDF.run_rdf should emit deprecation warning."""
        from revelsMD.revels_rdf import RevelsRDF

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = RevelsRDF.run_rdf
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "run_rdf" in str(w[0].message)
            assert "revelsMD.rdf" in str(w[0].message)

    def test_revelsrdf_run_rdf_lambda_emits_deprecation_warning(self):
        """RevelsRDF.run_rdf_lambda should emit deprecation warning."""
        from revelsMD.revels_rdf import RevelsRDF

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = RevelsRDF.run_rdf_lambda
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "run_rdf_lambda" in str(w[0].message)
            assert "revelsMD.rdf" in str(w[0].message)

    def test_revelsrdf_single_frame_rdf_emits_deprecation_warning(self):
        """RevelsRDF.single_frame_rdf should emit deprecation warning."""
        from revelsMD.revels_rdf import RevelsRDF

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = RevelsRDF.single_frame_rdf
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "single_frame_rdf" in str(w[0].message)
            assert "revelsMD.rdf" in str(w[0].message)


class TestDeprecatedMethodsReturnCorrectFunctions:
    """Test that deprecated RevelsRDF methods return the same functions as the new imports."""

    def test_run_rdf_same_function(self):
        """RevelsRDF.run_rdf should return the same function as revelsMD.rdf.run_rdf."""
        from revelsMD.rdf import run_rdf

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from revelsMD.revels_rdf import RevelsRDF

            assert RevelsRDF.run_rdf is run_rdf

    def test_run_rdf_lambda_same_function(self):
        """RevelsRDF.run_rdf_lambda should return the same function as revelsMD.rdf.run_rdf_lambda."""
        from revelsMD.rdf import run_rdf_lambda

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from revelsMD.revels_rdf import RevelsRDF

            assert RevelsRDF.run_rdf_lambda is run_rdf_lambda

    def test_single_frame_rdf_same_function(self):
        """RevelsRDF.single_frame_rdf should return the same function as revelsMD.rdf.single_frame_rdf."""
        from revelsMD.rdf import single_frame_rdf

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from revelsMD.revels_rdf import RevelsRDF

            assert RevelsRDF.single_frame_rdf is single_frame_rdf
