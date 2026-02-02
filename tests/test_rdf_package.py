"""Tests for the revelsMD.rdf package structure and deprecation warnings."""

import warnings


class TestRDFPackageImports:
    """Test that RDF classes and functions are importable from the package."""

    def test_rdf_class_importable_from_rdf_package(self):
        """RDF class should be importable from revelsMD.rdf."""
        from revelsMD.rdf import RDF

        assert RDF is not None

    def test_compute_rdf_importable_from_rdf_package(self):
        """compute_rdf should be importable from revelsMD.rdf."""
        from revelsMD.rdf import compute_rdf

        assert callable(compute_rdf)


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

    def test_revelsrdf_run_rdf_lambda_emits_deprecation_warning(self):
        """RevelsRDF.run_rdf_lambda should emit deprecation warning."""
        from revelsMD.revels_rdf import RevelsRDF

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = RevelsRDF.run_rdf_lambda
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "run_rdf_lambda" in str(w[0].message)


class TestDeprecatedMethodsReturnCorrectFunctions:
    """Test that deprecated RevelsRDF methods return working functions."""

    def test_run_rdf_returns_callable(self):
        """RevelsRDF.run_rdf should return a callable."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from revelsMD.revels_rdf import RevelsRDF

            assert callable(RevelsRDF.run_rdf)

    def test_run_rdf_lambda_returns_callable(self):
        """RevelsRDF.run_rdf_lambda should return a callable."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from revelsMD.revels_rdf import RevelsRDF

            assert callable(RevelsRDF.run_rdf_lambda)
