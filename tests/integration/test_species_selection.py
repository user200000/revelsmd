"""
Loader species-selection tests on real committed data.

Second-species selection (LAMMPS numeric type '2', VASP element symbols)
has historically been bug-prone territory: nothing else end-to-end
exercises get_indices for these species on real files. These tests pin
the exact per-species atom counts of the committed subsets.
"""

import numpy as np
import pytest


@pytest.mark.integration
@pytest.mark.requires_example1
class TestLammpsSpeciesSelection:
    """Exact per-type selections from the Example 1 LAMMPS data file."""

    def test_per_type_counts(self, example1_trajectory):
        """Types '1' and '2' select 2304 and 576 atoms (2880 total)."""
        indices_1 = example1_trajectory.get_indices('1')
        indices_2 = example1_trajectory.get_indices('2')

        assert len(indices_1) == 2304
        assert len(indices_2) == 576
        assert len(indices_1) + len(indices_2) == 2880

    def test_types_are_disjoint(self, example1_trajectory):
        """No atom is selected by both types."""
        indices_1 = example1_trajectory.get_indices('1')
        indices_2 = example1_trajectory.get_indices('2')

        assert len(np.intersect1d(indices_1, indices_2)) == 0


@pytest.mark.integration
@pytest.mark.requires_vasp
class TestVaspSpeciesSelection:
    """Exact per-element selections from the BaSnF4 vasprun.xml subset."""

    def test_per_element_counts(self, vasp_trajectory):
        """Ba, Sn and F select 54, 54 and 216 atoms (324 total)."""
        indices_ba = vasp_trajectory.get_indices('Ba')
        indices_sn = vasp_trajectory.get_indices('Sn')
        indices_f = vasp_trajectory.get_indices('F')

        assert len(indices_ba) == 54
        assert len(indices_sn) == 54
        assert len(indices_f) == 216
        assert len(indices_ba) + len(indices_sn) + len(indices_f) == 324

    def test_elements_are_disjoint(self, vasp_trajectory):
        """No atom is selected by more than one element."""
        indices_ba = vasp_trajectory.get_indices('Ba')
        indices_sn = vasp_trajectory.get_indices('Sn')
        indices_f = vasp_trajectory.get_indices('F')

        combined = np.concatenate([indices_ba, indices_sn, indices_f])
        assert len(np.unique(combined)) == 324


@pytest.mark.integration
@pytest.mark.requires_example4
class TestOneNamePerString:
    """A string names one species: MDAnalysis would read 'Ow Hw1 Hw2' as a
    union of three names, losing the rigid grouping or mixing species."""

    def test_density_string_of_several_names_raises(self, example4_trajectory):
        from revelsMD.density import DensityGrid
        grid = DensityGrid(example4_trajectory, "number", nbins=10)
        with pytest.raises(ValueError, match=r"pass a list: \['Ow', 'Hw1', 'Hw2'\]"):
            grid.accumulate(example4_trajectory, "Ow Hw1 Hw2", rigid=True, stop=1)

    def test_rdf_species_of_several_names_raises(self, example4_trajectory):
        from revelsMD.rdf import RDF
        with pytest.raises(ValueError, match="'Ow Hw1' must be a single name"):
            RDF(example4_trajectory, "Ow Hw1", "Ow")
