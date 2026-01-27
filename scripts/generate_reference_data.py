#!/usr/bin/env python3
"""
Generate reference data for regression tests.

This script computes RDF and density results from known-good trajectories
and saves them as .npz files for use in regression testing. Run this script
once to establish a baseline, then the regression tests will compare future
results against these stored references.

Usage:
    python scripts/generate_reference_data.py

Requirements:
    - Example 1 data in examples/example_1_LJ/
    - Example 4 subset in tests/test_data/example_4_subset/
    - VASP subset in tests/test_data/example_3_vasp_subset/
"""

import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parents[1]
sys.path.insert(0, str(project_root))

from revelsMD.revels_rdf import RevelsRDF
from revelsMD.revels_3D import Revels3D

EXAMPLES_DIR = project_root / "examples"
TEST_DATA_DIR = project_root / "tests" / "test_data"
REFERENCE_DIR = project_root / "tests" / "reference_data"


def ensure_dir(path):
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)


def generate_lammps_references():
    """Generate reference data from Example 1 LAMMPS trajectory."""
    from revelsMD.trajectory_states import LammpsTrajectoryState

    dump_file = EXAMPLES_DIR / "example_1_LJ" / "dump.nh.lammps"
    data_file = EXAMPLES_DIR / "example_1_LJ" / "data.fin.nh.data"

    if not dump_file.exists():
        print("Skipping LAMMPS references: Example 1 data not available")
        return

    print("Loading Example 1 LAMMPS trajectory...")
    ts = LammpsTrajectoryState(
        str(dump_file),
        str(data_file),
        units='lj',
        atom_style="id resid type q x y z ix iy iz",
        charge_and_mass=False,
    )

    output_dir = REFERENCE_DIR / "lammps_example1"
    ensure_dir(output_dir)

    # RDF forward integration (5 frames for speed)
    print("  Computing RDF (forward integration)...")
    rdf_forward = RevelsRDF.run_rdf(
        ts, '1', '1', temp=1.35,
        delr=0.02, from_zero=True, start=0, stop=5
    )
    np.savez(
        output_dir / "rdf_forward.npz",
        r=rdf_forward[0],
        g_r=rdf_forward[1],
        frames_used=5,
        delr=0.02,
        temp=1.35,
        species='1'
    )

    # RDF backward integration
    print("  Computing RDF (backward integration)...")
    rdf_backward = RevelsRDF.run_rdf(
        ts, '1', '1', temp=1.35,
        delr=0.02, from_zero=False, start=0, stop=5
    )
    np.savez(
        output_dir / "rdf_backward.npz",
        r=rdf_backward[0],
        g_r=rdf_backward[1],
        frames_used=5,
        delr=0.02,
        temp=1.35,
        species='1'
    )

    # RDF lambda combination
    print("  Computing RDF lambda...")
    rdf_lambda = RevelsRDF.run_rdf_lambda(
        ts, '1', '1', temp=1.35,
        delr=0.02, start=0, stop=5
    )
    np.savez(
        output_dir / "rdf_lambda.npz",
        data=rdf_lambda,
        frames_used=5,
        delr=0.02,
        temp=1.35,
        species='1'
    )

    # 3D number density
    print("  Computing 3D number density...")
    gs = Revels3D.GridState(ts, 'number', nbins=30, temperature=1.35)
    gs.make_force_grid(ts, '1', kernel='triangular', rigid=False, start=0, stop=5)
    gs.get_real_density()
    np.savez(
        output_dir / "number_density.npz",
        rho=gs.rho,
        nbins=30,
        frames_used=5,
        temp=1.35,
        species='1',
        kernel='triangular'
    )

    print(f"  Saved LAMMPS references to {output_dir}")


def generate_mda_references():
    """Generate reference data from Example 4 MDA/GROMACS trajectory."""
    from revelsMD.trajectory_states import MDATrajectoryState

    # Use subset trajectory
    subset_dir = TEST_DATA_DIR / "example_4_subset"
    trr_file = subset_dir / "prod_100frames.trr"
    tpr_file = subset_dir / "prod.tpr"

    # Fall back to full trajectory
    if not trr_file.exists():
        trr_file = EXAMPLES_DIR / "example_4_rigid_water" / "prod.trr"
        tpr_file = EXAMPLES_DIR / "example_4_rigid_water" / "prod.tpr"

    if not trr_file.exists():
        print("Skipping MDA references: Example 4 data not available")
        return

    print("Loading Example 4 MDA trajectory...")
    ts = MDATrajectoryState(str(trr_file), str(tpr_file))

    output_dir = REFERENCE_DIR / "mda_example4"
    ensure_dir(output_dir)

    # RDF lambda (run_rdf has a bug with MDA, but run_rdf_lambda works)
    print("  Computing RDF lambda...")
    rdf_lambda = RevelsRDF.run_rdf_lambda(
        ts, 'Ow', 'Ow', temp=300,
        delr=0.1, start=0, stop=5
    )
    np.savez(
        output_dir / "rdf_lambda_ow.npz",
        data=rdf_lambda,
        frames_used=5,
        delr=0.1,
        temp=300,
        species='Ow'
    )

    # 3D number density
    print("  Computing 3D number density...")
    gs = Revels3D.GridState(ts, 'number', nbins=30, temperature=300)
    gs.make_force_grid(ts, 'Ow', kernel='triangular', rigid=False, start=0, stop=5)
    gs.get_real_density()
    np.savez(
        output_dir / "number_density_ow.npz",
        rho=gs.rho,
        nbins=30,
        frames_used=5,
        temp=300,
        species='Ow',
        kernel='triangular'
    )

    # Rigid molecule number density
    print("  Computing rigid molecule number density...")
    gs_rigid = Revels3D.GridState(ts, 'number', nbins=30, temperature=300)
    gs_rigid.make_force_grid(
        ts, ['Ow', 'Hw1', 'Hw2'], kernel='triangular', rigid=True, start=0, stop=5
    )
    gs_rigid.get_real_density()
    np.savez(
        output_dir / "number_density_rigid.npz",
        rho=gs_rigid.rho,
        nbins=30,
        frames_used=5,
        temp=300,
        species=['Ow', 'Hw1', 'Hw2'],
        kernel='triangular',
        rigid=True
    )

    # Polarisation density
    print("  Computing polarisation density...")
    gs_pol = Revels3D.GridState(ts, 'polarisation', nbins=30, temperature=300)
    gs_pol.make_force_grid(
        ts, ['Ow', 'Hw1', 'Hw2'], kernel='triangular', rigid=True, start=0, stop=5
    )
    gs_pol.get_real_density()
    np.savez(
        output_dir / "polarisation_density.npz",
        rho=gs_pol.rho,
        nbins=30,
        frames_used=5,
        temp=300,
        species=['Ow', 'Hw1', 'Hw2'],
        kernel='triangular'
    )

    print(f"  Saved MDA references to {output_dir}")


def generate_vasp_references():
    """Generate reference data from VASP trajectory (BaSnF4 subset)."""
    from revelsMD.trajectory_states import VaspTrajectoryState

    # Use subset from Example 3 BaSnF4
    vasprun_file = TEST_DATA_DIR / "example_3_vasp_subset" / "vasprun.xml"

    if not vasprun_file.exists():
        print("Skipping VASP references: example_3_vasp_subset not available")
        return

    print("Loading VASP trajectory (BaSnF4 subset)...")
    ts = VaspTrajectoryState(str(vasprun_file))

    output_dir = REFERENCE_DIR / "vasp_example3"
    ensure_dir(output_dir)

    # RDF lambda for F-F (BaSnF4 contains F atoms)
    print("  Computing F-F RDF lambda...")
    rdf_lambda = RevelsRDF.run_rdf_lambda(
        ts, 'F', 'F', temp=600,
        delr=0.1, start=0, stop=10
    )
    np.savez(
        output_dir / "rdf_lambda_f_f.npz",
        data=rdf_lambda,
        frames_used=10,
        delr=0.1,
        temp=600,
        species='F'
    )

    # 3D number density for F
    print("  Computing F number density...")
    gs = Revels3D.GridState(ts, 'number', nbins=30, temperature=600)
    gs.make_force_grid(ts, 'F', kernel='triangular', rigid=False, start=0, stop=10)
    gs.get_real_density()
    np.savez(
        output_dir / "number_density_f.npz",
        rho=gs.rho,
        nbins=30,
        frames_used=10,
        temp=600,
        species='F',
        kernel='triangular'
    )

    print(f"  Saved VASP references to {output_dir}")


def generate_synthetic_references():
    """Generate reference data from synthetic NumPy trajectories."""
    from revelsMD.trajectory_states import NumpyTrajectoryState

    output_dir = REFERENCE_DIR / "synthetic"
    ensure_dir(output_dir)

    # Uniform gas trajectory
    print("Generating synthetic trajectory references...")
    np.random.seed(42)
    n_atoms = 500
    n_frames = 10
    box = 10.0

    positions = np.random.uniform(0, box, (n_frames, n_atoms, 3))
    forces = np.random.randn(n_frames, n_atoms, 3) * 0.1
    species = ['1'] * n_atoms

    ts = NumpyTrajectoryState(
        positions, forces, box, box, box, species, units='lj'
    )

    # RDF for uniform gas
    print("  Computing uniform gas RDF...")
    rdf = RevelsRDF.run_rdf_lambda(
        ts, '1', '1', temp=1.0,
        delr=0.1, start=0, stop=-1
    )
    np.savez(
        output_dir / "uniform_gas_rdf.npz",
        data=rdf,
        n_atoms=n_atoms,
        n_frames=n_frames,
        box=box,
        seed=42
    )

    # Number density for uniform gas
    print("  Computing uniform gas density...")
    gs = Revels3D.GridState(ts, 'number', nbins=30, temperature=1.0)
    gs.make_force_grid(ts, '1', kernel='triangular', rigid=False)
    gs.get_real_density()
    np.savez(
        output_dir / "uniform_gas_density.npz",
        rho=gs.rho,
        nbins=30,
        n_atoms=n_atoms,
        n_frames=n_frames,
        box=box,
        seed=42
    )

    print(f"  Saved synthetic references to {output_dir}")


def main():
    print("=" * 60)
    print("Generating reference data for regression tests")
    print("=" * 60)
    print()

    generate_lammps_references()
    print()

    generate_mda_references()
    print()

    generate_vasp_references()
    print()

    generate_synthetic_references()
    print()

    print("=" * 60)
    print("Reference data generation complete")
    print(f"Output directory: {REFERENCE_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
