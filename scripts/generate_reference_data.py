#!/usr/bin/env python3
"""
Generate reference data for regression tests.

This script computes RDF and density results from known-good trajectories
and saves them as .npz files for use in regression testing. The generated
files under tests/reference_data/ are committed to git: regeneration is a
reviewed change, committed together with the code change that motivated it
and justified in the pull request -- never a local fix for a red test.

Every .npz embeds a `provenance` entry (a JSON string) recording the git
commit, numpy version, revelsMD version and compute backend used to
generate it.

The script exits non-zero if any baseline family could not be generated
because its input data is missing, so that a partial regeneration cannot
masquerade as a complete one. Pass --allow-missing to downgrade missing
input data to a warning (exit 0).

Usage:
    python scripts/generate_reference_data.py [--allow-missing]

Requirements:
    All trajectory inputs are the trimmed subsets committed in tests/data/;
    no external data is needed. A missing input means a broken checkout.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

# Add project root to path
project_root = Path(__file__).parents[1]
sys.path.insert(0, str(project_root))

from revelsMD import __version__ as revelsmd_version
from revelsMD.backends import get_backend
from revelsMD.rdf import compute_rdf

# Canonical committed test data (trimmed trajectory subsets). Full-length
# trajectories are not needed by anything in the repository; they remain
# useful only for ad-hoc high-statistics validation during estimator
# development.
TEST_DATA_DIR = project_root / "tests" / "data"
REFERENCE_DIR = project_root / "tests" / "reference_data"

_PROVENANCE = None


def provenance() -> np.ndarray:
    """Build the provenance entry embedded in every reference file."""
    global _PROVENANCE
    if _PROVENANCE is None:
        try:
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=project_root, capture_output=True, text=True, check=True,
            ).stdout.strip()
        except Exception:
            commit = "unknown"
        _PROVENANCE = np.array(json.dumps({
            "git_commit": commit,
            "numpy_version": np.__version__,
            "revelsmd_version": revelsmd_version,
            "backend": get_backend(),
        }))
    return _PROVENANCE


def save_reference(path: Path, **arrays) -> None:
    """Save a reference .npz with the provenance entry included."""
    np.savez(path, provenance=provenance(), **arrays)


def report_input(path: Path) -> None:
    """Print the resolved path of a trajectory input actually used."""
    print(f"  Input: {Path(path).resolve()}")


def ensure_dir(path):
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)


def generate_lammps_references():
    """Generate reference data from Example 1 LAMMPS trajectory.

    Returns None on success, or a skip-reason string if input data is missing.
    """
    from revelsMD.trajectories import LammpsTrajectory

    dump_file = TEST_DATA_DIR / "example_1_LJ" / "dump.nh.lammps"
    data_file = TEST_DATA_DIR / "example_1_LJ" / "data.fin.nh.data"

    if not dump_file.exists() or not data_file.exists():
        reason = f"Example 1 data not available ({dump_file})"
        print(f"Skipping LAMMPS references: {reason}")
        return reason

    print("Loading Example 1 LAMMPS trajectory...")
    report_input(dump_file)
    report_input(data_file)
    ts = LammpsTrajectory(
        str(dump_file),
        str(data_file),
        units='lj',
        atom_style="id resid type q x y z ix iy iz",
        temperature=1.35,
    )

    output_dir = REFERENCE_DIR / "lammps_example1"
    ensure_dir(output_dir)

    # RDF forward integration (5 frames for speed)
    print("  Computing RDF (forward integration)...")
    rdf_forward = compute_rdf(
        ts, '1', '1',
        delr=0.02, integration='forward', start=0, stop=5
    )
    save_reference(
        output_dir / "rdf_forward.npz",
        r=rdf_forward.r,
        g_r=rdf_forward.g,
        g_count=rdf_forward.g_count,
        frames_used=5,
        delr=0.02,
        temp=1.35,
        species='1'
    )

    # RDF forward integration with frame stride (every 2nd of 10 frames)
    print("  Computing RDF (forward integration, strided)...")
    rdf_strided = compute_rdf(
        ts, '1', '1',
        delr=0.05, integration='forward', start=0, stop=10, period=2
    )
    save_reference(
        output_dir / "rdf_forward_strided.npz",
        r=rdf_strided.r,
        g=rdf_strided.g,
        frames_used=5,
        delr=0.05,
        period=2,
        temp=1.35,
        species='1'
    )

    # RDF backward integration
    print("  Computing RDF (backward integration)...")
    rdf_backward = compute_rdf(
        ts, '1', '1',
        delr=0.02, integration='backward', start=0, stop=5
    )
    save_reference(
        output_dir / "rdf_backward.npz",
        r=rdf_backward.r,
        g_r=rdf_backward.g,
        frames_used=5,
        delr=0.02,
        temp=1.35,
        species='1'
    )

    # RDF lambda combination
    print("  Computing RDF lambda...")
    rdf_lambda = compute_rdf(
        ts, '1', '1',
        delr=0.02, integration='lambda', start=0, stop=5
    )
    save_reference(
        output_dir / "rdf_lambda.npz",
        r=rdf_lambda.r,
        g=rdf_lambda.g,
        lam=rdf_lambda.lam,
        frames_used=5,
        delr=0.02,
        temp=1.35,
        species='1'
    )

    # 3D number density
    print("  Computing 3D number density...")
    from revelsMD.density import DensityGrid
    gs = DensityGrid(ts, 'number', nbins=30)
    gs.accumulate(ts, '1', kernel='triangular', start=0, stop=5)

    save_reference(
        output_dir / "number_density.npz",
        rho=gs.rho_force,
        nbins=30,
        frames_used=5,
        temp=1.35,
        species='1',
        kernel='triangular'
    )

    # 3D number density with the box kernel
    print("  Computing 3D number density (box kernel)...")
    gs_box = DensityGrid(ts, 'number', nbins=30)
    gs_box.accumulate(ts, '1', kernel='box', rigid=False, start=0, stop=5)

    save_reference(
        output_dir / "number_density_box.npz",
        rho=gs_box.rho_force,
        nbins=30,
        frames_used=5,
        temp=1.35,
        species='1',
        kernel='box'
    )

    print(f"  Saved LAMMPS references to {output_dir}")
    return None


def generate_lammps_example2_references():
    """Generate reference data from Example 2 LAMMPS 3D density trajectory.

    Returns None on success, or a skip-reason string if input data is missing.
    """
    from revelsMD.trajectories import LammpsTrajectory
    from revelsMD.density import DensityGrid

    dump_file = TEST_DATA_DIR / "example_2_LJ_3D" / "dump.nh.lammps"
    data_file = TEST_DATA_DIR / "example_2_LJ_3D" / "data.fin.nh.data"

    if not dump_file.exists() or not data_file.exists():
        reason = f"Example 2 data not available ({dump_file})"
        print(f"Skipping LAMMPS Example 2 references: {reason}")
        return reason

    print("Loading Example 2 LAMMPS trajectory...")
    report_input(dump_file)
    report_input(data_file)
    ts = LammpsTrajectory(
        str(dump_file),
        str(data_file),
        units='lj',
        atom_style="id resid type q x y z ix iy iz",
        temperature=1.35,
    )

    output_dir = REFERENCE_DIR / "lammps_example2"
    ensure_dir(output_dir)

    # Lambda-combined number density through a real loader
    # (block_size=2 gives 5 contiguous blocks from 10 frames)
    print("  Computing lambda-combined number density (block_size=2)...")
    gs = DensityGrid(ts, 'number', nbins=30)
    gs.accumulate(
        ts, '2', kernel='triangular', rigid=False, start=0, stop=10,
        compute_lambda=True, blocking='contiguous', block_size=2,
    )

    save_reference(
        output_dir / "number_density_lambda.npz",
        rho_force=gs.rho_force,
        rho_count=gs.rho_count,
        rho_lambda=gs.rho_lambda,
        lambda_weights=gs.lambda_weights,
        lambda_block_size=2,
        nbins=30,
        frames_used=10,
        temp=1.35,
        species='2',
        kernel='triangular'
    )

    print(f"  Saved LAMMPS Example 2 references to {output_dir}")
    return None


def generate_mda_references():
    """Generate reference data from Example 4 MDA/GROMACS trajectory.

    Returns None on success, or a skip-reason string if input data is missing.
    """
    from revelsMD.trajectories import MDATrajectory

    trr_file = TEST_DATA_DIR / "example_4_water" / "prod.trr"
    tpr_file = TEST_DATA_DIR / "example_4_water" / "prod.tpr"

    if not trr_file.exists():
        reason = f"Example 4 data not available ({trr_file})"
        print(f"Skipping MDA references: {reason}")
        return reason

    print("Loading Example 4 MDA trajectory...")
    report_input(trr_file)
    report_input(tpr_file)
    ts = MDATrajectory(str(trr_file), str(tpr_file), temperature=300)

    output_dir = REFERENCE_DIR / "mda_example4"
    ensure_dir(output_dir)

    # RDF lambda
    print("  Computing RDF lambda...")
    rdf_lambda = compute_rdf(
        ts, 'Ow', 'Ow',
        delr=0.1, integration='lambda', start=0, stop=5
    )
    save_reference(
        output_dir / "rdf_lambda_ow.npz",
        r=rdf_lambda.r,
        g=rdf_lambda.g,
        lam=rdf_lambda.lam,
        frames_used=5,
        delr=0.1,
        temp=300,
        species='Ow'
    )

    # Unlike-pair RDF (Ow-Hw1, forward). Pins the pair-enumeration path,
    # which is loader-agnostic, so this single instance suffices.
    print("  Computing unlike-pair RDF (Ow-Hw1, forward)...")
    rdf_unlike = compute_rdf(
        ts, 'Ow', 'Hw1',
        delr=0.1, integration='forward', start=0, stop=5
    )
    save_reference(
        output_dir / "rdf_forward_ow_hw1.npz",
        r=rdf_unlike.r,
        g=rdf_unlike.g,
        frames_used=5,
        delr=0.1,
        temp=300,
        species_a='Ow',
        species_b='Hw1'
    )

    # 3D number density
    print("  Computing 3D number density...")
    from revelsMD.density import DensityGrid
    gs = DensityGrid(ts, 'number', nbins=30)
    gs.accumulate(ts, 'Ow', kernel='triangular', rigid=False, start=0, stop=5)

    save_reference(
        output_dir / "number_density_ow.npz",
        rho=gs.rho_force,
        nbins=30,
        frames_used=5,
        temp=300,
        species='Ow',
        kernel='triangular'
    )

    # Rigid molecule number density
    print("  Computing rigid molecule number density...")
    gs_rigid = DensityGrid(ts, 'number', nbins=30)
    gs_rigid.accumulate(
        ts, ['Ow', 'Hw1', 'Hw2'], kernel='triangular', rigid=True, start=0, stop=5
    )

    save_reference(
        output_dir / "number_density_rigid.npz",
        rho=gs_rigid.rho_force,
        nbins=30,
        frames_used=5,
        temp=300,
        species=['Ow', 'Hw1', 'Hw2'],
        kernel='triangular',
        rigid=True
    )

    # Polarisation density
    print("  Computing polarisation density...")
    gs_pol = DensityGrid(ts, 'polarisation', nbins=30)
    gs_pol.accumulate(
        ts, ['Ow', 'Hw1', 'Hw2'], kernel='triangular', rigid=True, start=0, stop=5
    )

    save_reference(
        output_dir / "polarisation_density.npz",
        rho=gs_pol.rho_force,
        nbins=30,
        frames_used=5,
        temp=300,
        species=['Ow', 'Hw1', 'Hw2'],
        kernel='triangular'
    )

    print(f"  Saved MDA references to {output_dir}")
    return None


def generate_vasp_references():
    """Generate reference data from VASP trajectory (BaSnF4 subset).

    Returns None on success, or a skip-reason string if input data is missing.
    """
    from revelsMD.trajectories import VaspTrajectory

    # Use subset from Example 3 BaSnF4
    vasprun_file = TEST_DATA_DIR / "example_3_vasp" / "vasprun.xml"

    if not vasprun_file.exists():
        reason = f"example_3_vasp data not available ({vasprun_file})"
        print(f"Skipping VASP references: {reason}")
        return reason

    print("Loading VASP trajectory (BaSnF4 subset)...")
    report_input(vasprun_file)
    ts = VaspTrajectory(str(vasprun_file), temperature=600)

    output_dir = REFERENCE_DIR / "vasp_example3"
    ensure_dir(output_dir)

    # RDF lambda for F-F (BaSnF4 contains F atoms)
    print("  Computing F-F RDF lambda...")
    rdf_lambda = compute_rdf(
        ts, 'F', 'F',
        delr=0.1, integration='lambda', start=0, stop=10
    )
    save_reference(
        output_dir / "rdf_lambda_f_f.npz",
        r=rdf_lambda.r,
        g=rdf_lambda.g,
        lam=rdf_lambda.lam,
        frames_used=10,
        delr=0.1,
        temp=600,
        species='F'
    )

    # 3D number density for F
    print("  Computing F number density...")
    from revelsMD.density import DensityGrid
    gs = DensityGrid(ts, 'number', nbins=30)
    gs.accumulate(ts, 'F', kernel='triangular', rigid=False, start=0, stop=10)

    save_reference(
        output_dir / "number_density_f.npz",
        rho=gs.rho_force,
        nbins=30,
        frames_used=10,
        temp=600,
        species='F',
        kernel='triangular'
    )

    print(f"  Saved VASP references to {output_dir}")
    return None


def generate_synthetic_references():
    """Generate reference data from synthetic NumPy trajectories.

    Returns None on success (synthetic data needs no external inputs).
    """
    from revelsMD.trajectories import NumpyTrajectory

    output_dir = REFERENCE_DIR / "synthetic"
    ensure_dir(output_dir)

    # Uniform gas trajectory - must match conftest.py fixture parameters!
    print("Generating synthetic trajectory references...")
    np.random.seed(42)
    n_atoms = 500
    n_frames = 50  # Must match uniform_gas_trajectory fixture
    box = 10.0

    positions = np.random.uniform(0, box, (n_frames, n_atoms, 3))
    forces = np.random.randn(n_frames, n_atoms, 3) * 0.1
    species = ['1'] * n_atoms

    ts = NumpyTrajectory(
        positions, forces, box, box, box, species, units='lj', temperature=1.0
    )

    # RDF for uniform gas (use all frames)
    print("  Computing uniform gas RDF...")
    rdf = compute_rdf(
        ts, '1', '1',
        delr=0.1, integration='lambda', start=0, stop=None
    )
    save_reference(
        output_dir / "uniform_gas_rdf.npz",
        r=rdf.r,
        g=rdf.g,
        lam=rdf.lam,
        n_atoms=n_atoms,
        n_frames=n_frames,
        box=box,
        seed=42
    )

    # Number density for uniform gas
    print("  Computing uniform gas density...")
    from revelsMD.density import DensityGrid
    gs = DensityGrid(ts, 'number', nbins=30)
    gs.accumulate(ts, '1', kernel='triangular', rigid=False)

    # Lambda-combined density on a separate grid, so the stored rho comes
    # from exactly the same accumulation path as before (block accumulation
    # can differ in floating-point summation order).
    print("  Computing uniform gas lambda density (block_size=10)...")
    gs_lambda = DensityGrid(ts, 'number', nbins=30)
    gs_lambda.accumulate(
        ts, '1', kernel='triangular', rigid=False,
        compute_lambda=True, blocking='contiguous', block_size=10,
    )

    save_reference(
        output_dir / "uniform_gas_density.npz",
        rho=gs.rho_force,
        rho_lambda=gs_lambda.rho_lambda,
        lambda_weights=gs_lambda.lambda_weights,
        lambda_block_size=10,
        nbins=30,
        n_atoms=n_atoms,
        n_frames=n_frames,
        box=box,
        seed=42
    )

    print(f"  Saved synthetic references to {output_dir}")
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Generate regression-test reference data."
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="exit 0 even if some baseline families were skipped because "
             "their input data is missing (default: exit 1)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Generating reference data for regression tests")
    print("=" * 60)
    print()

    families = [
        ("lammps_example1", generate_lammps_references),
        ("lammps_example2", generate_lammps_example2_references),
        ("mda_example4", generate_mda_references),
        ("vasp_example3", generate_vasp_references),
        ("synthetic", generate_synthetic_references),
    ]

    skip_reasons = {}
    for name, generate in families:
        skip_reasons[name] = generate()
        print()

    print("=" * 60)
    print("Summary")
    print("-" * 60)
    for name, _ in families:
        reason = skip_reasons[name]
        if reason is None:
            print(f"  {name:<18} generated")
        else:
            print(f"  {name:<18} SKIPPED: {reason}")
    print("-" * 60)
    print(f"Output directory: {REFERENCE_DIR}")
    print("=" * 60)

    skipped = [name for name, reason in skip_reasons.items() if reason]
    if skipped:
        if args.allow_missing:
            print(
                f"WARNING: skipped families ({', '.join(skipped)}); "
                f"exiting 0 because --allow-missing was passed."
            )
        else:
            print(
                f"ERROR: skipped families ({', '.join(skipped)}); "
                f"the committed baselines were NOT fully regenerated. "
                f"Pass --allow-missing to tolerate this."
            )
            sys.exit(1)


if __name__ == "__main__":
    main()
