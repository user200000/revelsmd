#!/usr/bin/env python3
"""
Create a subset of a VASP vasprun.xml file with only the first N frames.

This extracts the XML header, atominfo, and first N <calculation> blocks
to create a smaller file suitable for testing.

Usage:
    python scripts/create_vasp_subset.py [n_frames]

Default: 50 frames
"""

import sys
import re
from pathlib import Path

def create_vasp_subset(input_path, output_path, n_frames=50):
    """Extract first n_frames from a vasprun.xml file."""

    print(f"Reading {input_path}...")
    with open(input_path, 'r') as f:
        content = f.read()

    # Find the end of the header section (before first <calculation>)
    first_calc = content.find('<calculation>')
    if first_calc == -1:
        raise ValueError("No <calculation> blocks found in file")

    header = content[:first_calc]

    # Find all calculation blocks
    calc_pattern = re.compile(r'<calculation>.*?</calculation>', re.DOTALL)
    calculations = calc_pattern.findall(content)

    print(f"Found {len(calculations)} calculation blocks")

    if len(calculations) < n_frames:
        print(f"Warning: Only {len(calculations)} frames available, using all")
        n_frames = len(calculations)

    # Find the closing tags after the last calculation
    last_calc_end = content.rfind('</calculation>') + len('</calculation>')
    footer = content[last_calc_end:]

    # Build the subset file
    subset = header
    for i in range(n_frames):
        subset += calculations[i] + '\n'
    subset += footer

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(subset)

    input_size = Path(input_path).stat().st_size / (1024 * 1024)
    output_size = output_path.stat().st_size / (1024 * 1024)

    print(f"Created {output_path}")
    print(f"  Input:  {input_size:.1f} MB ({len(calculations)} frames)")
    print(f"  Output: {output_size:.1f} MB ({n_frames} frames)")
    print(f"  Reduction: {100 * (1 - output_size/input_size):.1f}%")


def main():
    n_frames = int(sys.argv[1]) if len(sys.argv) > 1 else 50

    project_root = Path(__file__).parents[1]
    input_path = project_root / "examples" / "example_3_BaSnF4" / "r1" / "vasprun.xml"
    output_path = project_root / "tests" / "test_data" / "example_3_vasp_subset" / "vasprun.xml"

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    create_vasp_subset(input_path, output_path, n_frames)


if __name__ == "__main__":
    main()
