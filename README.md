## RevelsMD
RevelsMD is a code for force based estimation of equilibrium structural features from atomistic simulations, with greatly reduced variance

The code can calculate both three-dimensional solvent densities for NVT simulations, and radial distribution functions. The code can parse Gromacs, Lammps and Vasp inputs as well as being directly interfaced with via numpy arrays.

Four worked examples are provided for the code:
1.	Lennard jones sphere radial distribution functions (as in https://doi.org/10.1063/5.0053737)
2.	Solvation of an immobilised Lennard jones sphere in a solvent of identicle Lennard Jones spheres.
3.	Ion conduction in a fast ion conductor (In development).
4.	Solvation of a static water molecule (as in https://aip.scitation.org/doi/abs/10.1063/1.5111697)

The code is currently in a pre release state and can be installed after cloning using pip locally:
```
pip install .
```
The code may at this stage be best installed in its own conda or pyenv environment.
