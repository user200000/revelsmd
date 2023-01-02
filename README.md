RevelsMD is a code for force based estimation of fundamental statistical mechanical quantities from atomistic simulations. 

The code can calculate both three-dimensional solvent densities for NVT simulations, and radial distribution functions. The code can parse Gromacs, Lammps and Vasp inputs as well as being directly interfaced with via numpy arrays.

Four worked examples are provided for the code:
1.	Lennard jones sphere radial distribution functions (as in https://doi.org/10.1063/5.0053737)
2.	Solvation of an immobilised Lennard jones sphere.
3.	Ion conduction in a fast ion conductor (In development).
4.	Solvation of a static water molecule (as in https://aip.scitation.org/doi/abs/10.1063/1.5111697)

The code is presently in a scientifically complete state, and can be used reliably, however the last steps of development are still in progress to make the code installable from pypi, a conda yaml for an environment in which the code will run is provided, and a mirror of this environment can be created using conda create.
