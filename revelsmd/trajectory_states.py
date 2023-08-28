import numpy as np
from tqdm import tqdm
import MDAnalysis as MD
from lxml import etree # type: ignore
from typing import List, Union, Optional, Any, Dict
from pymatgen.core import Structure, Lattice
from pymatgen.io.ase import AseAtomsAdaptor
from ase.io.cube import write_cube
from revelsmd.revels_tools.lammps_parser import first_read
from revelsmd.revels_tools.vasp_parser import *

class MDATrajectoryState:
    def  __init__(self,trajectory_file,topology_file):
        """
        Trajectory state object storing all the details of the molecular dynamics simulation we wish to analyze:
        init args:
        variety(str): lowercase name of the code of interest
        trajectory_file(str or list): filename of the trajectory.
        topology_file(str): filename of the topology file, required for mda and lammps inputs (not required for vasp)
        generated quantities:
        boltzmann constant in the required units
        """
        self.variety='mda'
        self.trajectory_file=trajectory_file
        self.topology_file=topology_file
        mdanalysis_universe = MD.Universe(topology_file,trajectory_file)
        self.box_x=mdanalysis_universe.dimensions[0]
        self.box_y=mdanalysis_universe.dimensions[1]
        self.box_z=mdanalysis_universe.dimensions[2]
        self.mdanalysis_universe=mdanalysis_universe
        self.frames=len(mdanalysis_universe.trajectory)
        self.charge_and_mass=True
        self.units='mda'

    def get_indicies(self,atype):
        return np.array(self.mdanalysis_universe.select_atoms('name '+str(atype)).ids)
    def get_charges(self,atype):
        return np.array(self.mdanalysis_universe.select_atoms('name '+str(atype)).charges)
    def get_masses(self,atype):
        return np.array(self.mdanalysis_universe.select_atoms('name '+str(atype)).masses)


class NumpyTrajectoryState:
    def  __init__(self,positions,forces,box_x,box_y,box_z,species_list,units='real',charge_list=False,mass_list=False):
        """
        Trajectory state object storing all the details of the molecular dynamics simulation we wish to analyze:
        init args:
        variety(str): lowercase name of the code of interest
        trajectory_file(str or list): filename of the trajectory.
        topology_file(str): filename of the topology file, required for mda and lammps inputs (not required for vasp)
        generated quantities:
        boltzmann constant in the required units
        """

        if np.shape(positions)!= np.shape(forces):
            print("force and position arrays are incomensurate")
        elif np.shape(positions)[1] != len(species_list):
            print("species list and trajectory arrays are incomensurate")
        self.variety = 'numpy'
        self.positions = positions
        self.forces = forces
        self.species_string = species_list
        self.box_x = box_x
        self.box_y = box_y
        self.box_z = box_z
        self.units = units
        self.frames=np.shape(positions)[0]
        self.charge_and_mass=True
        if charge_list == False:
            self.charge_and_mass=False
        else:
                self.charge_list=charge_list
        if mass_list == False:
            self.charge_and_mass=False
        else:
            self.mass_list=mass_list
                
    def get_indicies(self,atype):
        return np.where(np.array(self.species_string)==atype)[0]
                    

class LammpsTrajectoryState:
    def  __init__(self,trajectory_file,topology_file=False,units='real', atom_style="full",charge_and_mass=True):
        """
        Trajectory state object storing all the details of the molecular dynamics simulation we wish to analyze:
        init args:
        variety(str): lowercase name of the code of interest
        trajectory_file(str or list): filename of the trajectory.
        topology_file(str): filename of the topology file, required for mda and lammps inputs (not required for vasp)
        generated quantities:
        boltzmann constant in the required units
        """
        self.variety='lammps'
        self.trajectory_file=trajectory_file
        self.topology_file=topology_file
        self.frames,self.num_ats,self.dic,self.header_length,self.dimgrid=first_read(trajectory_file)
        mdanalysis_universe = MD.Universe(topology_file, atom_style=atom_style)
        self.box_x=mdanalysis_universe.dimensions[0]
        self.box_y=mdanalysis_universe.dimensions[1]
        self.box_z=mdanalysis_universe.dimensions[2]
        self.mdanalysis_universe=mdanalysis_universe
        self.charge_and_mass=charge_and_mass
        self.units = units

    def get_indicies(self,atype):
        return self.mdanalysis_universe.select_atoms('type '+str(atype)).ids-1
    def get_charges(self,atype):
        return self.mdanalysis_universe.select_atoms('type '+str(atype)).charges
    def get_masses(self,atype):
        return self.mdanalysis_universe.select_atoms('type '+str(atype)).masses

class VaspTrajectoryState:
    def  __init__(self,trajectory_file):
        """
        Trajectory state object storing all the details of the molecular dynamics simulation we wish to analyze:
        init args:
        variety(str): lowercase name of the code of interest
        trajectory_file(str or list): filename of the trajectory.
        topology_file(str): filename of the topology file, required for mda and lammps inputs (not required for vasp)
        generated quantities:
        boltzmann constant in the required units
        """
        if type(trajectory_file)==list:
            self.trajectory_file=trajectory_file
            self.Vasprun=Vasprun(trajectory_file[0])
            self.Vasprun.start=self.Vasprun.structures[0]
            self.frames = len(self.Vasprun.structures)
            if np.sum(np.array(self.Vasprun.start.lattice.angles)==90.0)==3:
                self.box_x=self.Vasprun.start.lattice.matrix[0,0]
                self.box_y=self.Vasprun.start.lattice.matrix[1,1]
                self.box_z=self.Vasprun.start.lattice.matrix[2,2]
            else:
                print("cell not cubic/orthorombic, code presently operates solely in this case do not continue unless deviation is miniscule")
                self.box_x=self.Vasprun.start.lattice.matrix[0,0]
                self.box_y=self.Vasprun.start.lattice.matrix[1,1]
                self.box_z=self.Vasprun.start.lattice.matrix[2,2]
            self.units = 'metal'
            self.charge_and_mass=False
            self.variety='vasp'
            self.positions=self.Vasprun.cart_coords
            self.forces=self.Vasprun.forces
            start=self.Vasprun.structures[0]
            
            for item in trajectory_file[1:]:
                self.Vasprun=Vasprun(item)
                self.frames+=len(self.Vasprun.structures)
                self.positions=np.append(self.positions,self.Vasprun.cart_coords,axis=0)
                self.forces=np.append(self.forces,self.Vasprun.forces,axis=0)
            self.Vasprun.start=start

        else:
            self.trajectory_file=trajectory_file
            self.Vasprun=Vasprun(trajectory_file)
            self.Vasprun.start=self.Vasprun.structures[0]
            self.frames = len(self.Vasprun.structures)
            self.box_x=self.Vasprun.start.lattice.matrix[0,0]
            self.box_y=self.Vasprun.start.lattice.matrix[1,1]
            self.box_z=self.Vasprun.start.lattice.matrix[2,2]
            if np.sum(np.array(self.Vasprun.start.lattice.angles)==90.0)==3:
                self.box_x=self.Vasprun.start.lattice.matrix[0,0]
                self.box_y=self.Vasprun.start.lattice.matrix[1,1]
                self.box_z=self.Vasprun.start.lattice.matrix[2,2]
            else:
                print("cell not cubic/orthorombic, code presently operates solely in this case do not continue unless deviation is miniscule")
            self.units = 'metal'
            self.charge_and_mass=False
            self.variety='vasp'
            self.positions=self.Vasprun.cart_coords
            self.forces=self.Vasprun.forces
            

    def get_indicies(self,atype):
        return self.Vasprun.start.indices_from_symbol(atype)


        
