import numpy as np
from tqdm import tqdm
import MDAnalysis as MD
import scipy.constants as constants
from lxml import etree # type: ignore
from typing import List, Union, Optional, Any, Dict
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from ase.io.cube import write_cube
import copy



class TrajectoryStates:
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
            self.frames,self.num_ats,self.dic,self.header_length,self.dimgrid=RevelsMDTools.LammpsParser.first_read(trajectory_file)
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
                self.Vasprun=RevelsMDTools.VaspParser.Vasprun(trajectory_file[0])
                self.Vasprun.start=self.Vasprun.structures[0]
                self.frames = len(self.Vasprun.structures)
                self.box_x=self.Vasprun.start.lattice.matrix[0,0]
                self.box_y=self.Vasprun.start.lattice.matrix[1,1]
                self.box_z=self.Vasprun.start.lattice.matrix[2,2]
                if np.sum(np.array(self.Vasprun.start.lattice.angles)==90.0)!=3:
                    print("Error all angles of box must be right angles") 
                self.units = 'metal'
                self.charge_and_mass=False
                self.variety='vasp'
                self.positions=self.Vasprun.cart_coords
                self.forces=self.Vasprun.forces
                start=self.Vasprun.structures[0]
                
                for item in trajectory_file[1:]:
                    self.Vasprun=RevelsMDTools.VaspParser.Vasprun(item)
                    self.frames+=len(self.Vasprun.structures)
                    self.positions=np.append(self.positions,self.Vasprun.cart_coords,axis=0)
                    self.forces=np.append(self.forces,self.Vasprun.forces,axis=0)
                self.Vasprun.start=start

            else:
                self.trajectory_file=trajectory_file
                self.Vasprun=RevelsMDTools.VaspParser.Vasprun(trajectory_file)
                self.Vasprun.start=self.Vasprun.structures[0]
                self.frames = len(self.Vasprun.structures)
                self.box_x=self.Vasprun.start.lattice.matrix[0,0]
                self.box_y=self.Vasprun.start.lattice.matrix[1,1]
                self.box_z=self.Vasprun.start.lattice.matrix[2,2]
                if np.sum(np.array(self.Vasprun.start.lattice.angles)==90.0)!=3:
                    print("Error all angles of box must be right angles") 
                self.units = 'metal'
                self.charge_and_mass=False
                self.variety='vasp'
                self.positions=self.Vasprun.cart_coords
                self.forces=self.Vasprun.forces
                

        def get_indicies(self,atype):
            return self.Vasprun.start.indices_from_symbol(atype)


            
class RevelsMDTools:
    class ConversionFactors:
        def generate_boltzmann(units):
            """
            A function which when fed the units desired will return the boltzmann constant in said units
            args:
            units(str): name of unit system which we wish to generate the value of the boltzmann constant in
            returns:
            boltzmann constant in the required units
            """
            if units == 'lj':
                return 1
            elif units == 'real':
                return constants.physical_constants['molar gas constant'][0] / constants.calorie / 1000
            elif units == 'metal':
                return constants.physical_constants['Boltzmann constant in eV/K'][0]
            elif units == 'mda':
                return constants.physical_constants['molar gas constant'][0] / 1000
    class LammpsParser:
        def first_read(dumpFile): #This reads the information from the file header and lets us work out what's going on
            """
            A function which performs a first read of a lammps custom dump file in order to discover the values for each collumn
            and the number of atoms in the trajector, the number of frames and the size of the header.
            args:
            dumpFile: an open unparsed file in read mode
            returns:
            frames(int): The number of frames in the trajectory file
            num_ats(int): The number of atoms recorded in the trajectory file
            dic(list of strings): The values recorded in each sucessive collumn as named in the header
            header_length(int): length of the header placed every frame prior to the positions
            dimgrid(np.array(3,2)): returns the box size of the first frame
            """
            header_length=0 #Buffer for the length of the header
            dimgrid=np.zeros((3,2))
            closer=0 #Binary buffer tell us whever or not we've reached the last header line
            f= open(dumpFile,'r') #File opens here
            num_ats=0 #set atom number buffer
            while closer==0: #Using out boolian type buffer to keep process going till we reach the first line of atoms
                currentString= f.readline() #Read a line as a string
                if currentString[6:11]=="ATOMS": # Search for the header of the last line before the ascii componant of the first frame 
                    dic=currentString.split() #we want to grab this line to allow for the automation of the later stage
                    closer=1 # end the code by setting to one the binary buffer
                if currentString=='ITEM: NUMBER OF ATOMS\n': #we need to keep a look out for the number of atoms in the input file
                    header_length+=1 #were going to get another line in a minute so lets iterrate the headerline reader
                    currentString= f.readline() # read in another line in order to ge the number of atoms
                    num_ats=int(currentString) # rip number of atoms in dump from file
                header_length+=1 # increase the header counter for each headerline read
                if currentString=='ITEM: BOX BOUNDS pp pp pp\n': #we need to keep a look out for the number of atoms in the input file
                    header_length+=3 #were going to get another line in a minute so lets iterrate the headerline reader
                    currentString= f.readline() # read in another line in order to ge the number of atoms
                    dimgrid[0,:]= np.array(currentString.split())
                    currentString= f.readline() # read in another line in order to ge the number of atoms
                    dimgrid[1,:]= np.array(currentString.split())
                    currentString= f.readline() # read in another line in order to ge the number of atoms
                    dimgrid[2,:]= np.array(currentString.split())
            f.close() # close the file
            f=open(dumpFile,'r') # Open the dumpfile again
            numLines = sum(1 for line in open(dumpFile)) # count number of lines
            frames=numLines/float(num_ats+header_length) # calculate the number of frames
            if frames%1!=0: # check that the number of calculate frames is integer. this will not be correct if the header length varies or the dump stopped writing mid step
                print ("ERROR file incomplete or header unharmonious WARNING WARNING!!! Parser will fail at EOF") # error for the frame length fail
            return frames,num_ats,dic,header_length,dimgrid # send out stuff we need


        def get_a_frame(f,num_ats,header_length,strngdex): # this a single frame parser can be interdigtated into the text. The file needs to be open
            """
            A function which gets a single frame of information from a lammps custom dumps.
            args:
            f: an open file in read mode in the process of being parsed
            num_ats(int): The number of atoms recorded in the trajectory file
            header_length(int): length of the header placed every frame prior to the positions
            strgdex(np.array): collums to be returned in the array
            returns:
            vars_trest(np.array): a table of atoms from the trajectory with collums define by the strindex
            """
            vars_trest=np.zeros((num_ats,len(strngdex))) # create a storage array
            for line in range(header_length): # read the 
                currentString= f.readline()
            for line in range(num_ats):
                currentString= f.readline()
                currentString=currentString.split()
                col=0
                for k in strngdex:
                    vars_trest[line,col]=float(currentString[k])
                    col+=1
            return  vars_trest

        def define_strngdex(our_string,dic):
            """
            A function whithin the parser which assocates the requested constants in a lammps custom dump with the collumns in which they are requested:
            our_string(str): a list of strings which lable the collumns wanted from the lammps custom dump file
            dic(TrajectoryState.dic): a trajectory state dic object containing the relevant information regarding the trajectory
            returns:
            strngdex(np.array): an numpy arry of collumn headings in order in the list in our_string
            """
            strngdex=[None]*len(our_string)
            eledex=0
            for ele in our_string:
                strngdex[eledex]=int((dic.index(ele))-2)
                eledex+=1
            return strngdex

        def frame_skip(f,num_ats,num_skip,header_length):
            """
            A function which skips a certain number of frames from an open file in the process of being read
            args:
            f: an open file in read mode in the process of being parsed
            num_ats(int): The number of atoms recorded in the trajectory file
            num_skip(int): a function which skips a certain number of frames
            header_length(int): length of the header placed every frame prior to the positions
            """
            for toSkip in range(num_skip*(num_ats+header_length)):
                ignore=f.readline()


    class VaspParser:
        def parse_varray(varray: etree.Element) -> Union[List[List[float]], 
                                                 List[List[int]],
                                                    List[List[bool]]]:
                """Parse <varray> data.
                Args:
                    varray (etree.Element): xml <varray> element.
                Returns:
                    (list(list): A nested list of either float, int, or bool.
                """
                m: Union[List[List[int]], List[List[float]], List[List[bool]]]
                varray_type = varray.get("type", None)
                v_list = [v.text.split() for v in varray.findall("v")] 
                if varray_type == 'int':
                    m = [[int(number) for number in v] for v in v_list]
                elif varray_type == 'logical':
                    m = [[i == "T" for i in v] for v in v_list]
                else:
                    m = [[float(number) for number in v] for v in v_list]
                return m
        def parse_structure(structure: etree.Element) -> Dict[str, Any]:
                """Parse <structure> data..
                Args:
                    structure (etree.Element): xml <structure> element.
                Returns:
                    (dict): Dictionary of structure data:
                        `lattice`: cell matrix (list(list(float))..
                        `frac_coords`: atom fractional coordinates (list(list(float)).
                        `selective_dynamics`: selective dynamics (list(bool)|None).
                """
                latt = RevelsMDTools.VaspParser.parse_varray(structure.find("crystal").find("varray"))
                pos = RevelsMDTools.VaspParser.parse_varray(structure.find("varray"))
                sdyn = structure.find("varray/[@name='selective']")
                if sdyn:
                    sdyn = RevelsMDTools.VaspParser.parse_varray(sdyn)
                structure_dict = {'lattice': latt,
                                'frac_coords': pos,
                                'selective_dynamics': sdyn}
                return structure_dict

        def structure_from_structure_data(lattice: List[List[float]],
                                        atom_names: List[str],
                                        frac_coords: List[List[float]]) -> Structure:
            """Generate a pymatgen Structure.
            Args:
                lattice (list(list(float)): 3x3 cell matrix.
                atom_names (list(str)): list of atom name strings.
                frac_coords (list(list(float): Nx3 list of fractional coordinates.
            Returns:
                (pymatgen.Structure)
            """
            structure = Structure(lattice=lattice,
                                species=atom_names,
                                coords=frac_coords,
                                coords_are_cartesian=False)
            return structure

        class Vasprun:
            """
            Object for parsing vasprun.xml data.
            args:
                atom_names (list(str)): List of atom name strings.
                structures (list(pymatgen.Structure): List of structures as pymatgen Structure objects.
                frac_coords (np.array): timesteps x atoms x 3 numpy array of fractional coordinates.
                cart_coords (np.array): timesteps x atoms x 3 numpy array of cartesian coordinates.
                forces (:obj:`np.array`, optional): timesteps x atoms x 3 numpy array of forces.
            Examples:
                 vasprun = Vasprun('vasprun.xml')
                 cart_coords = vasprun.cart_coords
                 forces = vasprun.forces
            """

            def __init__(self,
                        filename: str) -> None:
                """Initialise a Vasprun object from a vasprun.xml file.
                Args:
                    filename (str): The vasprun.xml filename.
                Returns:
                    None
                """
                doc = etree.parse(filename)
                self.doc = doc.getroot()
                self._atom_names = None # type: Optional[List[str]]
                self._structures = None # type: Optional[List[Structure]]
            

            @property
            def structures(self) -> List[Structure]:
                """Getter for structures attribute.
                Returns:
                    (list(pymatgen.Structure)): A list of pymatgen Structure objects.
                Notes:
                    When first called this parses the vasprun XML data and
                    caches the result.
                """
                if not self._structures:
                    self._structures = self.parse_structures()
                return self._structures

            @property
            def atom_names(self) -> List[str]:
                """Getter for atom_names attribute.
                Returns:
                    (list(str)): A list of atom name strings.
                Notes:
                    When first called this parses the vasprun XML data and
                    caches the result.
                """
                if not self._atom_names:
                    self._atom_names = self.parse_atom_names()
                return self._atom_names

            def parse_atom_names(self) -> List[str]:
                """Return a list of atom names for the atoms in this calculation.
                Args:
                    None
                Returns:
                    (list(str))
                """
                atominfo = self.doc.find("atominfo")
                if atominfo is None:
                    raise ValueError("No atominfo found in file")
                atom_names = []
                for array in atominfo.findall("array"):
                    if array.attrib["name"] == "atoms":
                        atom_names = [rc.find("c").text.strip() for rc in array.find("set")]
                if not atom_names:
                    raise ValueError("No atomname found in file")
                return atom_names

            def parse_structures(self) -> List[Structure]:
                """Returns a list of pymatgen Structures for this calculation.
                Args:
                    None
                Returns:
                    (list(pymatgen.Structure))
                """
                structures = []
                for child in self.doc.iterfind("calculation"):
                    elem = child.find("structure")
                    structure_data = RevelsMDTools.VaspParser.parse_structure(elem)
                    structures.append(
                        RevelsMDTools.VaspParser.structure_from_structure_data(
                            lattice=structure_data['lattice'],
                            atom_names=self.atom_names,
                            frac_coords=structure_data['frac_coords']
                        )
                    )
                return structures

            @property
            def frac_coords(self) -> np.ndarray:
                """Fractional coordinates from each calculation structure.
                Args:
                    None
                Returns:
                    (np.ndarray): timesteps x atoms x 3 numpy array of fractional coordinates.
                """
                frac_coords = np.array([s.frac_coords for s in self.structures])
                return frac_coords

            @property
            def cart_coords(self) -> np.ndarray:
                """Cartesian coordinates from each calculation structure.
                Args:
                    None
                Returns:
                    (np.ndarray): timesteps x atoms x 3 numpy array of fractional coordinates.
                """
                frac_coords = np.array([s.frac_coords for s in self.structures])
                return frac_coords

            @property
            def cart_coords(self) -> np.ndarray:
                """Cartesian coordinates from each calculation structure.
                Args:
                    None
                Returns:
                    (np.ndarray): timesteps x atoms x 3 numpy array of cartesian coordinates.
                """
                cart_coords = np.array([s.cart_coords for s in self.structures])
                return cart_coords

            @property
            def forces(self) -> Optional[np.ndarray]:
                """Cartesian forces from each calculation structure
                (if present in the vasprun XML).
                Args:
                    None
                Returns:
                    (np.ndarray|None): timesteps x atoms x 3 numpy array of cartesian forces
                        if forces are included in the vasprun XML. If not, returns None.
                """
                forces = []
                for child in self.doc.iterfind("calculation"):
                    elem = child.find("varray/[@name='forces']")
                    if elem != None:
                        forces.append(RevelsMDTools.VaspParser.parse_varray(elem))
                if forces:
                    return np.array(forces)
                else:
                    return None

class RevelsMDRDF:
    def single_frame_rdf_like(pos_array,force_array,indicies,box_x,box_y,box_z,bins,return_conventional=False):
        """
        This is function for obtaining an single fram radial distribution function for a single species with itself

        args:
        pos_array (np.array(n,3)): An array with collumns rx, ry, rz
        force_array (np.array(n,3)): An array with collumns fx, fy, fz
        indicies (np.array): The row numbers for the species of interest
        box_x (float): The size of the box in the x direction
        box_y (float): The size of the box in the y direction
        box_z (float): The size of the box in the z direction
        bins (np.array(n)): The positions in r for which the radial distribution function will be calculated
        kwargs:
        return_convention(bool): UNDER CONSTRUCTION If true the conventional histogram based rdf will be returned with bins centred on delr value (default=false) 
        returns:
        A 2 dimensional numpy array of delr values and acommpanying delr values

        """
        pos_ang=pos_array[indicies,:] 
        force_total=force_array[indicies,:]
        storage_array=np.zeros(np.size(bins), dtype=np.longdouble)
        ns=len(indicies)
        rx=np.zeros((ns,ns))
        ry=np.zeros((ns,ns))
        rz=np.zeros((ns,ns))
        Fx=np.zeros((ns,ns))
        Fy=np.zeros((ns,ns))
        Fz=np.zeros((ns,ns))
        for x in range(ns):
            ry[x,:]=pos_ang[:,1]-pos_ang[x,1] 
            rx[x,:]=pos_ang[:,0]-pos_ang[x,0] 
            rz[x,:]=pos_ang[:,2]-pos_ang[x,2]
            Fx[x,:]=force_total[:,0]
            Fy[x,:]=force_total[:,1]
            Fz[x,:]=force_total[:,2]
        rx-= (np.ceil((np.abs(rx)-box_x/2)/box_x))*((box_x))*np.sign(rx)
        ry-= (np.ceil((np.abs(ry)-box_y/2)/box_y))*((box_y))*np.sign(ry)
        rz-= (np.ceil((np.abs(rz)-box_z/2)/box_z))*((box_z))*np.sign(rz)
        r= (rx*rx+ry*ry+rz*rz)**.5
        with np.errstate(divide='ignore',invalid='ignore'):
            dot_prod=((Fz*rz)+(Fy*ry)+(Fx*rx))/r/r/r
        dot_prod[(rx>box_x/2)+(ry>box_y/2)+(rz>box_z/2)]=0
        dp=dot_prod.reshape(-1)
        rn=r.reshape(-1) 

        digtized_array=np.digitize(rn,bins)-1
        dp[digtized_array==np.size(bins)-1]=0
        storage_array[(np.size(bins)-1)]= np.sum(dp[(digtized_array==np.size(bins)-1)]) #conduct heaviside for our first bin
        for l in range(np.size(bins)-2,-1,-1):
            storage_array[l]= np.sum(dp[(digtized_array==l)])#conduct subsequent heavisides with a rolling sum
        return storage_array

    def single_frame_rdf_unlike(pos_array,force_array,indicies,box_x,box_y,box_z,bins,return_conventional=False):
        """
        This is function for obtaining an single fram radial distribution function for a single species with itself

        args:
        pos_array (np.array(n,3)): An array with collumns rx, ry, rz
        force_array (np.array(n,3)): An array with collumns fx, fy, fz
        indicies ([np.array,np.array]): The row numbers for the species of interest for the first and second species respectively
        box_x (float): The size of the box in the x direction
        box_y (float): The size of the box in the y direction
        box_z (float): The size of the box in the z direction
        bins (np.array(n)): The positions in r for which the radial distribution function will be calculated
        kwargs:
        return_convention(bool): UNDER CONSTRUCTION  If true the conventional histogram based rdf will be returned with bins centred on delr value (default=false)
        returns:
        A 2 dimensional numpy array of delr values and acommpanying delr values

        """
        pos_ang_1=pos_array[indicies[0],:] 
        force_total_1=force_array[indicies[0],:]
        pos_ang_2=pos_array[indicies[1],:] 
        force_total_2=force_array[indicies[1],:]
        storage_array=np.zeros(np.size(bins), dtype=np.longdouble)
        n1=len(indicies[0])
        n2=len(indicies[1])
        rx=np.zeros((n2,n1))
        ry=np.zeros((n2,n1))
        rz=np.zeros((n2,n1))
        Fx=np.zeros((n2,n1))
        Fy=np.zeros((n2,n1))
        Fz=np.zeros((n2,n1))
        for x in range(n2):
            ry[x,:]=pos_ang_1[:,1]-pos_ang_2[x,1] 
            rx[x,:]=pos_ang_1[:,0]-pos_ang_2[x,0] 
            rz[x,:]=pos_ang_1[:,2]-pos_ang_2[x,2]
            Fx[x,:]=force_total_1[:,0]-force_total_2[x,0] 
            Fy[x,:]=force_total_1[:,1]-force_total_2[x,1] 
            Fz[x,:]=force_total_1[:,2]-force_total_2[x,2] 
        rx-= (np.ceil((np.abs(rx)-box_x/2)/box_x))*((box_x))*np.sign(rx)
        ry-= (np.ceil((np.abs(ry)-box_y/2)/box_y))*((box_y))*np.sign(ry)
        rz-= (np.ceil((np.abs(rz)-box_z/2)/box_z))*((box_z))*np.sign(rz)
        r= (rx*rx+ry*ry+rz*rz)**.5
        with np.errstate(divide='ignore',invalid='ignore'):
            dot_prod=((Fz*rz)+(Fy*ry)+(Fx*rx))/r/r/r
        dot_prod[(rx>box_x/2)+(ry>box_y/2)+(rz>box_z/2)]=0
        dp=dot_prod.reshape(-1)
        rn=r.reshape(-1) 
        digtized_array=np.digitize(rn,bins)-1
        dp[digtized_array==np.size(bins)-1]=0
        storage_array[(np.size(bins)-1)]= np.sum(dp[(digtized_array==np.size(bins)-1)]) #conduct heaviside for our first bin
        for l in range(np.size(bins)-2,-1,-1):
            storage_array[l]= np.sum(dp[(digtized_array==l)])#conduct subsequent heavisides with a rolling sum
        return storage_array

    def run_rdf (TS,atom_a,atom_b,temp,delr=.01,start=0,stop=-1,period=1, rmax=True, from_zero=True):
        """
        This is the master function for running a force RDF.

        args:
        TS (A RevelsMD trajectory state object): An object obtaining all of the trajectory paramaters
        atom_a (string): The type of the first atom for which the rdf will be calculated
        atom_b (string): The type of the second atom for which the rdf will be calculated, if atom_a == atom_b a like pairs rdf is automatically calculated
        temp (float): Temperature of the system
        delr (float): The spacing between radial points in an RDF (this is not a bin width as this is not a histogram but a heaviside)
        kwargs:
        start (int): The first frame for which the radial distribution function will be calculated
        stop (int): The last value for which the radial distribution function will be calculated
        period (int): The jumps made between sampled frames
        rmax (float): The maximum radial position defaults to follow the minimum image convention
        from_zero (bool): A boolian value if True the Heviside is taken from zero if false it is take from rmax

        returns:
        A 2 dimensional numpy array of r values and acommpanying rdf values

        """
        
        if atom_a == atom_b:
            single_frame_function = RevelsMDRDF.single_frame_rdf_like
            indicies = TS.get_indicies(atom_a)
            prefactor= float(TS.box_x*TS.box_y*TS.box_z)/(float(len(indicies))*float(len(indicies)-1))

        else:
            indicies = [np.array(TS.get_indicies(atom_a)),np.array(TS.get_indicies(atom_b))]
            single_frame_function = RevelsMDRDF.single_frame_rdf_unlike
            prefactor= float(TS.box_x*TS.box_y*TS.box_z)/(float(len(indicies[1]))*float(len(indicies[0])))/2

        if start > TS.frames:
            print('First frame index exceeds frames in trajectory')
            return
        if stop > TS.frames:
            print('Final frame index exceeds frames in trajectory')
            return
        to_run=range(int(start%TS.frames),int(stop%TS.frames),period)
        if len(to_run) == 0:
            print('Final frame ocurs before first frame in trajectory')
            return
        if TS.variety == 'lammps':
            f=open(TS.trajectory_file)
            neededQuantities=['x','y','z','fx','fy','fz']
            stringdex=RevelsMDTools.LammpsParser.define_strngdex(neededQuantities,TS.dic)
            if rmax:
                bins= np.arange(0,np.max([TS.box_x/2,TS.box_y/2,TS.box_z/2]),delr)
            else:
                bins= np.arange(0,float(rmax),delr)
            accumulated_storage_array=np.zeros(np.size(bins), dtype=np.longdouble)
            for frame_count in tqdm(to_run):
                vars_trest=RevelsMDTools.LammpsParser.get_a_frame(f,TS.num_ats,TS.header_length,stringdex)
                accumulated_storage_array+=single_frame_function(vars_trest[:,:3],vars_trest[:,3:],indicies,TS.box_x,TS.box_y,TS.box_z,bins)
                RevelsMDTools.LammpsParser.frame_skip(f,TS.num_ats,period-1,TS.header_length)
        elif TS.variety == 'mda':
            if rmax:
                bins= np.arange(0,TS.box_x/2,delr)
            else:
                bins= np.arange(0,float(rmax),delr)
            accumulated_storage_array=np.zeros(np.size(bins), dtype=np.longdouble)
            for frame_count in tqdm(TS.mdanalysis_universe.trajectory[int(start%TS.frames):int(stop%TS.frames):period]):
                accumulated_storage_array+=single_frame_function(TS.mdanalysis_universe.trajectory.atoms.positions,TS.mdanalysis_universe.trajectory.atoms.forces,indicies,TS.box_x,TS.box_y,TS.box_z,bins)
        elif TS.variety == 'vasp':
            if rmax:
                bins= np.arange(0,TS.box_x/2,delr)
            else:
                bins= np.arange(0,float(rmax),delr)
            accumulated_storage_array=np.zeros(np.size(bins), dtype=np.longdouble)
            for frame_count in tqdm(to_run):
                accumulated_storage_array+=single_frame_function(TS.positions[frame_count],TS.forces[frame_count],indicies,TS.box_x,TS.box_y,TS.box_z,bins)
        elif TS.variety == 'numpy':
            if rmax:
                bins= np.arange(0,TS.box_x/2,delr)
            else:
                bins= np.arange(0,float(rmax),delr)
            accumulated_storage_array=np.zeros(np.size(bins), dtype=np.longdouble)
            for frame_count in tqdm(to_run):
                accumulated_storage_array+=single_frame_function(TS.positions[frame_count],TS.forces[frame_count],indicies,TS.box_x,TS.box_y,TS.box_z,bins)
        accumulated_storage_array=np.nan_to_num(accumulated_storage_array)
        accumulated_storage_array*=prefactor/(4*np.pi*len(to_run)*RevelsMDTools.ConversionFactors.generate_boltzmann(TS.units)*temp)
        if from_zero == True:
            return np.array([bins,np.cumsum(accumulated_storage_array)])
        else:
            return np.array([bins,1-np.cumsum(accumulated_storage_array[::-1])[::-1]])

    
    def run_rdf_lambda (TS,atom_a,atom_b,temp,delr=.01,start=0,stop=-1,period=1, rmax=True):
        """
        This is the master function for running a linear combination of forward and backward RDFs

        args:
        TS (A RevelsMD trajectory state object): An object obtaining all of the trajectory paramaters
        atom_a (string): The type of the first atom for which the rdf will be calculated
        atom_b (string): The type of the second atom for which the rdf will be calculated, if atom_a == atom_b a like pairs rdf is automatically calculated
        temp (float): Temperature of the system
        delr (float): The spacing between radial points in an RDF (this is not a bin width as this is not a histogram but a heaviside)
        kwargs:
        start (int): The first frame for which the radial distribution function will be calculated
        stop (int): The last value for which the radial distribution function will be calculated
        period (int): The jumps made between sampled frames
        rmax (float): The maximum radial position defaults to follow the minimum image convention
        from_zero (bool): A boolian value if True the Heviside is taken from zero if false it is take from rmax

        returns:
        A 2 dimensional numpy array of r values and acommpanying rdf values

        """
        if atom_a == atom_b:
            single_frame_function = RevelsMDRDF.single_frame_rdf_like
            indicies = TS.get_indicies(atom_a)
            prefactor= float(TS.box_x*TS.box_y*TS.box_z)/(float(len(indicies))*float(len(indicies)-1))

        else:
            indicies = [TS.get_indicies(atom_a),TS.get_indicies(atom_b)]
            single_frame_function = RevelsMDRDF.single_frame_rdf_unlike
            prefactor= float(TS.box_x*TS.box_y*TS.box_z)/(float(len(indicies[1]))*float(len(indicies[0])))/2

        if start > TS.frames:
            print('First frame index exceeds frames in trajectory')
            return
        if stop > TS.frames:
            print('Final frame index exceeds frames in trajectory')
            return
        to_run=range(int(start%TS.frames),int(stop%TS.frames),period)
        if len(to_run) == 0:
            print('Final frame ocurs before first frame in trajectory')
            return
        list_store = []
        if TS.variety == 'lammps':
            f=open(TS.trajectory_file)
            neededQuantities=['x','y','z','fx','fy','fz']
            stringdex=RevelsMDTools.LammpsParser.define_strngdex(neededQuantities,TS.dic)
            if rmax:
                bins= np.arange(0,np.max([TS.box_x/2,TS.box_y/2,TS.box_z/2]),delr)
            else:
                bins= np.arange(0,float(rmax),delr)
            accumulated_storage_array=np.zeros(np.size(bins), dtype=np.longdouble)
            for frame_count in tqdm(to_run):
                vars_trest=RevelsMDTools.LammpsParser.get_a_frame(f,TS.num_ats,TS.header_length,stringdex)
                this_frame=single_frame_function(vars_trest[:,:3],vars_trest[:,3:],indicies,TS.box_x,TS.box_y,TS.box_z,bins)
                accumulated_storage_array+=this_frame
                list_store.append(this_frame)
                RevelsMDTools.LammpsParser.frame_skip(f,TS.num_ats,period-1,TS.header_length)
        elif TS.variety == 'mda':
            if rmax:
                bins= np.arange(0,TS.box_x/2,delr)
            else:
                bins= np.arange(0,float(rmax),delr)
            accumulated_storage_array=np.zeros(np.size(bins), dtype=np.longdouble)
            for frame_count in tqdm(TS.mdanalysis_universe.trajectory[int(start%TS.frames):int(stop%TS.frames):period]):
                this_frame=single_frame_function(TS.mdanalysis_universe.atoms.positions,TS.mdanalysis_universe.atoms.forces,indicies,TS.box_x,TS.box_y,TS.box_z,bins)
                accumulated_storage_array+=this_frame
                list_store.append(this_frame)
        elif TS.variety == 'vasp':
            if rmax:
                bins= np.arange(0,TS.box_x/2,delr)
            else:
                bins= np.arange(0,float(rmax),delr)
            accumulated_storage_array=np.zeros(np.size(bins), dtype=np.longdouble)
            for frame_count in tqdm(to_run):
                this_frame=single_frame_function(TS.positions[frame_count],TS.forces[frame_count],indicies,TS.box_x,TS.box_y,TS.box_z,bins)
                accumulated_storage_array+=this_frame
                list_store.append(this_frame)
        elif TS.variety == 'numpy':
            if rmax:
                bins= np.arange(0,TS.box_x/2,delr)
            else:
                bins= np.arange(0,float(rmax),delr)
            accumulated_storage_array=np.zeros(np.size(bins), dtype=np.longdouble)
            for frame_count in tqdm(to_run):
                this_frame=single_frame_function(TS.positions[frame_count],TS.forces[frame_count],indicies,TS.box_x,TS.box_y,TS.box_z,bins)
                accumulated_storage_array+=this_frame
                list_store.append(this_frame)
        base_array=np.nan_to_num(np.array(list_store))
        base_array*=prefactor/(4*np.pi*RevelsMDTools.ConversionFactors.generate_boltzmann(TS.units)*temp)
        accumulated_storage_array=np.nan_to_num(accumulated_storage_array)
        accumulated_storage_array*=prefactor/(4*np.pi*len(to_run)*RevelsMDTools.ConversionFactors.generate_boltzmann(TS.units)*temp)
        exp_zero_rdf=np.array(np.cumsum(accumulated_storage_array)[:-1])
        exp_inf_rdf=np.array(1-np.cumsum(accumulated_storage_array[::-1])[::-1][1:])
        exp_delta=exp_inf_rdf-exp_zero_rdf
        base_zero_rdf=np.array(np.cumsum(base_array,axis=1))[:,:-1]
        base_inf_rdf=np.array(1-np.cumsum(base_array[:,::-1],axis=1)[:,::-1][:,1:])
        base_delta = base_inf_rdf - base_zero_rdf
        var_del=np.mean((base_delta-exp_delta)**2,axis=0)
        cov_inf=np.mean((base_delta-exp_delta)*(base_inf_rdf-exp_inf_rdf),axis=0)
        combination = cov_inf/var_del
        return np.transpose(np.array([bins[1:],np.mean(base_inf_rdf*(1-combination)+(base_zero_rdf*combination),axis=0),combination]))

class RevelsMD3D:
    class GridState:
        def __init__(self,TS,density_type,temperature,nbins=100,nbinsx=False,nbinsy=False,nbinsz=False):
            """
            This class is where all the calculations of a 3 dimensional densities are calculated

            args:
            TS (A RevelsMD trajectory state object): An object obtaining all of the trajectory paramaters
            density_type (string): type of density we wish to calculate
            temperature (float): temperature of the system being modelled
            kwargs:
            nbins (int): the number of bins in all directions, can be overwritten by nbinsx, nbinsy and nbinsz
            nbinsx (int): the number of bins in the x direction, overwrites nbins
            nbinsy (int): the number of bins in the y direction, overwrites nbins
            nbinsz (int): the number of bins in the z direction, overwrites nbins

            """
            if nbinsx == False:
                nbinsx=nbins
            if nbinsy == False:
                nbinsy=nbins
            if nbinsz == False:
                nbinsz=nbins
            lx=TS.box_x/nbinsx
            ly=TS.box_y/nbinsy
            lz=TS.box_z/nbinsz
            self.box_array =np.array([TS.box_x,TS.box_y,TS.box_z])
            self.binsx=np.arange(0,TS.box_x+lx,lx)
            self.binsy=np.arange(0,TS.box_y+ly,ly)
            self.binsz=np.arange(0,TS.box_z+lz,lz)
            if TS.variety == 'vasp':
                self.cell=TS.Vasprun.start.lattice.matrix
            self.temperature=temperature
            self.lx=lx
            self.ly=ly
            self.lz=lz
            self.count=0
            self.units=TS.units
            self.nbinsx=nbinsx
            self.nbinsy=nbinsy
            self.nbinsz=nbinsz
            self.forceX=np.zeros([nbinsx,nbinsy,nbinsz])
            self.forceY=np.zeros([nbinsx,nbinsy,nbinsz])
            self.forceZ=np.zeros([nbinsx,nbinsy,nbinsz])
            self.counter=np.zeros([nbinsx,nbinsy,nbinsz])
            self.grid_progress='Generated'
            density_type=density_type.lower()
            if density_type in ['number','charge','polarisation']:
                self.density_type=density_type
            else:
                print("Density type must be one of 'number','charge' or ,'polarisation'")

        def make_force_grid(self,TS,atom_names,rigid=False,centre_location=True,kernel="triangular",polarisation_axis=0,start=0,stop=-1,period=1):
            """
            This class is where all the calculations of a 3 dimensional densities are calculated

            args:
            TS (A RevelsMD trajectory state object): An object obtaining all of the trajectory paramaters
            density_type (string): type of density we wish to calculate
            kwargs:
            nbins (int): the number of bins in all directions, can be overwritten by nbinsx, nbinsy and nbinsz
            nbinsx (int): the number of bins in the x direction, overwrites nbins
            nbinsy (int): the number of bins in the y direction, overwrites nbins
            nbinsz (int): the number of bins in the z direction, overwrites nbins

            """            
            if start > TS.frames:
                print('First frame index exceeds frames in trajectory')
                return
            self.start=start
            if stop > TS.frames:
                print('Final frame index exceeds frames in trajectory')
                return
            self.stop=stop
            to_run=range(int(start%TS.frames),int(stop%TS.frames),period)
            if len(to_run) == 0:
                print('Final frame ocurs before first frame in trajectory')
                return
            self.period=period
            self.kernel=kernel
            self.to_run=range(int(start%TS.frames),int(stop%TS.frames),period)
            self.SS=RevelsMD3D.SelectionState(TS,atom_names=atom_names,centre_location=centre_location)
            if self.density_type.lower() == 'number':
                if self.SS.indistinguishable_set==False:
                    if rigid == True:
                        if centre_location == True:
                            self.single_frame_function = RevelsMD3D.Estimators.single_frame_rigid_number_com_grid
                        elif type(centre_location) is int:
                            self.single_frame_function = RevelsMD3D.Estimators.single_frame_rigid_number_atom_grid
                        else:
                            print("error centre location must be True (com) or int (specific atom)")
                    else:
                        self.single_frame_function =  RevelsMD3D.Estimators.single_frame_number_many_grid
                else:
                    self.single_frame_function = RevelsMD3D.Estimators.single_frame_number_single_grid
            elif self.density_type.lower() == "charge":
                if self.SS.indistinguishable_set==False:
                    if rigid:
                        if centre_location == True:
                            self.single_frame_function = RevelsMD3D.Estimators.single_frame_rigid_charge_com_grid
                        elif type(centre_location) is int:
                            self.single_frame_function = RevelsMD3D.Estimators.single_frame_rigid_charge_atom_grid
                        else:
                            print("error centre location must be True (com) or int (specific atom)")
                    else:
                        self.single_frame_function = RevelsMD3D.Estimators.single_frame_charge_many_grid
                else:
                    self.single_frame_function = RevelsMD3D.Estimators.single_frame_number_single_grid
                
            elif self.density_type.lower() == "polarisation":
                if self.SS.indistinguishable_set==False:
                    if rigid:
                        if centre_location == True:
                            self.single_frame_function = RevelsMD3D.Estimators.single_frame_rigid_polarisation_com_grid
                            self.SS.polarisation_axis=polarisation_axis
                        elif type(centre_location) is int:
                            self.single_frame_function = RevelsMD3D.Estimators.single_frame_rigid_polarisation_atom_grid
                            self.SS.polarisation_axis=polarisation_axis
                        else:
                            print("error centre location must be True (com) or a spexified atom")
                    else:
                        print("At present the code only calculates polarisation densities for rigid molecules")
                else:
                    print("A single atom does not have a polarisation density please specify a molecule (this molecules must be rigid")
            else:
                print("the only supported densities at this time are: number, polarisation and charge")
            if TS.variety == 'lammps':
                f=open(TS.trajectory_file)
                neededQuantities=['x','y','z','fx','fy','fz']
                stringdex=RevelsMDTools.LammpsParser.define_strngdex(neededQuantities,TS.dic)

                for frame_count in tqdm(self.to_run):
                    vars_trest=RevelsMDTools.LammpsParser.get_a_frame(f,TS.num_ats,TS.header_length,stringdex)
                    self.single_frame_function(vars_trest[:,:3],vars_trest[:,3:],TS,self,self.SS,kernel=self.kernel)
                    RevelsMDTools.LammpsParser.frame_skip(f,TS.num_ats,period-1,TS.header_length)
            elif TS.variety == 'mda':
                print( self.single_frame_function)
                for frame_count in tqdm(self.to_run):
                    self.single_frame_function(TS.mdanalysis_universe.trajectory[frame_count].positions,TS.mdanalysis_universe.trajectory[frame_count].forces,TS,self,self.SS,kernel=self.kernel)
            elif TS.variety == 'vasp':
                for frame_count in tqdm(self.to_run):
                    self.single_frame_function(TS.positions[frame_count],TS.forces[frame_count],TS,self,self.SS,kernel=self.kernel)
            elif TS.variety == 'numpy':
                for frame_count in tqdm(self.to_run):
                    self.single_frame_function(TS.positions[frame_count],TS.forces[frame_count],TS,self,self.SS,kernel=self.kernel)
            self.frames_processed = self.to_run
            self.grid_progress='Allocated'
            
        def get_real_density(self):
            if self.grid_progress == 'Generated':
                print("You must run make_force_grid before attempting to obtain a density")
                return
            # Perfom the forward fourier transforms of the force density, after having normalised
            with np.errstate(divide='ignore',invalid='ignore'):
                forceX=np.fft.fftn(self.forceX/self.lx/self.ly/self.lz/self.count)
                forceY=np.fft.fftn(self.forceY/self.lx/self.ly/self.lz/self.count)
                forceZ=np.fft.fftn(self.forceZ/self.lx/self.ly/self.lz/self.count)
            #prepare the k vectors
            xrep, yrep, zrep = self.get_kvectors()
            for n in range(len(xrep)):
                forceX[n,:,:] = xrep[n] * forceX[n,:,:] # perform a row wise dot product for the x dimension
            for m in range(len(yrep)):
                forceY[:,m,:] = yrep[m] * forceY[:,m,:] # perform a row wise dot product for the y dimension
            for l in range(len(zrep)):
                forceZ[:,:,l] = zrep[l] * forceZ[:,:,l] # perform a row wise dot product for the z dimension
            #Perform equation 23 from Borgis et al., Mol. Phys. 111, 3486–3492 (2013)
            with np.errstate(divide='ignore',invalid='ignore'):
                self.del_rho_k = (complex(0,1) / (self.temperature*RevelsMDTools.ConversionFactors.generate_boltzmann(self.units)*self.get_ksquared()) * (forceX + forceY + forceZ))
            self.del_rho_k[0,0,0] = 0
            del_rho_n = np.fft.ifftn(self.del_rho_k) #inverse fast fourier transform back to real space.
            self.del_rho_n = -1*np.real(del_rho_n)
            self.get_particle_density()
            self.rho=self.del_rho_n+np.mean(self.particle_density)

        def get_particle_density(self):
            '''
            Using the results from make force grids to get a conventional density of the type we are calculating by counting.
            If the force grid has not been made this will yield zeros
            '''
            if self.grid_progress == 'Generated':
                print("You must run make_force_grid before attempting to obtain a density")
                return
            with np.errstate(divide='ignore',invalid='ignore'):
                self.particle_density=self.counter/self.lx/self.ly/self.lz/self.count
        
        def get_kvectors(self):
            xrep = 2*np.pi*np.fft.fftfreq(self.nbinsx, d=self.lx)
            yrep = 2*np.pi*np.fft.fftfreq(self.nbinsy, d=self.ly)
            zrep = 2*np.pi*np.fft.fftfreq(self.nbinsz, d=self.lz)
            return xrep, yrep, zrep
        
        def get_kvectors_no_right(self):
            xunit=np.cross(self.cell[:,1],self.cell[:,2])/np.dot(self.cell[:,0],np.cross(self.cell[:,1],self.cell[:,2]))*2*np.pi
            yunit=np.cross(self.cell[:,2],self.cell[:,0])/np.dot(self.cell[:,1],np.cross(self.cell[:,2],self.cell[:,0]))*2*np.pi
            zunit=np.cross(self.cell[:,0],self.cell[:,1])/np.dot(self.cell[:,2],np.cross(self.cell[:,0],self.cell[:,1]))*2*np.pi
            xrep = np.array([np.nan_to_num(np.fft.fftfreq(self.nbinsx, d=xunit[0]),0),np.nan_to_num(np.fft.fftfreq(self.nbinsx, d=xunit[1]),0),np.nan_to_num(np.fft.fftfreq(self.nbinsx, d=xunit[2]),0)])
            yrep = np.array([np.nan_to_num(np.fft.fftfreq(self.nbinsy, d=yunit[0]),0),np.nan_to_num(np.fft.fftfreq(self.nbinsy, d=yunit[1]),0),np.nan_to_num(np.fft.fftfreq(self.nbinsy, d=yunit[2]),0)])
            zrep = np.array([np.nan_to_num(np.fft.fftfreq(self.nbinsz, d=zunit[0]),0),np.nan_to_num(np.fft.fftfreq(self.nbinsz, d=zunit[1]),0),np.nan_to_num(np.fft.fftfreq(self.nbinsz, d=zunit[2]),0)])
            return xrep, yrep, zrep


        def get_ksquared(self):
            xrep, yrep, zrep = self.get_kvectors()

            # Propagation to 2D
            xrep = np.repeat(xrep[:, np.newaxis, np.newaxis], self.nbinsy, axis=1)
            zrep = np.repeat(zrep[np.newaxis, np.newaxis, :], self.nbinsx, axis=0)
            yrep = np.repeat(yrep[np.newaxis, :, np.newaxis], self.nbinsx, axis=0)
    
            # Propagation to 3D
            xrep = np.repeat(xrep[:, :, :], self.nbinsz, axis=2)
            yrep = np.repeat(yrep[:, :, :], self.nbinsz, axis=2)
            zrep = np.repeat(zrep[:, :, :], self.nbinsy, axis=1)
        
            return (xrep*xrep)+(yrep*yrep)+(zrep*zrep) # generate the square of kvectorsa
        def write_to_cube(self, structure, grid, filename, convert_pmg=True):
            """Write 3D density to a .cube file.
            
            Args:
                structure (pymatgen.Structure): Example structure, used to define the cell geometry.
                grid (np.array): 3D numpy array.
                filename (str): filename to write to.
                
            Returns:
                None
                
            """
            if convert_pmg == True:
                 atoms = AseAtomsAdaptor.get_atoms(structure)
                 del atoms[np.array(self.SS.indicies)]
            else:
                atoms=structure
            with open(filename, 'w') as f:
                write_cube(f, atoms, data=grid)

        def get_lambda(self,TS,sections=None):
            """
            This section generates a linear combination of the conventional and force methods using the method presented in J. Chem. Phys. 154, 191101.
            It returns a copy of a grid state with additional features added 
            """
            if self.grid_progress == 'Generated':
                print("You must run make_force_grid before attempting to obtain a value of lambda")
                return
            if self.grid_progress == 'Lambda':
                print("This grid was generated from a previous get_lambda")
                return
            GS_Lambda = copy.deepcopy(self)
            if sections == None:
                sections = TS.frames
            GS_Lambda.get_real_density()
            GS_Lambda.expected_rho=np.copy(GS_Lambda.rho)
            GS_Lambda.expected_particle_density=np.copy(GS_Lambda.particle_density)
            GS_Lambda.delta=GS_Lambda.expected_rho-GS_Lambda.expected_particle_density
            GS_Lambda.cov_buffer_particle=np.zeros([GS_Lambda.nbinsx,GS_Lambda.nbinsy,GS_Lambda.nbinsz])
            GS_Lambda.cov_buffer_force=np.zeros([GS_Lambda.nbinsx,GS_Lambda.nbinsy,GS_Lambda.nbinsz])
            GS_Lambda.var_buffer=np.zeros([GS_Lambda.nbinsx,GS_Lambda.nbinsy,GS_Lambda.nbinsz])
            for k in tqdm(range(sections)):
                GS_Lambda.forceX*=0
                GS_Lambda.forceY*=0
                GS_Lambda.forceZ*=0
                GS_Lambda.particle_density*=0
                GS_Lambda.counter*=0
                GS_Lambda.del_rho_k*=0
                GS_Lambda.del_rho_n*=0
                GS_Lambda.rho*=0
                GS_Lambda.count*=0
                if TS.variety == 'lammps':
                    f=open(TS.trajectory_file)
                    neededQuantities=['x','y','z','fx','fy','fz']
                    stringdex=RevelsMDTools.LammpsParser.define_strngdex(neededQuantities,TS.dic)
                    for frame_count in (GS_Lambda.to_run[np.arange(k,len(GS_Lambda.to_run)//sections,sections)]):
                        vars_trest=RevelsMDTools.LammpsParser.get_a_frame(f,TS.num_ats,TS.header_length,stringdex)
                        GS_Lambda.single_frame_function(vars_trest[:,:3],vars_trest[:,3:],TS,GS_Lambda,GS_Lambda.SS,kernel=GS_Lambda.kernel)
                        RevelsMDTools.LammpsParser.frame_skip(f,TS.num_ats,period-1,TS.header_length)
                elif TS.variety == 'mda':
                    for frame_count in (np.array(GS_Lambda.to_run)[np.arange(k,sections*(len(GS_Lambda.to_run)//sections),sections)]):
                        GS_Lambda.single_frame_function(TS.mdanalysis_universe.trajectory[frame_count].positions,TS.mdanalysis_universe.trajectory[frame_count].forces,TS,GS_Lambda,GS_Lambda.SS,kernel=GS_Lambda.kernel)
                elif TS.variety == 'vasp':
                    for frame_count in (np.array(GS_Lambda.to_run)[np.arange(k,sections*(len(GS_Lambda.to_run)//sections),sections)]):
                        GS_Lambda.single_frame_function(TS.positions[frame_count],TS.forces[frame_count],TS,GS_Lambda,GS_Lambda.SS,kernel=GS_Lambda.kernel)
                GS_Lambda.get_real_density()
                delta_cur=(GS_Lambda.rho-GS_Lambda.particle_density)
                GS_Lambda.var_buffer+=(delta_cur-GS_Lambda.delta)**2
                GS_Lambda.cov_buffer_force+=(delta_cur-GS_Lambda.delta)*(GS_Lambda.rho-GS_Lambda.expected_rho)
                GS_Lambda.cov_buffer_particle+=(delta_cur-GS_Lambda.delta)*(GS_Lambda.particle_density-GS_Lambda.expected_particle_density)
            GS_Lambda.combination=1-(GS_Lambda.cov_buffer_force/GS_Lambda.var_buffer)
            GS_Lambda.optimal_density=(1-GS_Lambda.combination)*GS_Lambda.expected_particle_density+GS_Lambda.combination*GS_Lambda.expected_rho
            return GS_Lambda

    class Estimators:
        def single_frame_rigid_number_com_grid(positions,forces,TS,GS,SS,kernel='triangular'):
            coms = RevelsMD3D.HelperFunctions.find_coms(positions,TS,GS,SS)
            rigid_forces = RevelsMD3D.HelperFunctions.sum_forces(SS,forces)
            RevelsMD3D.HelperFunctions.process_frame(TS,GS,coms,rigid_forces,kernel=kernel)

        
        def single_frame_rigid_number_atom_grid(positions,forces,TS,GS,SS,kernel='triangular'):
        
            rigid_forces = RevelsMD3D.HelperFunctions.sum_forces(SS,forces)
            RevelsMD3D.HelperFunctions.process_frame(TS,GS,positions[SS.indicies[SS.centre_location],:],rigid_forces,kernel=kernel)

        def single_frame_number_many_grid(positions,forces,TS,GS,SS,kernel='triangular'):
        
            for count in range(len(SS.indicies)):
                RevelsMD3D.HelperFunctions.process_frame(TS,GS,positions[SS.indicies[count],:],forces[SS.indicies[count],:],kernel=kernel)

        def single_frame_number_single_grid(positions,forces,TS,GS,SS,kernel='triangular'):
            RevelsMD3D.HelperFunctions.process_frame(TS,GS,positions[SS.indicies,:],forces[SS.indicies,:],kernel=kernel)

        def single_frame_rigid_charge_atom_grid(positions,forces,TS,GS,SS,kernel='triangular'):
            rigid_forces=RevelsMD3D.HelperFunctions.sum_forces(SS,forces)
            RevelsMD3D.HelperFunctions.process_frame(TS,GS,positions[SS.indicies[SS.centre_location],:],rigid_forces,a=SS.charges[SS.centre_location],kernel=kernel)

        def single_frame_charge_many_grid(positions,forces,TS,GS,SS,kernel='triangular'):
            for count in range(len(SS.indicies)):
                RevelsMD3D.HelperFunctions.process_frame(TS,GS,positions[SS.indicies[count],:],forces[SS.indicies[count],:],a=SS.charges[count],kernel=kernel)

        def single_frame_charge_single_grid(positions,forces,TS,GS,SS,kernel='triangular'):
            RevelsMD3D.HelperFunctions.process_frame(TS,GS,positions[SS.indicies,:],forces[SS.indicies,:],a=SS.charges,kernel=kernel)

        def single_frame_rigid_charge_com_grids(positions,forces,TS,GS,SS,kernel='triangular'):
            coms = RevelsMD3D.HelperFunctions.find_coms(positions,TS,GS,SS)
            rigid_forces = RevelsMD3D.HelperFunctions.sum_forces(SS,forces)
            RevelsMD3D.HelperFunctions.process_frame(TS,GS,coms,rigid_forces,kernel=kernel,a=SS.charges)

        def single_frame_rigid_polarisation_com_grid(positions,forces,TS,GS,SS,kernel='triangular'):
            coms,molecular_dipole = RevelsMD3D.HelperFunctions.find_coms(positions,TS,GS,SS,calc_dipoles=True)
            rigid_forces = RevelsMD3D.HelperFunctions.sum_forces(SS,forces)
            RevelsMD3D.HelperFunctions.process_frame(TS,GS,coms,rigid_forces,a=molecular_dipole[:,GS.SS.polarisation_axis],kernel=kernel)


        def single_frame_rigid_polarisation_atom_grid(positions,forces,TS,GS,SS,kernel='triangular'):
            coms,molecular_dipole = RevelsMD3D.HelperFunctions.find_coms(positions,TS,GS,SS,calc_dipoles=True)
            rigid_forces = RevelsMD3D.HelperFunctions.sum_forces(SS,forces)
            RevelsMD3D.HelperFunctions.process_frame(TS,GS,coms,rigid_forces,a=molecular_dipole[:,GS.SS.polarisation_axis],kernel=kernel)
            
            
    class SelectionState:
        def __init__(self,TS,atom_names,centre_location):
            if type(atom_names) == list and len(atom_names)>1:
                self.indistinguishable_set=False
                self.indicies=[]
                self.charges=[]
                self.masses=[]
                for atom in atom_names:
                    self.indicies.append(TS.get_indicies(atom))
                    if TS.charge_and_mass == True:
                        self.charges.append(TS.get_charges(atom))
                        self.masses.append(TS.get_masses(atom))
                if centre_location < len(atom_names):
                    self.centre_location=centre_location
                else:
                    print("centre_location greater than highest index in atom list")
                    return
            else:
                if type(atom_names) == list:
                    atom_names=atom_names[0]
                self.indistinguishable_set=True
                self.indicies=TS.get_indicies(atom_names)
                if TS.charge_and_mass == True:
                    self.charges=TS.get_charges(atom_names)
                    self.masses=TS.get_masses(atom_names)
        def position_centre(self,species_number):
            if species_number < len (self.indicies):
                self.species_number=species_number
            else:
                print("species_number out of range")

            
                    

    class HelperFunctions:

        def process_frame(TS,GS, positions, forces,a=1,kernel='triangular'):
            #"""TODO""" 
            GS.count += 1

            # correct for the rare ocassion where the centre of mass is in a different image from oxygen create an array in a singular direction
            homeZ = np.remainder(positions[:,2], TS.box_z)
            homeY = np.remainder(positions[:,1], TS.box_y)
            homeX = np.remainder(positions[:,0], TS.box_x)

            # Assigning each atom to a linearised grid point. 
            # The linearised grid works by first moving along the x axis.
            # On completion of a row we move up one step on the y axis. 
            # This process continues till y bins are exhausted and we move up a step on the z axis.

            #Force storage array for each orthogonal force componant
            #Forces in each direction
            fox = forces[:,0] # generate singular array for force in each direction 
            foy = forces[:,1] # generate singular array for force in each direction 
            foz = forces[:,2] # generate singular array for force in each direction 

            # use digitise to get atom grid indicies for the atoms in each direction 
            # https://docs.scipy.org/doc/numpy/reference/generated/numpy.digitize.html
            #Number density storage array
            x = np.digitize(homeX, GS.binsx)
            y = np.digitize(homeY, GS.binsy)
            z = np.digitize(homeZ, GS.binsz)

            if kernel.lower() == "triangular":
                RevelsMD3D.HelperFunctions.triangular_allocation(GS,x,y,z,homeX,homeY,homeZ,fox,foy,foz,a)

            elif kernel.lower() == "box":
                RevelsMD3D.HelperFunctions.box_allocation(GS,x,y,z,fox,foy,foz,a)


        def box_allocation(GS,x,y,z,fox,foy,foz,a):

            ##### RAPID GRIDDING METHOD WILL ONLY WORK WHEN ONLY ONE PARTICLE IS IN EACH VOXEL 
            ##### FOR MEANINGFUL GRIDS THIS ALWAYS THE CASE
            ##### BUT KEEP IN MIND

            # TODO Add a runtime check that this condition is True.
            x-=1
            y-=1
            z-=1
            # Force in x 
            GS.forceX[x, y ,z] += fox * a            
            # Force in y
            GS.forceY[x, y ,z] += foy * a
            # Force in z
            GS.forceZ[x, y ,z] += foz * a
            # conventional histogram box kernel
            GS.counter[x, y ,z] += a
        
        def triangular_allocation(GS,x,y,z,homeX,homeY,homeZ,fox,foy,foz,a):
                      # For triangular kernel we define grid directions
            fracx = 1 + ((homeX - (x * GS.lx)) / GS.lx)
            fracy = 1 + ((homeY - (y * GS.ly)) / GS.ly)
            fracz = 1 + ((homeZ - (z * GS.lz)) / GS.lz)

            # Calculate the fraction of each atom on each vertex
            f_000 = (1-fracx) * (1-fracy) * (1-fracz)
            f_001 = (1-fracx) * (1-fracy) * fracz
            f_010 = (1-fracx) * fracy * (1-fracz)
            f_100 = fracx * (1-fracy) * (1-fracz)
            f_101 = fracx * (1-fracy) * fracz
            f_011 = (1-fracx) * fracy * fracz
            f_110 = fracx * fracy * (1-fracz)
            f_111 = fracx * fracy *fracz

            # Generate indicies of the eight neighbouring vertices for each atom.
            # calculating the indicies using remainders to rehome outliers in the grids
            gx = ((x-1)%GS.nbinsx, x%GS.nbinsx)
            gy = ((y-1)%GS.nbinsy, y%GS.nbinsy)
            gz = ((z-1)%GS.nbinsz, z%GS.nbinsz)

            ##### RAPID GRIDDING METHOD WILL ONLY WORK WHEN ONLY ONE PARTICLE IS IN EACH VOXEL 
            ##### FOR MEANINGFUL GRIDS THIS ALWAYS THE CASE
            ##### BUT KEEP IN MIND

            # TODO Add a runtime check that this condition is True.
            # Force in x 
            # calculated for each vertex from the index list
            # calculating the grid point fractions by calculating the fraction across the voxels in each cartesian direction
            GS.forceX[gx[0], gy[0], gz[0]] += fox * f_000 * a
            GS.forceX[gx[0], gy[0], gz[1]] += fox * f_001 * a
            GS.forceX[gx[0], gy[1], gz[0]] += fox * f_010 * a
            GS.forceX[gx[1], gy[0], gz[0]] += fox * f_100 * a
            GS.forceX[gx[1], gy[0], gz[1]] += fox * f_101 * a
            GS.forceX[gx[0], gy[1], gz[1]] += fox * f_011 * a
            GS.forceX[gx[1], gy[1], gz[0]] += fox * f_110 * a
            GS.forceX[gx[1], gy[1], gz[1]] += fox * f_111 * a
            
            
            # Force in y
            # calculated for each vertex from the index list
            # calculating the grid point fractions by calculating the fraction across the voxels in each cartesian direction
            GS.forceY[gx[0], gy[0], gz[0]] += foy * f_000 * a
            GS.forceY[gx[0], gy[0], gz[1]] += foy * f_001 * a
            GS.forceY[gx[0], gy[1], gz[0]] += foy * f_010 * a
            GS.forceY[gx[1], gy[0], gz[0]] += foy * f_100 * a
            GS.forceY[gx[1], gy[0], gz[1]] += foy * f_101 * a
            GS.forceY[gx[0], gy[1], gz[1]] += foy * f_011 * a
            GS.forceY[gx[1], gy[1], gz[0]] += foy * f_110 * a
            GS.forceY[gx[1], gy[1], gz[1]] += foy * f_111 * a

            # Force in z
            # calculated for each vertex from the index list
            # calculating the grid point fractions by calculating the fraction across the voxels in each cartesian direction
            GS.forceZ[gx[0], gy[0], gz[0]] += foz * f_000 * a
            GS.forceZ[gx[0], gy[0], gz[1]] += foz * f_001 * a
            GS.forceZ[gx[0], gy[1], gz[0]] += foz * f_010 * a
            GS.forceZ[gx[1], gy[0], gz[0]] += foz * f_100 * a
            GS.forceZ[gx[1], gy[0], gz[1]] += foz * f_101 * a
            GS.forceZ[gx[0], gy[1], gz[1]] += foz * f_011 * a
            GS.forceZ[gx[1], gy[1], gz[0]] += foz * f_110 * a
            GS.forceZ[gx[1], gy[1], gz[1]] += foz * f_111 * a

            # conventional histogram triangular kernel
            # calculated for each vertex from the index list
            # calculating the grid point fractions by calculating the fraction across the voxels in each cartesian direction
            GS.counter[gx[0], gy[0], gz[0]] += f_000 * a
            GS.counter[gx[0], gy[0], gz[1]] += f_001 * a
            GS.counter[gx[0], gy[1], gz[0]] += f_010 * a
            GS.counter[gx[1], gy[0], gz[0]] += f_100 * a
            GS.counter[gx[1], gy[0], gz[1]] += f_101 * a
            GS.counter[gx[0], gy[1], gz[1]] += f_011 * a
            GS.counter[gx[1], gy[1], gz[0]] += f_110 * a
            GS.counter[gx[1], gy[1], gz[1]] += f_111 * a               

        def rigid_check(SS):
            if SS.indistinguishable_set == False:
                for element in SS.indicies[1:]:
                    failure_state=1
                    failure_state*=SS.indicies[1]==element
                    if failure_state != 1:
                        print("error: all atom types in a rigid molecule must have the same length")
                        return False
                    return True

        def find_coms(positons,TS,GS,SS,calc_dipoles=False):
            mass_tot=SS.masses[0]
            mass_cumulant=positons[SS.indicies[0]]*SS.masses[0][:,np.newaxis]
            for species_index in range(1,len(SS.indicies)):
                diffs = positons[SS.indicies[0]]-positons[SS.indicies[species_index]]
                logical_diffs = np.transpose(np.array([TS.box_x*(diffs[:,0]<-TS.box_x/2) - TS.box_x*(diffs[:,0]>TS.box_x/2),TS.box_y*(diffs[:,1]<-TS.box_y/2) - TS.box_y*(diffs[:,1]>TS.box_y/2),TS.box_z*(diffs[:,2]<-TS.box_z/2) - TS.box_z*(diffs[:,2]>TS.box_z/2)]))
                diffs += logical_diffs
                mass_tot+=SS.masses[species_index]
                mass_cumulant+=positons[SS.indicies[species_index]]*SS.masses[species_index][:,np.newaxis]
            coms=mass_cumulant/mass_tot[:,np.newaxis]
            if calc_dipoles:
                charges=GS.SS.charges[0]
                charges_cumulant=charges[:,np.newaxis]*(positons[SS.indicies[0]]-coms)
                for species_index in range(1,len(SS.indicies)):
                    seperation=(positons[SS.indicies[species_index]]-coms)
                    seperation[:,0]-= (np.ceil((np.abs(seperation[:,0])-TS.box_x/2)/TS.box_x))*((TS.box_x))*np.sign(seperation[:,0])
                    seperation[:,1]-= (np.ceil((np.abs(seperation[:,1])-TS.box_y/2)/TS.box_y))*((TS.box_y))*np.sign(seperation[:,1])
                    seperation[:,2]-= (np.ceil((np.abs(seperation[:,2])-TS.box_z/2)/TS.box_z))*((TS.box_z))*np.sign(seperation[:,2])
                    charges_cumulant+=charges[species_index]*seperation
                    molecular_dipole=charges_cumulant
                    return coms, molecular_dipole
            else:
                return coms

        def sum_forces(SS,forces):
            rigid_forces = forces[SS.indicies[0],:]
            for rigid_body_componant in SS.indicies[1:]:
                rigid_forces += forces[rigid_body_componant,:]
            return rigid_forces