import numpy as np
from tqdm import tqdm
import MDAnalysis as MD
from lxml import etree # type: ignore
from typing import List, Union, Optional, Any, Dict
from pymatgen.core import Structure, Lattice
from pymatgen.io.ase import AseAtomsAdaptor
from ase.io.cube import write_cube
import copy
from revelsmd.revels_tools.lammps_parser import define_strngdex,frame_skip,get_a_frame
from revelsmd.revels_tools.conversion_factors import generate_boltzmann

class Revels3D:
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
            self.box_x=TS.box_x
            self.box_y=TS.box_y
            self.box_z=TS.box_z
            self.box_array =np.array([TS.box_x,TS.box_y,TS.box_z])
            self.binsx=np.arange(0,TS.box_x+lx,lx)
            self.binsy=np.arange(0,TS.box_y+ly,ly)
            self.binsz=np.arange(0,TS.box_z+lz,lz)
            self.voxel_volume=np.prod(self.box_array)/np.prod([nbinsx,nbinsy,nbinsz])
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
            self.SS=Revels3D.SelectionState(TS,atom_names=atom_names,centre_location=centre_location)
            if self.density_type.lower() == 'number':
                if self.SS.indistinguishable_set==False:
                    if rigid == True:
                        if centre_location == True:
                            self.single_frame_function = Revels3D.Estimators.single_frame_rigid_number_com_grid
                        elif type(centre_location) is int:
                            self.single_frame_function = Revels3D.Estimators.single_frame_rigid_number_atom_grid
                        else:
                            print("error centre location must be True (com) or int (specific atom)")
                    else:
                        self.single_frame_function =  Revels3D.Estimators.single_frame_number_many_grid
                else:
                    self.single_frame_function = Revels3D.Estimators.single_frame_number_single_grid
            elif self.density_type.lower() == "charge":
                if self.SS.indistinguishable_set==False:
                    if rigid:
                        if centre_location == True:
                            self.single_frame_function = Revels3D.Estimators.single_frame_rigid_charge_com_grid
                        elif type(centre_location) is int:
                            self.single_frame_function = Revels3D.Estimators.single_frame_rigid_charge_atom_grid
                        else:
                            print("error centre location must be True (com) or int (specific atom)")
                    else:
                        self.single_frame_function = Revels3D.Estimators.single_frame_charge_many_grid
                else:
                    self.single_frame_function = Revels3D.Estimators.single_frame_number_single_grid
                
            elif self.density_type.lower() == "polarisation":
                if self.SS.indistinguishable_set==False:
                    if rigid:
                        if centre_location == True:
                            self.single_frame_function = Revels3D.Estimators.single_frame_rigid_polarisation_com_grid
                            self.SS.polarisation_axis=polarisation_axis
                        elif type(centre_location) is int:
                            self.single_frame_function = Revels3D.Estimators.single_frame_rigid_polarisation_atom_grid
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
                stringdex=define_strngdex(neededQuantities,TS.dic)

                for frame_count in tqdm(self.to_run):
                    vars_trest=get_a_frame(f,TS.num_ats,TS.header_length,stringdex)
                    self.single_frame_function(vars_trest[:,:3],vars_trest[:,3:],TS,self,self.SS,kernel=self.kernel)
                    frame_skip(f,TS.num_ats,period-1,TS.header_length)
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
                forceX=np.fft.fftn(self.forceX/self.count/self.voxel_volume)
                forceY=np.fft.fftn(self.forceY/self.count/self.voxel_volume)
                forceZ=np.fft.fftn(self.forceZ/self.count/self.voxel_volume)
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
                self.del_rho_k = (complex(0,1) / (self.temperature*generate_boltzmann(self.units)*self.get_ksquared()) * (forceX + forceY + forceZ))
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
                self.particle_density=self.counter/self.voxel_volume/self.count
        
        def get_kvectors(self):
            '''
            Generates kvectors for a cubic cell
            '''
            xrep = 2*np.pi*np.fft.fftfreq(self.nbinsx, d=self.lx)
            yrep = 2*np.pi*np.fft.fftfreq(self.nbinsy, d=self.ly)
            zrep = 2*np.pi*np.fft.fftfreq(self.nbinsz, d=self.lz)
            return xrep, yrep, zrep
        


        def get_ksquared(self):
            '''
            Generates the ksquared array for a cubic cell, the code does this only when needed due to memory conservation
            '''
            xrep, yrep, zrep = self.get_kvectors()
            # Propagation to 2D
            xrep = np.repeat(xrep[:, np.newaxis, np.newaxis], self.nbinsy, axis=1)
            zrep = np.repeat(zrep[np.newaxis, np.newaxis, :], self.nbinsx, axis=0)
            yrep = np.repeat(yrep[np.newaxis, :, np.newaxis], self.nbinsx, axis=0)
    
            # Propagation to 3D
            xrep = np.repeat(xrep[:, :, :], self.nbinsz, axis=2)
            yrep = np.repeat(yrep[:, :, :], self.nbinsz, axis=2)
            zrep = np.repeat(zrep[:, :, :], self.nbinsy, axis=1)
            return (xrep*xrep)+(yrep*yrep)+(zrep*zrep) # generate the square of kvectors
        
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
                    stringdex=define_strngdex(neededQuantities,TS.dic)
                    for frame_count in (GS_Lambda.to_run[np.arange(k,len(GS_Lambda.to_run)//sections,sections)]):
                        vars_trest=get_a_frame(f,TS.num_ats,TS.header_length,stringdex)
                        GS_Lambda.single_frame_function(vars_trest[:,:3],vars_trest[:,3:],TS,GS_Lambda,GS_Lambda.SS,kernel=GS_Lambda.kernel)
                        frame_skip(f,TS.num_ats,period-1,TS.header_length)
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
            coms = Revels3D.HelperFunctions.find_coms(positions,TS,GS,SS)
            rigid_forces = Revels3D.HelperFunctions.sum_forces(SS,forces)
            Revels3D.HelperFunctions.process_frame(TS,GS,coms,rigid_forces,kernel=kernel)

        
        def single_frame_rigid_number_atom_grid(positions,forces,TS,GS,SS,kernel='triangular'):
        
            rigid_forces = Revels3D.HelperFunctions.sum_forces(SS,forces)
            Revels3D.HelperFunctions.process_frame(TS,GS,positions[SS.indicies[SS.centre_location],:],rigid_forces,kernel=kernel)

        def single_frame_number_many_grid(positions,forces,TS,GS,SS,kernel='triangular'):
        
            for count in range(len(SS.indicies)):
                Revels3D.HelperFunctions.process_frame(TS,GS,positions[SS.indicies[count],:],forces[SS.indicies[count],:],kernel=kernel)

        def single_frame_number_single_grid(positions,forces,TS,GS,SS,kernel='triangular'):
            Revels3D.HelperFunctions.process_frame(TS,GS,positions[SS.indicies,:],forces[SS.indicies,:],kernel=kernel)

        def single_frame_rigid_charge_atom_grid(positions,forces,TS,GS,SS,kernel='triangular'):
            rigid_forces=Revels3D.HelperFunctions.sum_forces(SS,forces)
            Revels3D.HelperFunctions.process_frame(TS,GS,positions[SS.indicies[SS.centre_location],:],rigid_forces,a=SS.charges[SS.centre_location],kernel=kernel)

        def single_frame_charge_many_grid(positions,forces,TS,GS,SS,kernel='triangular'):
            for count in range(len(SS.indicies)):
                Revels3D.HelperFunctions.process_frame(TS,GS,positions[SS.indicies[count],:],forces[SS.indicies[count],:],a=SS.charges[count],kernel=kernel)

        def single_frame_charge_single_grid(positions,forces,TS,GS,SS,kernel='triangular'):
            Revels3D.HelperFunctions.process_frame(TS,GS,positions[SS.indicies,:],forces[SS.indicies,:],a=SS.charges,kernel=kernel)

        def single_frame_rigid_charge_com_grids(positions,forces,TS,GS,SS,kernel='triangular'):
            coms = Revels3D.HelperFunctions.find_coms(positions,TS,GS,SS)
            rigid_forces = Revels3D.HelperFunctions.sum_forces(SS,forces)
            Revels3D.HelperFunctions.process_frame(TS,GS,coms,rigid_forces,kernel=kernel,a=SS.charges)

        def single_frame_rigid_polarisation_com_grid(positions,forces,TS,GS,SS,kernel='triangular'):
            coms,molecular_dipole = Revels3D.HelperFunctions.find_coms(positions,TS,GS,SS,calc_dipoles=True)
            rigid_forces = Revels3D.HelperFunctions.sum_forces(SS,forces)
            Revels3D.HelperFunctions.process_frame(TS,GS,coms,rigid_forces,a=molecular_dipole[:,GS.SS.polarisation_axis],kernel=kernel)


        def single_frame_rigid_polarisation_atom_grid(positions,forces,TS,GS,SS,kernel='triangular'):
            coms,molecular_dipole = Revels3D.HelperFunctions.find_coms(positions,TS,GS,SS,calc_dipoles=True)
            rigid_forces = Revels3D.HelperFunctions.sum_forces(SS,forces)
            Revels3D.HelperFunctions.process_frame(TS,GS,coms,rigid_forces,a=molecular_dipole[:,GS.SS.polarisation_axis],kernel=kernel)
            
            
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
            homeZ = np.remainder(positions[:,2], GS.box_z)
            homeY = np.remainder(positions[:,1], GS.box_y)
            homeX = np.remainder(positions[:,0], GS.box_x)

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
                Revels3D.HelperFunctions.triangular_allocation(GS,x,y,z,homeX,homeY,homeZ,fox,foy,foz,a)

            elif kernel.lower() == "box":
                Revels3D.HelperFunctions.box_allocation(GS,x,y,z,fox,foy,foz,a)


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
