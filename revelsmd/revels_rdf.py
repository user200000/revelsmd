import numpy as np
from revelsmd.revels_tools.lammps_parser import define_strngdex,frame_skip,get_a_frame
from revelsmd.revels_tools.conversion_factors import generate_boltzmann
from tqdm import tqdm
class RevelsRDF:
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
            single_frame_function = RevelsRDF.single_frame_rdf_like
            indicies = TS.get_indicies(atom_a)
            prefactor= float(TS.box_x*TS.box_y*TS.box_z)/(float(len(indicies))*float(len(indicies)-1))

        else:
            indicies = [np.array(TS.get_indicies(atom_a)),np.array(TS.get_indicies(atom_b))]
            single_frame_function = RevelsRDF.single_frame_rdf_unlike
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
            stringdex=define_strngdex(neededQuantities,TS.dic)
            if rmax:
                bins= np.arange(0,np.max([TS.box_x/2,TS.box_y/2,TS.box_z/2]),delr)
            else:
                bins= np.arange(0,float(rmax),delr)
            accumulated_storage_array=np.zeros(np.size(bins), dtype=np.longdouble)
            for frame_count in tqdm(to_run):
                vars_trest=get_a_frame(f,TS.num_ats,TS.header_length,stringdex)
                accumulated_storage_array+=single_frame_function(vars_trest[:,:3],vars_trest[:,3:],indicies,TS.box_x,TS.box_y,TS.box_z,bins)
                frame_skip(f,TS.num_ats,period-1,TS.header_length)
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
        accumulated_storage_array*=prefactor/(4*np.pi*len(to_run)*generate_boltzmann(TS.units)*temp)
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
            single_frame_function = RevelsRDF.single_frame_rdf_like
            indicies = TS.get_indicies(atom_a)
            prefactor= float(TS.box_x*TS.box_y*TS.box_z)/(float(len(indicies))*float(len(indicies)-1))

        else:
            indicies = [TS.get_indicies(atom_a),TS.get_indicies(atom_b)]
            single_frame_function = RevelsRDF.single_frame_rdf_unlike
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
            stringdex=define_strngdex(neededQuantities,TS.dic)
            if rmax:
                bins= np.arange(0,np.max([TS.box_x/2,TS.box_y/2,TS.box_z/2]),delr)
            else:
                bins= np.arange(0,float(rmax),delr)
            accumulated_storage_array=np.zeros(np.size(bins), dtype=np.longdouble)
            for frame_count in tqdm(to_run):
                vars_trest=get_a_frame(f,TS.num_ats,TS.header_length,stringdex)
                this_frame=single_frame_function(vars_trest[:,:3],vars_trest[:,3:],indicies,TS.box_x,TS.box_y,TS.box_z,bins)
                accumulated_storage_array+=this_frame
                list_store.append(this_frame)
                frame_skip(f,TS.num_ats,period-1,TS.header_length)
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
        base_array*=prefactor/(4*np.pi*generate_boltzmann(TS.units)*temp)
        accumulated_storage_array=np.nan_to_num(accumulated_storage_array)
        accumulated_storage_array*=prefactor/(4*np.pi*len(to_run)*generate_boltzmann(TS.units)*temp)
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
