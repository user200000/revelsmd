import numpy as np
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