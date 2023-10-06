import scipy.constants as constants
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
