#! /usr/bin/env python
"""
setup.py for revelsmd

@author: Samuel W Coles (swc57@bath.ac.uk)
"""

# System imports
import io
from os import path
from setuptools import setup, find_packages
from revelsmd import __version__

PACKAGES = find_packages(exclude=['tests*'])

# versioning

ISRELEASED = False
VERSION = __version__


THIS_DIRECTORY = path.abspath(path.dirname(__file__))
with io.open(path.join(THIS_DIRECTORY, 'README.md')) as f:
    LONG_DESCRIPTION = f.read()

INFO = {
        'name': 'revelsmd',
        'description': 'Reduced varience calculations from'
                       'molecular dynamics using force based estimators.',
        'author': 'Samuel W. Coles',
        'author_email': 'swc57@bath.ac.uk',
        'packages': PACKAGES,
        'include_package_data': True,
        'setup_requires': ['numpy', 'ase','pymatgen',
                           'scipy>=1.9.3', 'tqdm','lxml','MDanalysis>=2.4.2','recommonmark'],
        'install_requires': ['numpy', 'ase','pymatgen',
                           'scipy>=1.9.3', 'tqdm','lxml','MDanalysis>=2.4.2','recommonmark'],
        'version': VERSION,
        'license': 'MIT',
        'long_description': LONG_DESCRIPTION,
        'long_description_content_type': 'text/markdown',
        'classifiers': ['Development Status :: 4 - Beta',
                        'Intended Audience :: Science/Research',
                        'License :: OSI Approved :: MIT License',
                        'Natural Language :: English',
                        'Operating System :: OS Independent',
                        'Programming Language :: Python :: 3.10',
                        'Topic :: Scientific/Engineering',
                        'Topic :: Scientific/Engineering :: Chemistry',
                        'Topic :: Scientific/Engineering :: Physics']
        }

####################################################################
# this is where setup starts
####################################################################


def setup_package():
    """
    Runs package setup
    """
    setup(**INFO)


if __name__ == '__main__':
    setup_package()
