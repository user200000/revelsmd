# Configuration file for the Sphinx documentation builder.
import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from revelsMD import *

# -- Project information -----------------------------------------------------
project = 'revelsMD'
copyright = '2025, Samuel Coles'
author = 'Samuel Coles'
release = '0.1'
version = '0.1.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'myst_nb',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output -------------------------------------------------
epub_show_urls = 'footnote'

# -- Autodoc settings --------------------------------------------------------
autodoc_typehints = 'description'
autosummary_generate = True
autodoc_member_order = 'bysource'
exclude_patterns = ['_build', '**.ipynb_checkpoints', 'Thumbs.db', '.DS_Store']
nbsphinx_execute = 'never'
source_suffix = ['.rst', '.ipynb', '.md']
nb_execution_mode = "off"
