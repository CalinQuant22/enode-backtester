# Configuration file for the Sphinx documentation builder.

# -- Path setup --------------------------------------------------------------

import os
import sys

# Add project root (one level above docs/) to the Python path
# PROJECT TREE:
# project/
#   backtester/
#   docs/
#       source/conf.py   <-- this file
sys.path.insert(0, os.path.abspath("../.."))


# -- Project information -----------------------------------------------------

project = "enode-backtester"
copyright = "2025, Calin Ignat, Oscar Thiele Serrano"
author = "Calin Ignat, Oscar Thiele Serrano"
release = "0.1.0"


# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx_autodoc_typehints",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = []

# Allow .md (Markdown) files
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# Napoleon settings (good defaults)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True

# Autodoc settings
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# Mock imports for modules that might not be available during doc build
autodoc_mock_imports = [
    'pandas', 'numpy', 'pyfolio', 'pydantic',
    'dash', 'plotly', 'dash_bootstrap_components',
    'plotly.graph_objects', 'plotly.express'
]


# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
