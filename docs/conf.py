# Configuration file for the Sphinx documentation builder.
#
# For a full list of configuration options, see the documentation:
# http://www.sphinx-doc.org/en/master/usage/configuration.html


# Project information
# --------------------------------------------------

project = 'Apollo'
version = '0.2.0'
release = ''

copyright = '2018, Georgia Power Company'
author = 'Chris Barrick, Zach Jones, Fred Maier'


# Configuration
# --------------------------------------------------

needs_sphinx = '1.7'  # v1.7.0 was released 2018-02-12
master_doc = 'index'
language = 'en'
pygments_style = 'sphinx'

templates_path = ['_templates']
source_suffix = ['.rst']
exclude_patterns = ['_build', '_static', 'Thumbs.db', '.DS_Store']

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.coverage',
    'sphinx.ext.doctest',
    'sphinx.ext.githubpages',
    'sphinx.ext.ifconfig',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
]


# Theme
# --------------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_logo = '_static/logo/apollo-logo-text-color.svg'
html_static_path = ['_static']
html_css_files = ['css/overrides.css']

# Theme specific,
# see https://sphinx-rtd-theme.readthedocs.io/en/latest/configuring.html
html_theme_options = {
    'logo_only': True,
    'display_version': True,
    'style_nav_header_background': '#EEEEEE',

    # Sidebar
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 3,
    'includehidden': True,
    'titles_only': False
}


# Extension: sphinx.ext.intersphinx
# --------------------------------------------------

# A mapping:  id -> (target, invintory)
# where  target  is the base URL of the target documentation,
# and  invintory  is the name of the inventory file, or  None  for the default.
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://docs.scipy.org/doc/numpy/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference/', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
    'matplotlib': ('https://matplotlib.org/', None),
    'xarray': ('http://xarray.pydata.org/en/stable/', None),
    'sklearn': ('http://scikit-learn.org/stable', None),
}


# Extension: sphinx.ext.autodoc
# --------------------------------------------------

autodoc_default_options = {
    'members': True,
}


# Extension: sphinx.ext.autosummary
# --------------------------------------------------

autosummary_generate = True
autosummary_generate_overwrite = True


# Extension: sphinx.ext.napoleon
# --------------------------------------------------

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_param = False
napoleon_use_rtype = False


# Extension: sphinx.ext.todo
# --------------------------------------------------

# Toggle output for  ..todo::  and  ..todolist::
todo_include_todos = True


# Path setup
# --------------------------------------------------
# All extensions and modules to document with autodoc must be in sys.path.

def add_path(path):
    '''Add a directory to the import path, relative to the documentation root.
    '''
    import os
    import sys
    path = os.path.abspath(path)
    sys.path.insert(0, path)


add_path('..')  # The root of the repo, puts the `apollo` package on the path.
