# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath(".."))
import odc.geo

# Work-around for type annotation failing due to guarded imports.
# Some methods in crs return geom types, but geom depends on crs
# so geom import into crs is only done during typechecking.
# by patching it here we enable typehinting to find correct types
# for crs methods that return geometries.
import odc.geo.crs
import odc.geo.geom

odc.geo.crs.geom = odc.geo.geom

# autotypes work around 2
import odc.geo.overlap

odc.geo.overlap.CRS = odc.geo.crs.CRS

# Try to force font-cache warning to happen earlier
from matplotlib import pyplot as plt

# -- Project information -----------------------------------------------------

project = "odc-geo"
copyright = "2022, ODC"
author = "ODC"

# The full version, including alpha/beta/rc tags
release = odc.geo.__version__
version = ".".join(release.split(".", 2)[:2])


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx_autodoc_typehints",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.extlinks",
    "sphinx.ext.mathjax",
    "jupyter_sphinx",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- copied from odc-stac mostly
# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "friendly"

autosummary_generate = True

extlinks = {
    "issue": ("https://github.com/opendatacube/odc-geo/issues/%s", "issue %s"),
    "pull": ("https://github.com/opendatacube/odc-geo/pulls/%s", "PR %s"),
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pyproj": ("https://pyproj4.github.io/pyproj/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "xarray": ("https://xarray.pydata.org/en/stable/", None),
    "rasterio": ("https://rasterio.readthedocs.io/en/latest/", None),
    "shapely": ("https://shapely.readthedocs.io/en/latest/", None),
    "folium": ("https://python-visualization.github.io/folium/latest/", None),
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

html_theme_options = {
    "collapse_navigation": False,
    "logo_only": True,
}

# html_logo = '_static/logo.svg'
html_last_updated_fmt = "%b %d, %Y"
html_show_sphinx = False

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_css_files = ["xr-fixes.css"]
