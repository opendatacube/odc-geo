Overview
########

This library combines geometry shape classes from shapely_ with CRS from pyproj_ to provide
projection aware :py:class:`~odc.geo.geom.Geometry`. It exposes all the functionality provided by
:py:mod:`shapely` modules, but will refuse operations between geometries defined in different
projections. Geometries can be brought into a common projection with
:py:meth:`~odc.geo.geom.Geometry.to_crs` method.

Based on that foundation a number of data types and utilities useful for working with geospatial
metadata are implemented. Of particular importance is :py:class:`~odc.geo.geobox.GeoBox`. It is an
abstraction for a geo-registered bounded pixel plane where a linear mapping from pixel coordinates
to the real world is defined.

.. jupyter-execute::

   from odc.geo.geobox import GeoBox
   GeoBox.from_bbox(
      (-2_000_000, -5_000_000,
        2_250_000, -1_000_000),
      "epsg:3577", resolution=1000)

To make working with geo-registered raster data easier an integration with xarray_ is provided.
Importing ``odc.geo.xr`` enables ``.odc.`` accessor on every :py:class:`xarray.Dataset` and
:py:class:`xarray.DataArray` that exposes geospatial information of the raster loaded with `Open
Datacube`_ or rioxarray_. Methods for attaching geospatial information to xarray objects in a robust
way are also provided. Geospatial information attached in this way survives most operations you
might do on the data: basic mathematical operations, type conversions, cropping, serialization to
most formats like zarr, netcdf, GeoTIFF.

Installation
############

Using pip
*********

.. code-block:: bash

   pip install odc-geo

Using Conda
***********

.. code-block:: bash

   conda install -c conda-forge odc-geo


.. _rioxarray: https://corteva.github.io/rioxarray/stable/
.. _xarray: https://docs.xarray.dev/en/stable/
.. _shapely: https://shapely.readthedocs.io/en/stable/manual.html
.. _pyproj: https://pyproj4.github.io/pyproj/stable/
.. _`Open Datacube`: https://github.com/opendatacube/datacube-core
