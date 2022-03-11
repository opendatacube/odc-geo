Overview
########

This library combines geometry shape classes from shapely_ with CRS from pyproj_
to provide a number of data types and utilities useful for working with
geospatial metadata.

There is also integration with xarray_. Importing ``odc.geo.xr`` enables
``.odc.`` accessor on every :py:class:`xarray.Dataset` and
:py:class:`xarray.DataArray` that exposes geospatial information of the raster
loaded with `Open Datacube`_ or rioxarray_. Methods for attaching geospatial
information to xarray objects in a robust way are also provided. Geospatial
information attached in this way survives most operations you might do on the
data: basic mathematical operations, type conversions, cropping, serialization
to most formats like zarr, netcdf, GeoTIFF.


Installation
############

Using pip
*********

.. code-block:: bash

   pip install odc-geo

Using Conda
***********

It's not on ``conda-forge`` yet.

.. code-block:: bash

   conda install -c kirill-odc odc-geo


.. _rioxarray: https://corteva.github.io/rioxarray/stable/
.. _xarray: https://docs.xarray.dev/en/stable/
.. _shapely: https://shapely.readthedocs.io/en/stable/manual.html
.. _pyproj: https://pyproj4.github.io/pyproj/stable/
.. _`Open Datacube`: https://github.com/opendatacube/datacube-core
