Xarray Interop
##############

Importing :py:mod:`odc.geo.xr` enables ``.odc`` extension on :py:class:`xarray.DataArray` and
:py:class:`xarray.Dataset`. You can access geospatial information recorded on those arrays. In the
example below we make a sample array filled with zeros, but ``.odc.geobox`` will work on data loaded
by other libraries like :py:mod:`rioxarray` or :py:mod:`odc.stac`. One can also use
:py:meth:`odc.geo.xr.assign_crs` to inject geospatial information into arrays without one present.

.. jupyter-execute::

    from odc.geo.geobox import GeoBox
    from odc.geo.xr import ODCExtensionDa, xr_zeros

    xx = xr_zeros(
        GeoBox.from_bbox(
            (-2_000_000, -5_000_000, 2_250_000, -1_000_000), "epsg:3577", resolution=1000
        ),
        chunks=(1000, 1000),
    )
    assert isinstance(xx.odc, ODCExtensionDa)
    display(xx)
    display(xx.odc.geobox)
