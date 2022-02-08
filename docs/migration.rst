#####################
Changes from Datacube
#####################

Majority of the functionality that was present in
:py:mod:`datacube.utils.geometry` module remains available. However there are
some changes to the API and import locations are now more granular.

Solving X/Y vs Y/X Confusion
============================

Previously various X,Y properties were supplied in plain tuples. In some places
these would be expected in X,Y order, like a coordinate for example, and in
others Y,X order was used. Documentation as to what order was expected where was
usually present, but not quite everywhere, in some places you had to move along
the call stack to be certain what order is the right one.

In :py:mod:`odc.geo` this problem is addressed by introducing intermediate
representation :py:class:`~odc.geo.XY` and switching over methods that
previously accepted or produced plain tuples to instead operate with
:py:class:`~odc.geo.XY` family of classes.


Resolution
----------

Resolution parameter was probably the most likely to be problematic. Plain
tuples are no longer accepted. However a single number is accepted and implies
square pixels with inverted Y axis.

* If using square pixels with inverted Y axis use a single number instead of a tuple

  * Instead of ``resolution=(-10, 10)`` do ``resolution=10``

* For other cases use :py:func:`~odc.geo.resyx_` or :py:func:`~odc.geo.resxy_`
  to indicate supplied order at the call site

Shape
-----

* New class :py:class:`~odc.geo.Shape2d` is available and behaves like a normal
  tuple in most situations

* Plain tuples are still accepted and are expected to be in ``Y,X`` (``nrows,
  ncols``) order

* Methods that were previously accepting separate ``width:int, height:int``
  parameters were changed to expect a single ``shape`` parameter instead.

  * Use :py:func:`~odc.geo.wh_` to construct shape object from ``width, height``
    for minimal and safest change to call site code.

    .. code-block:: diff

       + from odc.geo import wh_
       - GeoBox(w, h, ...)
       + GeoBox(wh_(w, h), ...)

Points in 2D
------------

Things like ``origin`` and ``alignment`` can no longer be supplied as plain
tuples, use :py:func:`~odc.geo.xy_` or :py:func:`~odc.geo.yx_` as appropriate.
There are also :py:func:`~odc.geo.lonlat_` and :py:func:`~odc.geo.latlon_`
methods.

Index in 2D
-----------

Things like a pixel location or a tile index. Plain tuples are still accepted
and order is API dependent. But it is now possible to indicate at call site what
order is used with :py:func:`~odc.geo.ixy_` and :py:func:`~odc.geo.iyx_`.



:py:class:`~odc.geo.geobox.GeoBox`
==================================

* :py:class:`~odc.geo.geobox.GeoBox` now lives in a separate name-space from plain geometry classes
* Constructor changed from accepting ``GeoBox(width, height, ...)`` to ``GeoBox(shape, ...)``, where
  ``shape=(nrows, ncols)``.



:py:class:`~odc.geo.gridspec.GridSpec`
======================================

* Constructor changed

  * Tile size is now specified in pixels rather than CRS units

  * Tile size can not be negative, negative tile size was used to indicate tile
    index direction being opposite of axis direction

  * ``flipx,flipy`` parameters are now used to control tile index direction

  * ``origin`` parameter now always refers to the location of the bottom left
    corner of the tile with index ``0,0`` regardless of the tile index
    direction.

  * ``.tile_resolution`` is now called ``.tile_shape``, because that's what it is.

  * Removed under-defined ``tile_buffer=`` parameter from
    ``.tiles_from_geopolygon``. With this parameter supplied, returned tiles
    were of bigger size than specified in constructor, and would be overlapping,
    so not even tiles. It was not clear from the documentation and tests if that
    was deliberate, albeit confusing, choice or was just an error of
    implementation. You can achieve the same effect by buffering query polygon
    on input and then buffering output geoboxes on output. `Relevant commit`_.


* New convenience methods for construction:
  :py:func:`~odc.geo.gridspec.GridSpec.from_sample_tile`,
  :py:func:`~odc.geo.gridspec.GridSpec.web_tiles`

Main change is that grid is now specified in pixels rather than CRS units. This
is the only way to ensure that there are no gaps in the general case. It is now
easier to specify inverted tile indexes, like what is used by slippy map regime.
Previous mechanism of using negative tile sizes was not well documented or
tested, and it was not clear how you were supposed to anchor the grid with the
negative tile size, ``origin`` parameter was under-specified.


.. _`Relevant commit`: https://github.com/opendatacube/odc-geo/commit/d6aca737028fff55f92eced9cadbaa0d5b37199e
