.. _api-reference:

API Reference
#############

.. highlight:: python

odc.geo.xr
**********

Interfacing with :py:class:`xarray.DataArray` and :py:class:`xarray.Dataset` classes.

.. currentmodule:: odc.geo.xr
.. autosummary::
   :toctree: _api/

   ODCExtension
   ODCExtension.geobox
   ODCExtension.output_geobox
   ODCExtension.spatial_dims
   ODCExtension.crs
   ODCExtension.map_bounds
   ODCExtension.crop
   ODCExtension.mask
   ODCExtension.explore

   ODCExtensionDa
   ODCExtensionDa.assign_crs
   ODCExtensionDa.reload
   ODCExtensionDa.write_cog
   ODCExtensionDa.to_cog
   ODCExtensionDa.reproject
   ODCExtensionDa.colorize
   ODCExtensionDa.add_to
   ODCExtensionDa.compress
   ODCExtensionDa.ydim
   ODCExtensionDa.xdim
   ODCExtensionDa.nodata

   ODCExtensionDs
   ODCExtensionDs.to_rgba
   ODCExtensionDs.reload

   assign_crs
   rasterize
   spatial_dims
   wrap_xr
   xr_coords
   xr_reproject
   xr_zeros
   colorize
   to_rgba
   add_to
   rio_reproject
   to_cog
   write_cog
   compress
   mask
   crop

odc.geo.cog
***********

Cloud Optimized GeoTIFF construction

.. currentmodule:: odc.geo.cog
.. autosummary::
   :toctree: _api/

   save_cog_with_dask
   cog_gbox
   to_cog
   write_cog
   write_cog_layers

odc.geo.geobox
**************

Methods for creating and manipulating geoboxes.

.. include:: api-geobox.rst

odc.geo.crs
***********

Contains CRS class and related operations.

.. currentmodule:: odc.geo.crs
.. autosummary::
   :toctree: _api/

   CRS
   CRS.utm
   CRS.to_epsg
   CRS.to_wkt
   CRS.transformer_to_crs
   CRS.authority

   CRS.units
   CRS.dimensions

   CRSMismatchError

   norm_crs
   norm_crs_or_error
   crs_units_per_degree


odc.geo.geom
************

Shapely geometry classes with CRS information attached.

.. currentmodule:: odc.geo.geom
.. autosummary::
   :toctree: _api/

   Geometry
   Geometry.to_crs
   Geometry.geojson
   Geometry.explore
   BoundingBox
   BoundingBox.explore
   BoundingBox.from_xy
   BoundingBox.from_points
   BoundingBox.from_transform
   BoundingBox.transform
   BoundingBox.buffered
   BoundingBox.boundary
   BoundingBox.qr2sample

   box
   line
   point
   polygon
   multigeom
   multiline
   multipoint
   multipolygon
   polygon_from_transform

   chop_along_antimeridian
   clip_lon180
   common_crs
   densify
   intersects
   lonlat_bounds
   mid_longitude
   projected_lon
   sides

   bbox_intersection
   bbox_union
   unary_intersection
   unary_union
   triangulate

odc.geo
*******

Basic types used for unambigous specification of X/Y values.

.. currentmodule:: odc.geo
.. autosummary::
   :toctree: _api/

   XY
   XY.xy
   XY.yx
   XY.lonlat
   XY.latlon
   XY.x
   XY.y
   XY.lon
   XY.lat
   XY.wh
   XY.shape

   Resolution
   Shape2d
   Index2d
   AnchorEnum

   xy_
   yx_

   res_
   resxy_
   resyx_

   wh_
   shape_

   ixy_
   iyx_


odc.geo.roi
***********

Various helper methods for working with 2d slices into arrays.

.. currentmodule:: odc.geo.roi
.. autosummary::
   :toctree: _api/

   polygon_path
   roi_boundary
   roi_center
   roi_from_points
   roi_intersect
   roi_is_empty
   roi_is_full
   roi_normalise
   roi_pad
   roi_shape
   scaled_down_roi
   scaled_down_shape
   scaled_up_roi

odc.geo.math
************

.. currentmodule:: odc.geo.math
.. autosummary::
   :toctree: _api/

   Bin1D
   Bin1D.bin
   Bin1D.from_sample_bin

   Poly2d
   Poly2d.fit
   Poly2d.grid2d
   Poly2d.with_input_transform

   affine_from_axis
   align_down
   align_up
   apply_affine
   clamp
   data_resolution_and_offset
   edge_index
   is_affine_st
   is_almost_int
   maybe_int
   maybe_zero
   norm_xy
   snap_scale
   snap_affine
   split_float
   split_translation

   quasi_random_r2

odc.geo.gridspec
****************

.. currentmodule:: odc.geo.gridspec
.. autosummary::
   :toctree: _api/

   GridSpec
   GridSpec.from_sample_tile
   GridSpec.web_tiles

   GridSpec.alignment
   GridSpec.dimensions
   GridSpec.tile_shape

   GridSpec.pt2idx
   GridSpec.tile_geobox
   GridSpec.__getitem__
   GridSpec.tiles
   GridSpec.tiles_from_geopolygon

   GridSpec.geojson

odc.geo.overlap
***************

.. currentmodule:: odc.geo.overlap
.. autosummary::
   :toctree: _api/

   ReprojectInfo
   affine_from_pts
   compute_output_geobox
   compute_reproject_roi

odc.geo.converters
******************

.. currentmodule:: odc.geo.converters
.. autosummary::
   :toctree: _api/

   from_geopandas
   extract_gcps_raw
   extract_gcps_raw
   map_crs
   rio_geobox
