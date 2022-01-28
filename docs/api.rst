.. _api-reference:

API Reference
#############

.. highlight:: python


odc.geo.crs
***********

.. currentmodule:: odc.geo.crs
.. autosummary::
   :toctree: _api/

   CRS

   CRSMismatchError
   norm_crs
   norm_crs_or_error
   crs_units_per_degree

.. include:: crs.rst

odc.geo.geobox
**************

.. currentmodule:: odc.geo.geobox
.. autosummary::
   :toctree: _api/

   GeoBox
   GeoboxTiles

   affine_transform_pix
   bounding_box_in_pixel_domain
   flipx
   flipy
   gbox_boundary
   geobox_intersection_conservative
   geobox_union_conservative
   pad
   pad_wh
   rotate
   scaled_down_geobox
   translate_pix
   zoom_out
   zoom_to

.. include:: geobox.rst


odc.geo.geom
************

.. currentmodule:: odc.geo.geom
.. autosummary::
   :toctree: _api/

   Geometry
   Geometry.to_crs
   Geometry.geojson
   BoundingBox

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

odc.geo.roi
***********

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
