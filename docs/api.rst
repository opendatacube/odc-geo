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


odc.geo.geom
************

.. currentmodule:: odc.geo.geom
.. autosummary::
   :toctree: _api/

   Geometry
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

