GeoBox
======

.. currentmodule:: odc.geo.geobox
.. autosummary::
   :toctree: _api/

   GeoBox
   GeoBox.from_geopolygon
   GeoBox.from_bbox
   GeoBox.affine
   GeoBox.anchor
   GeoBox.boundary
   GeoBox.buffered
   GeoBox.coordinates
   GeoBox.crs
   GeoBox.dimensions
   GeoBox.dims
   GeoBox.flipx
   GeoBox.flipy
   GeoBox.extent
   GeoBox.explore
   GeoBox.footprint
   GeoBox.boundingbox
   GeoBox.map_bounds
   GeoBox.geographic_extent
   GeoBox.overlap_roi
   GeoBox.height
   GeoBox.is_empty
   GeoBox.left
   GeoBox.right
   GeoBox.top
   GeoBox.bottom
   GeoBox.pad
   GeoBox.pad_wh
   GeoBox.resolution
   GeoBox.aspect
   GeoBox.rotate
   GeoBox.shape
   GeoBox.transform
   GeoBox.translate_pix
   GeoBox.snap_to
   GeoBox.enclosing
   GeoBox.to_crs
   GeoBox.width
   GeoBox.zoom_out
   GeoBox.zoom_to
   GeoBox.qr2sample
   GeoBox.__mul__
   GeoBox.__rmul__


GeoboxTiles
===========

.. currentmodule:: odc.geo.geobox
.. autosummary::
   :toctree: _api/

   GeoboxTiles
   GeoboxTiles.base
   GeoboxTiles.chunk_shape
   GeoboxTiles.chunks
   GeoboxTiles.range_from_bbox
   GeoboxTiles.shape
   GeoboxTiles.tiles
   GeoboxTiles.grid_intersect
   GeoboxTiles.roi


Standalone Methods
==================

.. currentmodule:: odc.geo.geobox
.. autosummary::
   :toctree: _api/

   bounding_box_in_pixel_domain
   geobox_intersection_conservative
   geobox_union_conservative
   scaled_down_geobox

   affine_transform_pix
   flipx
   flipy
   gbox_boundary
   pad
   pad_wh
   rotate
   translate_pix
   zoom_out
   zoom_to
