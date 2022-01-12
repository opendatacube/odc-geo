# This file is part of the Open Data Cube, see https://opendatacube.org for more information
#
# Copyright (c) 2015-2020 ODC Contributors
# SPDX-License-Identifier: Apache-2.0
""" Geometric shapes and operations on them
"""
# import order is important here
#  _crs <-- _geom <-- _geobox <- other
# isort: skip_file

from ._version import __version__

from ._crs import (
    CRS,
    CRSError,
    CRSMismatchError,
    MaybeCRS,
    SomeCRS,
    crs_units_per_degree,
)

from ._geom import (
    BoundingBox,
    CoordList,
    Geometry,
    bbox_intersection,
    bbox_union,
    box,
    chop_along_antimeridian,
    clip_lon180,
    common_crs,
    intersects,
    line,
    lonlat_bounds,
    mid_longitude,
    multigeom,
    multiline,
    multipoint,
    multipolygon,
    point,
    polygon_from_transform,
    polygon,
    projected_lon,
    sides,
    unary_intersection,
    unary_union,
)

from ._geobox import (
    Coordinate,
    GeoBox,
    assign_crs,
    geobox_intersection_conservative,
    geobox_union_conservative,
    scaled_down_geobox,
)

from ._gridspec import GridSpec

from .tools import (
    affine_from_pts,
    apply_affine,
    compute_axis_overlap,
    compute_reproject_roi,
    decompose_rws,
    get_scale_at_point,
    is_affine_st,
    native_pix_transform,
    roi_boundary,
    roi_center,
    roi_from_points,
    roi_intersect,
    roi_is_empty,
    roi_is_full,
    roi_normalise,
    roi_pad,
    roi_shape,
    scaled_down_roi,
    scaled_down_shape,
    scaled_up_roi,
    split_translation,
    w_,
)
from ._warp import rio_reproject, warp_affine


__all__ = [
    "__version__",
    "affine_from_pts",
    "apply_affine",
    "assign_crs",
    "bbox_intersection",
    "bbox_union",
    "BoundingBox",
    "box",
    "chop_along_antimeridian",
    "clip_lon180",
    "common_crs",
    "compute_axis_overlap",
    "compute_reproject_roi",
    "Coordinate",
    "CoordList",
    "crs_units_per_degree",
    "CRS",
    "CRSError",
    "CRSMismatchError",
    "decompose_rws",
    "geobox_intersection_conservative",
    "geobox_union_conservative",
    "GeoBox",
    "Geometry",
    "get_scale_at_point",
    "GridSpec",
    "intersects",
    "is_affine_st",
    "line",
    "lonlat_bounds",
    "MaybeCRS",
    "mid_longitude",
    "multigeom",
    "multiline",
    "multipoint",
    "multipolygon",
    "native_pix_transform",
    "point",
    "polygon_from_transform",
    "polygon",
    "projected_lon",
    "rio_reproject",
    "roi_boundary",
    "roi_center",
    "roi_from_points",
    "roi_intersect",
    "roi_is_empty",
    "roi_is_full",
    "roi_normalise",
    "roi_pad",
    "roi_shape",
    "scaled_down_geobox",
    "scaled_down_roi",
    "scaled_down_shape",
    "scaled_up_roi",
    "sides",
    "SomeCRS",
    "split_translation",
    "unary_intersection",
    "unary_union",
    "w_",
    "warp_affine",
]
