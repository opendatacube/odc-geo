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

from .crs import (
    CRS,
    CRSError,
    CRSMismatchError,
    MaybeCRS,
    SomeCRS,
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

from ._roi import (
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
    w_,
)

__all__ = [
    "__version__",
    "bbox_intersection",
    "bbox_union",
    "BoundingBox",
    "box",
    "chop_along_antimeridian",
    "clip_lon180",
    "common_crs",
    "CoordList",
    "CRS",
    "CRSError",
    "CRSMismatchError",
    "Geometry",
    "intersects",
    "line",
    "lonlat_bounds",
    "MaybeCRS",
    "mid_longitude",
    "multigeom",
    "multiline",
    "multipoint",
    "multipolygon",
    "point",
    "polygon_from_transform",
    "polygon",
    "projected_lon",
    "roi_boundary",
    "roi_center",
    "roi_from_points",
    "roi_intersect",
    "roi_is_empty",
    "roi_is_full",
    "roi_normalise",
    "roi_pad",
    "roi_shape",
    "scaled_down_roi",
    "scaled_down_shape",
    "scaled_up_roi",
    "sides",
    "SomeCRS",
    "unary_intersection",
    "unary_union",
    "w_",
]
