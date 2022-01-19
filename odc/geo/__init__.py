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

from .geom import (
    BoundingBox,
    CoordList,
    Geometry,
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
    "BoundingBox",
    "CoordList",
    "CRS",
    "CRSError",
    "CRSMismatchError",
    "Geometry",
    "MaybeCRS",
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
    "SomeCRS",
    "w_",
]
