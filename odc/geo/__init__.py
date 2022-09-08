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

from .types import (
    XY,
    Index2d,
    Shape2d,
    Resolution,
    SomeShape,
    SomeResolution,
    SomeIndex2d,
    Unset,
    xy_,
    yx_,
    res_,
    resxy_,
    resyx_,
    ixy_,
    iyx_,
    wh_,
    shape_,
)

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

__all__ = [
    "__version__",
    "XY",
    "Index2d",
    "Shape2d",
    "Resolution",
    "SomeShape",
    "SomeResolution",
    "SomeIndex2d",
    "BoundingBox",
    "CoordList",
    "CRS",
    "CRSError",
    "CRSMismatchError",
    "Geometry",
    "MaybeCRS",
    "SomeCRS",
    "Unset",
    "xy_",
    "yx_",
    "res_",
    "resxy_",
    "resyx_",
    "ixy_",
    "iyx_",
    "wh_",
    "shape_",
]
