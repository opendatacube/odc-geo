# This file is part of the Open Data Cube, see https://opendatacube.org for more information
#
# Copyright (c) 2015-2020 ODC Contributors
# SPDX-License-Identifier: Apache-2.0
"""Geometric shapes and operations on them"""
# import order is important here
#  _crs <-- _geom <-- _geobox <- other
# isort: skip_file

from .types import (
    AnchorEnum,
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

from .geobox import GeoBox, GeoboxTiles
from .gcp import GCPGeoBox, GCPMapping

__all__ = [
    "AnchorEnum",
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
    "GeoBox",
    "GeoboxTiles",
    "GCPGeoBox",
    "GCPMapping",
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


def __getattr__(name: str) -> str:
    from importlib.metadata import version  # pylint: disable=import-outside-toplevel

    if name == "__version__":
        return version("odc_geo")
    raise AttributeError(f"module {__name__} has no attribute {name}")
