# This file is part of the Open Data Cube, see https://opendatacube.org for more information
#
# Copyright (c) 2015-2020 ODC Contributors
# SPDX-License-Identifier: Apache-2.0
"""
Xarray Intergration.

Importing this module enables `.odc.` extension on :py:class:`xarray.DataArray` and
:py:class:`xarray.Dataset` objects.

For more details see: :py:class:`~odc.geo.xr.ODCExtension` and
:py:class:`~odc.geo.xr.ODCExtensionDs`.
"""
from ._interop import have
from ._xr_interop import (
    ODCExtension,
    ODCExtensionDa,
    ODCExtensionDs,
    assign_crs,
    colorize,
    rasterize,
    register_geobox,
    spatial_dims,
    to_rgba,
    wrap_xr,
    xr_coords,
    xr_reproject,
    xr_zeros,
)

wrap = wrap_xr

__all__ = [
    "ODCExtension",
    "ODCExtensionDa",
    "ODCExtensionDs",
    "assign_crs",
    "rasterize",
    "register_geobox",
    "spatial_dims",
    "xr_coords",
    "wrap",
    "wrap_xr",
    "xr_reproject",
    "xr_zeros",
    "colorize",
    "to_rgba",
]

# pylint: disable=import-outside-toplevel,unused-import
if have.rasterio:
    from ._xr_interop import add_to, compress, rio_reproject, to_cog, write_cog

    __all__.extend(
        [
            "to_cog",
            "write_cog",
            "add_to",
            "rio_reproject",
            "compress",
        ]
    )
