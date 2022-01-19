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
from ._xr_interop import (
    ODCExtension,
    ODCExtensionDa,
    ODCExtensionDs,
    assign_crs,
    register_geobox,
    spatial_dims,
    xr_coords,
)

__all__ = (
    "ODCExtension",
    "ODCExtensionDa",
    "ODCExtensionDs",
    "assign_crs",
    "register_geobox",
    "spatial_dims",
    "xr_coords",
)
