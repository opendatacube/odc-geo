# This file is part of the Open Data Cube, see https://opendatacube.org for more information
#
# Copyright (c) 2015-2020 ODC Contributors
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Iterable, Optional, Tuple, Union

import numpy as np
import rasterio.warp
from affine import Affine

from .gcp import GCPGeoBox
from .geobox import GeoBox
from .types import wh_

# pylint: disable=invalid-name, too-many-arguments
Resampling = Union[str, int, rasterio.warp.Resampling]
Nodata = Optional[Union[int, float]]
_WRP_CRS = "epsg:3857"


def resampling_s2rio(name: str) -> rasterio.warp.Resampling:
    """
    Convert from string to rasterio.warp.Resampling enum, raises ValueError on bad input.
    """
    try:
        return getattr(rasterio.warp.Resampling, name.lower())
    except AttributeError:
        raise ValueError(f"Bad resampling parameter: {name}") from None


def is_resampling_nn(resampling: Resampling) -> bool:
    """
    :returns: True if resampling mode is nearest neighbour
    :returns: False otherwise
    """
    if isinstance(resampling, str):
        return resampling.lower() == "nearest"
    return resampling == rasterio.warp.Resampling.nearest


def warp_affine_rio(
    src: np.ndarray,
    dst: np.ndarray,
    A: Affine,
    resampling: Resampling,
    src_nodata: Nodata = None,
    dst_nodata: Nodata = None,
    **kwargs,
) -> np.ndarray:
    """
    Perform Affine warp using rasterio as backend library.

    :param        src: image as ndarray
    :param        dst: image as ndarray
    :param          A: Affine transform, maps from dst_coords to src_coords
    :param resampling: str|rasterio.warp.Resampling resampling strategy
    :param src_nodata: Value representing "no data" in the source image
    :param dst_nodata: Value to represent "no data" in the destination image

    :param     kwargs: any other args to pass to ``rasterio.warp.reproject``

    :returns: dst
    """
    assert src.ndim == dst.ndim
    assert src.ndim == 2
    sh, sw = src.shape
    dh, dw = dst.shape

    s_gbox = GeoBox(wh_(sw, sh), Affine.identity(), _WRP_CRS)
    d_gbox = GeoBox(wh_(dw, dh), A, _WRP_CRS)
    return _rio_reproject(
        src, dst, s_gbox, d_gbox, resampling, src_nodata, dst_nodata, **kwargs
    )


def warp_affine(
    src: np.ndarray,
    dst: np.ndarray,
    A: Affine,
    resampling: Resampling,
    src_nodata: Nodata = None,
    dst_nodata: Nodata = None,
    **kwargs,
) -> np.ndarray:
    """
    Perform Affine warp using best available backend (GDAL via rasterio is the only one so far).

    :param        src: image as ndarray
    :param        dst: image as ndarray
    :param          A: Affine transformm, maps from dst_coords to src_coords
    :param resampling: str resampling strategy
    :param src_nodata: Value representing "no data" in the source image
    :param dst_nodata: Value to represent "no data" in the destination image

    :param     kwargs: any other args to pass to implementation

    :returns: dst
    """
    return warp_affine_rio(
        src, dst, A, resampling, src_nodata=src_nodata, dst_nodata=dst_nodata, **kwargs
    )


def rio_reproject(
    src: np.ndarray,
    dst: np.ndarray,
    s_gbox: Union[GeoBox, GCPGeoBox],
    d_gbox: GeoBox,
    resampling: Resampling,
    src_nodata: Nodata = None,
    dst_nodata: Nodata = None,
    ydim: Optional[int] = None,
    **kwargs,
) -> np.ndarray:
    """
    Perform reproject from ndarray->ndarray using rasterio as backend library.

    :param        src: image as ndarray
    :param        dst: image as ndarray
    :param     s_gbox: GeoBox of source image
    :param     d_gbox: GeoBox of destination image
    :param resampling: str|rasterio.warp.Resampling resampling strategy
    :param src_nodata: Value representing "no data" in the source image
    :param dst_nodata: Value to represent "no data" in the destination image
    :param       ydim: Which dimension is y-axis, next one must be x

    :param     kwargs: any other args to pass to ``rasterio.warp.reproject``

    :returns: dst
    """
    assert src.ndim == dst.ndim
    if dst_nodata is None:
        if dst.dtype.kind == "f":
            dst_nodata = np.nan

    if src.ndim == 2:
        return _rio_reproject(
            src, dst, s_gbox, d_gbox, resampling, src_nodata, dst_nodata, **kwargs
        )

    if ydim is None:
        # Assume last two dimensions are Y/X
        ydim = src.ndim - 2

    extra_dims = (*src.shape[:ydim], *src.shape[ydim + 2 :])
    # Selects each 2d plane in [...]YX[B] array
    slices: Iterable[Any] = (
        (*idx[:ydim], slice(None), slice(None), *idx[ydim:])
        for idx in np.ndindex(*extra_dims)
    )
    for roi in slices:
        _rio_reproject(
            src[roi],
            dst[roi],
            s_gbox,
            d_gbox,
            resampling,
            src_nodata,
            dst_nodata,
            **kwargs,
        )
    return dst


def _rio_reproject(
    src: np.ndarray,
    dst: np.ndarray,
    s_gbox: Union[GCPGeoBox, GeoBox],
    d_gbox: GeoBox,
    resampling: Resampling,
    src_nodata: Nodata = None,
    dst_nodata: Nodata = None,
    **kwargs,
) -> np.ndarray:
    assert src.ndim == dst.ndim
    assert src.ndim == 2

    if "XSCALE" not in kwargs and "YSCALE" not in kwargs:
        # Work around for issue in GDAL
        #    https://github.com/OSGeo/gdal/issues/7750
        # Force GDAL into performing bilinear,cubic,lanczos with raidius=1px.
        # GDAL is trying to be smart and pick sampling radius based on scale
        # change, but does it without consideration for rotational component
        # of the transform, leading to blury output in some situations
        # See also:
        #    https://github.com/opendatacube/datacube-core/issues/1448
        kwargs.update(XSCALE=1, YSCALE=1)

    dtype_remap = {"int8": "int16", "bool": "uint8"}

    def _alias_or_convert(arr: np.ndarray) -> Tuple[np.ndarray, bool]:
        if arr.dtype.name not in dtype_remap:
            return arr, False
        wk_dtype = dtype_remap[arr.dtype.name]
        if arr.dtype.name == "bool":
            F, T = (np.array(v, dtype=wk_dtype) for v in [0, 255])
            return np.where(arr, T, F), True
        return arr.astype(wk_dtype), False

    if isinstance(resampling, str):
        resampling = resampling_s2rio(resampling)

    src_transform = None
    gcps = None

    if isinstance(s_gbox, GCPGeoBox):
        gcps = s_gbox.gcps()
    else:
        src_transform = s_gbox.transform

    # GDAL support for int8 is patchy, warp doesn't support it, so we need to convert to int16
    src, src_is_bool = _alias_or_convert(src)
    _dst, _ = _alias_or_convert(dst)

    rasterio.warp.reproject(
        src,
        _dst,
        src_transform=src_transform,
        gcps=gcps,
        src_crs=str(s_gbox.crs),
        dst_transform=d_gbox.transform,
        dst_crs=str(d_gbox.crs),
        resampling=resampling,
        src_nodata=src_nodata,
        dst_nodata=dst_nodata,
        **kwargs,
    )

    if dst is not _dst:
        # int8 workaround copy pixels back to int8
        if src_is_bool:
            # undo [0, 1] to [0, 255] stretching of the src
            np.copyto(dst, _dst > 127, casting="unsafe")
        else:
            np.copyto(dst, _dst, casting="unsafe")

    return dst
