# This file is part of the Open Data Cube, see https://opendatacube.org for more information
#
# Copyright (c) 2015-2020 ODC Contributors
# SPDX-License-Identifier: Apache-2.0
"""
Write Cloud Optimized GeoTIFFs from xarrays.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, Literal, Optional, Tuple, Union

import numpy as np

from ..geobox import GeoBox
from ..math import align_down_pow2, align_up
from ..types import MaybeNodata, Shape2d, SomeShape, shape_, wh_

# pylint: disable=too-many-locals,too-many-branches,too-many-arguments,too-many-statements,too-many-instance-attributes

AxisOrder = Union[Literal["YX"], Literal["YXS"], Literal["SYX"]]

# map compressor name to level name in GDAL
GDAL_COMP: Dict[str, str] = {
    "DEFLATE": "ZLEVEL",
    "ADOBE_DEFLATE": "ZLEVEL",
    "ZSTD": "ZSTD_LEVEL",
    "WEBP": "WEBP_LEVEL",
    "LERC": "MAX_Z_ERROR",
    "LERC_DEFLATE": "MAX_Z_ERROR",
    "LERC_ZSTD": "MAX_Z_ERROR",
    "JPEG": "JPEG_QUALITY",
}

GEOTIFF_TAGS = {
    34264,  # ModelTransformation
    34735,  # GeoKeyDirectory
    34736,  # GeoDoubleParams
    34737,  # GeoAsciiParams
    33550,  # ModelPixelScale
    33922,  # ModelTiePoint
    #
    42112,  # GDAL_METADATA
    42113,  # GDAL_NODATA
    #
    # probably never used in the wild
    33920,  # IrasB Transformation Matrix
    50844,  # RPCCoefficientTag
}


@dataclass
class CogMeta:
    """
    COG metadata.
    """

    axis: AxisOrder
    shape: Shape2d
    tile: Shape2d
    nsamples: int
    dtype: Any
    compression: int
    predictor: int
    compressionargs: Dict[str, Any] = field(default_factory=dict, repr=False)
    gbox: Optional[GeoBox] = None
    overviews: Tuple["CogMeta", ...] = field(default=(), repr=False)
    nodata: MaybeNodata = None

    def _pix_shape(self, shape: Shape2d) -> Tuple[int, ...]:
        if self.axis == "YX":
            return shape.shape
        if self.axis == "YXS":
            return (*shape.shape, self.nsamples)
        return (self.nsamples, *shape.shape)

    @property
    def chunks(self) -> Tuple[int, ...]:
        return self._pix_shape(self.tile)

    @property
    def pix_shape(self) -> Tuple[int, ...]:
        return self._pix_shape(self.shape)

    @property
    def num_planes(self):
        if self.axis == "SYX":
            return self.nsamples
        return 1

    def flatten(self) -> Tuple["CogMeta", ...]:
        return (self, *self.overviews)

    @property
    def chunked(self) -> Shape2d:
        """
        Shape in chunks.
        """
        ny, nx = ((N + n - 1) // n for N, n in zip(self.shape.yx, self.tile.yx))
        return shape_((ny, nx))

    @property
    def num_tiles(self):
        ny, nx = self.chunked.yx
        return self.num_planes * ny * nx

    def tidx(self, sample_idx: Optional[int] = None) -> Iterator[Tuple[int, int, int]]:
        """``[([sample|plane]_idx, iy, ix), ...]``"""
        if sample_idx is not None:
            assert sample_idx < self.num_planes
            yield from ((sample_idx, y, x) for y, x in np.ndindex(self.chunked.yx))
        else:
            yield from np.ndindex((self.num_planes, *self.chunked.yx))

    def flat_tile_idx(self, idx: Tuple[int, int, int]) -> int:
        """Convert from sample,iy,ix to flat tile index."""
        ns = self.num_planes
        ny, nx = self.chunked.yx
        for n, i in zip((ns, ny, nx), idx):
            if i < 0 or i >= n:
                raise IndexError()

        sample, y, x = idx
        return sample * (ny * nx) + y * nx + x

    def cog_tidx(self) -> Iterator[Tuple[int, int, int, int]]:
        """``[(ifd_idx, plane_idx, iy, ix), ...]``"""
        idx_layers = list(enumerate(self.flatten()))[::-1]
        for idx, mm in idx_layers:
            yield from ((idx, pi, yi, xi) for pi, yi, xi in mm.tidx())

    def __dask_tokenize__(self):
        return (
            "odc.CogMeta",
            self.axis,
            *self.shape.yx,
            *self.tile.yx,
            self.nsamples,
            self.dtype,
            self.compression,
            self.predictor,
            self.gbox,
            len(self.overviews),
            self.nodata,
        )


def adjust_blocksize(block: int, dim: int = 0) -> int:
    if 0 < dim < block:
        return align_up(dim, 16)
    return align_up(block, 16)


def norm_blocksize(block: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    if isinstance(block, int):
        block = adjust_blocksize(block)
        return (block, block)

    b1, b2 = map(adjust_blocksize, block)
    return (b1, b2)


def num_overviews(block: int, dim: int) -> int:
    c = 0
    while block < dim:
        dim = dim // 2
        c += 1
    return c


def compute_cog_spec(
    data_shape: SomeShape,
    tile_shape: SomeShape,
    *,
    max_pad: Optional[int] = None,
) -> Tuple[Shape2d, Shape2d, int]:
    data_shape = shape_(data_shape)
    tile_shape = shape_(shape_(tile_shape).map(adjust_blocksize))
    n1, n2 = (num_overviews(b, dim) for dim, b in zip(data_shape.xy, tile_shape.xy))
    n = max(n1, n2)
    pad = 2**n
    if max_pad is not None and max_pad < pad:
        pad = 0 if max_pad == 0 else align_down_pow2(max_pad)

    if pad > 0:
        data_shape = shape_(data_shape.map(lambda d: align_up(d, pad)))
    return (data_shape, tile_shape, n)


def cog_gbox(
    gbox: GeoBox,
    *,
    tile: Union[None, int, Tuple[int, int], Shape2d] = None,
    nlevels: Optional[int] = None,
) -> GeoBox:
    """
    Return padded gbox with safe dimensions for COG.

    1. Compute number of desired overviews
    2. Expand gebox on the right/bottom to have exact pixel shrink across all levels
    """

    if nlevels is None:
        if tile is None:
            tile = wh_(256, 256)
        if isinstance(tile, int):
            tile = wh_(tile, tile)
        new_shape, _, _ = compute_cog_spec(gbox.shape, tile)
    else:
        pad = 1 << nlevels
        new_shape = shape_(gbox.shape.map(lambda d: align_up(d, pad)))
    return gbox.expand(new_shape)


def yaxis_from_shape(
    shape: Tuple[int, ...], gbox: Optional[GeoBox] = None
) -> Tuple[AxisOrder, int]:
    ndim = len(shape)

    if ndim == 2:
        return "YX", 0

    if ndim != 3:
        raise ValueError("Can only work with 2-d or 3-d data")
    if shape[-1] in (3, 4):  # YXS in RGB(A)
        return "YXS", 0

    if gbox is None:
        return "SYX", 1
    if gbox.shape == shape[:2]:  # YXS
        return "YXS", 0
    if gbox.shape == shape[1:]:  # SYX
        return "SYX", 1

    raise ValueError("Geobox and image shape do not match")
