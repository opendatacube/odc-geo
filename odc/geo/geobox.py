# This file is part of the Open Data Cube, see https://opendatacube.org for more information
#
# Copyright (c) 2015-2020 ODC Contributors
# SPDX-License-Identifier: Apache-2.0
import itertools
import math
from collections import OrderedDict, namedtuple
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy
from affine import Affine

from .crs import CRS, MaybeCRS
from .geom import (
    BoundingBox,
    Geometry,
    bbox_intersection,
    bbox_union,
    polygon_from_transform,
)
from .roi import align_up, polygon_path, roi_normalise, roi_shape
from .math import clamp, is_affine_st, is_almost_int

# pylint: disable=invalid-name
MaybeInt = Optional[int]
MaybeFloat = Optional[float]


Coordinate = namedtuple("Coordinate", ("values", "units", "resolution"))


def _align_pix(left: float, right: float, res: float, off: float) -> Tuple[float, int]:
    if res < 0:
        res = -res
        val = math.ceil((right - off) / res) * res + off
        width = max(1, int(math.ceil((val - left - 0.1 * res) / res)))
    else:
        val = math.floor((left - off) / res) * res + off
        width = max(1, int(math.ceil((right - val - 0.1 * res) / res)))
    return val, width


class GeoBox:
    """
    Defines the location and resolution of a rectangular grid of data,
    including it's :py:class:`CRS`.

    :param crs: Coordinate Reference System
    :param affine: Affine transformation defining the location of the geobox
    """

    def __init__(self, width: int, height: int, affine: Affine, crs: MaybeCRS):
        assert is_affine_st(
            affine
        ), "Only axis-aligned geoboxes are currently supported"
        self.width = width
        self.height = height
        self.affine = affine
        self.extent = polygon_from_transform(width, height, affine, crs=crs)

    @staticmethod
    def from_geopolygon(
        geopolygon: Geometry,
        resolution: Union[float, int, Tuple[float, float]],
        crs: MaybeCRS = None,
        align: Optional[Tuple[float, float]] = None,
    ) -> "GeoBox":
        """
        :param resolution: (y_resolution, x_resolution)
        :param crs: CRS to use, if different from the geopolygon
        :param align: Align geobox such that point 'align' lies on the pixel boundary.
        """

        if isinstance(resolution, float):
            resolution = -resolution, resolution
        elif isinstance(resolution, int):
            resolution = float(-resolution), float(resolution)

        align = align or (0.0, 0.0)
        assert (
            0.0 <= align[1] <= abs(resolution[1])
        ), "X align must be in [0, abs(x_resolution)] range"
        assert (
            0.0 <= align[0] <= abs(resolution[0])
        ), "Y align must be in [0, abs(y_resolution)] range"

        if crs is None:
            crs = geopolygon.crs
        else:
            geopolygon = geopolygon.to_crs(crs)

        bounding_box = geopolygon.boundingbox
        offx, width = _align_pix(
            bounding_box.left, bounding_box.right, resolution[1], align[1]
        )
        offy, height = _align_pix(
            bounding_box.bottom, bounding_box.top, resolution[0], align[0]
        )
        affine = Affine.translation(offx, offy) * Affine.scale(
            resolution[1], resolution[0]
        )
        return GeoBox(crs=crs, affine=affine, width=width, height=height)

    def buffered(self, xbuff: float, ybuff: Optional[float] = None) -> "GeoBox":
        """
        Produce a tile buffered by xbuff, ybuff (in CRS units)
        """
        if ybuff is None:
            ybuff = xbuff

        by, bx = (
            _round_to_res(buf, res) for buf, res in zip((ybuff, xbuff), self.resolution)
        )
        affine = self.affine * Affine.translation(-bx, -by)

        return GeoBox(
            width=self.width + 2 * bx,
            height=self.height + 2 * by,
            affine=affine,
            crs=self.crs,
        )

    def __getitem__(self, roi) -> "GeoBox":
        if isinstance(roi, int):
            roi = (slice(roi, roi + 1), slice(None, None))

        if isinstance(roi, slice):
            roi = (roi, slice(None, None))

        if len(roi) > 2:
            raise ValueError("Expect 2d slice")

        if not all(s.step is None or s.step == 1 for s in roi):
            raise NotImplementedError("scaling not implemented, yet")

        roi = roi_normalise(roi, self.shape)
        ty, tx = (s.start for s in roi)
        h, w = roi_shape(roi)

        affine = self.affine * Affine.translation(tx, ty)

        return GeoBox(width=w, height=h, affine=affine, crs=self.crs)

    def __or__(self, other) -> "GeoBox":
        """A geobox that encompasses both self and other."""
        return geobox_union_conservative([self, other])

    def __and__(self, other) -> "GeoBox":
        """A geobox that is contained in both self and other."""
        return geobox_intersection_conservative([self, other])

    def is_empty(self) -> bool:
        return self.width == 0 or self.height == 0

    def __bool__(self) -> bool:
        return not self.is_empty()

    def __hash__(self):
        return hash((*self.shape, self.crs, self.affine))

    @property
    def transform(self) -> Affine:
        return self.affine

    @property
    def shape(self) -> Tuple[int, int]:
        return self.height, self.width

    @property
    def crs(self) -> Optional[CRS]:
        return self.extent.crs

    @property
    def dimensions(self) -> Tuple[str, str]:
        """
        List of dimension names of the GeoBox
        """
        crs = self.crs
        if crs is None:
            return ("y", "x")
        return crs.dimensions

    @property
    def resolution(self) -> Tuple[float, float]:
        """
        Resolution in Y,X dimensions
        """
        return self.affine.e, self.affine.a

    @property
    def alignment(self) -> Tuple[float, float]:
        """
        Alignment of pixel boundaries in Y,X dimensions
        """
        return self.affine.yoff % abs(self.affine.e), self.affine.xoff % abs(
            self.affine.a
        )

    @property
    def coordinates(self) -> Dict[str, Coordinate]:
        """
        dict of coordinate labels
        """
        yres, xres = self.resolution
        yoff, xoff = self.affine.yoff, self.affine.xoff

        xs = numpy.arange(self.width) * xres + (xoff + xres / 2)
        ys = numpy.arange(self.height) * yres + (yoff + yres / 2)

        units = self.crs.units if self.crs is not None else ("1", "1")

        return OrderedDict(
            (dim, Coordinate(labels, units, res))
            for dim, labels, units, res in zip(
                self.dimensions, (ys, xs), units, (yres, xres)
            )
        )

    @property
    def geographic_extent(self) -> Geometry:
        """GeoBox extent in EPSG:4326"""
        if self.crs is None or self.crs.geographic:
            return self.extent
        return self.extent.to_crs(CRS("EPSG:4326"))

    coords = coordinates
    dims = dimensions

    def __str__(self):
        return f"GeoBox({self.geographic_extent})"

    def __repr__(self):
        return f"GeoBox({self.width}, {self.height}, {self.affine!r}, {self.crs})"

    def __eq__(self, other):
        if not isinstance(other, GeoBox):
            return False

        return (
            self.shape == other.shape
            and self.transform == other.transform
            and self.crs == other.crs
        )


def gbox_boundary(gbox: GeoBox, pts_per_side: int = 16) -> Geometry:
    """Return points in pixel space along the perimeter of a GeoBox, or a 2d array."""
    H, W = gbox.shape[:2]
    xx = numpy.linspace(0, W, pts_per_side, dtype="float32")
    yy = numpy.linspace(0, H, pts_per_side, dtype="float32")

    return polygon_path(xx, yy).T[:-1]


def bounding_box_in_pixel_domain(geobox: GeoBox, reference: GeoBox) -> BoundingBox:
    """
    Returns the bounding box of `geobox` with respect to the pixel grid
    defined by `reference` when their coordinate grids are compatible,
    that is, have the same CRS, same pixel size and orientation, and
    are related by whole pixel translation,
    otherwise raises `ValueError`.
    """
    tol = 1.0e-8

    if reference.crs != geobox.crs:
        raise ValueError("Cannot combine geoboxes in different CRSs")

    a, b, c, d, e, f, *_ = ~reference.affine * geobox.affine

    if not (
        numpy.isclose(a, 1)
        and numpy.isclose(b, 0)
        and is_almost_int(c, tol)
        and numpy.isclose(d, 0)
        and numpy.isclose(e, 1)
        and is_almost_int(f, tol)
    ):
        raise ValueError("Incompatible grids")

    tx, ty = round(c), round(f)
    return BoundingBox(tx, ty, tx + geobox.width, ty + geobox.height)


def geobox_union_conservative(geoboxes: List[GeoBox]) -> GeoBox:
    """Union of geoboxes. Fails whenever incompatible grids are encountered."""
    if len(geoboxes) == 0:
        raise ValueError("No geoboxes supplied")

    reference, *_ = geoboxes

    bbox = bbox_union(
        bounding_box_in_pixel_domain(geobox, reference=reference) for geobox in geoboxes
    )

    affine = reference.affine * Affine.translation(*bbox[:2])

    return GeoBox(
        width=bbox.width, height=bbox.height, affine=affine, crs=reference.crs
    )


def geobox_intersection_conservative(geoboxes: List[GeoBox]) -> GeoBox:
    """
    Intersection of geoboxes. Fails whenever incompatible grids are encountered.
    """
    if len(geoboxes) == 0:
        raise ValueError("No geoboxes supplied")

    reference, *_ = geoboxes

    bbox = bbox_intersection(
        bounding_box_in_pixel_domain(geobox, reference=reference) for geobox in geoboxes
    )

    # standardise empty geobox representation
    if bbox.left > bbox.right:
        bbox = BoundingBox(
            left=bbox.left, bottom=bbox.bottom, right=bbox.left, top=bbox.top
        )
    if bbox.bottom > bbox.top:
        bbox = BoundingBox(
            left=bbox.left, bottom=bbox.bottom, right=bbox.right, top=bbox.bottom
        )

    affine = reference.affine * Affine.translation(*bbox[:2])

    return GeoBox(
        width=bbox.width, height=bbox.height, affine=affine, crs=reference.crs
    )


def scaled_down_geobox(src_geobox: GeoBox, scaler: int) -> GeoBox:
    """Given a source geobox and integer scaler compute geobox of a scaled down image.

    Output geobox will be padded when shape is not a multiple of scaler.
    Example: 5x4, scaler=2 -> 3x2

    NOTE: here we assume that pixel coordinates are 0,0 at the top-left
          corner of a top-left pixel.

    """
    assert scaler > 1

    H, W = (X // scaler + (1 if X % scaler else 0) for X in src_geobox.shape)

    # Since 0,0 is at the corner of a pixel, not center, there is no
    # translation between pixel plane coords due to scaling
    A = src_geobox.transform * Affine.scale(scaler, scaler)

    return GeoBox(W, H, A, src_geobox.crs)


def _round_to_res(value: float, res: float) -> int:
    res = abs(res)
    return int(math.ceil((value - 0.1 * res) / res))


def flipy(gbox: GeoBox) -> GeoBox:
    """
    :returns: GeoBox covering the same region but with Y-axis flipped
    """
    H, W = gbox.shape
    A = Affine.translation(0, H) * Affine.scale(1, -1)
    A = gbox.affine * A
    return GeoBox(W, H, A, gbox.crs)


def flipx(gbox: GeoBox) -> GeoBox:
    """
    :returns: GeoBox covering the same region but with X-axis flipped
    """
    H, W = gbox.shape
    A = Affine.translation(W, 0) * Affine.scale(-1, 1)
    A = gbox.affine * A
    return GeoBox(W, H, A, gbox.crs)


def translate_pix(gbox: GeoBox, tx: float, ty: float) -> GeoBox:
    """
    Shift GeoBox in pixel plane. (0,0) of the new GeoBox will be at the same
    location as pixel (tx, ty) in the original GeoBox.
    """
    H, W = gbox.shape
    A = gbox.affine * Affine.translation(tx, ty)
    return GeoBox(W, H, A, gbox.crs)


def pad(gbox: GeoBox, padx: int, pady: MaybeInt = None) -> GeoBox:
    """
    Expand GeoBox by fixed number of pixels on each side
    """
    # false positive for -pady, it's never None by the time it runs
    # pylint: disable=invalid-unary-operand-type

    pady = padx if pady is None else pady

    H, W = gbox.shape
    A = gbox.affine * Affine.translation(-padx, -pady)
    return GeoBox(W + padx * 2, H + pady * 2, A, gbox.crs)


def pad_wh(gbox: GeoBox, alignx: int = 16, aligny: MaybeInt = None) -> GeoBox:
    """
    Expand GeoBox such that width and height are multiples of supplied number.
    """
    aligny = alignx if aligny is None else aligny
    H, W = gbox.shape

    return GeoBox(align_up(W, alignx), align_up(H, aligny), gbox.affine, gbox.crs)


def zoom_out(gbox: GeoBox, factor: float) -> GeoBox:
    """
    factor > 1 --> smaller width/height, fewer but bigger pixels
    factor < 1 --> bigger width/height, more but smaller pixels

    :returns: GeoBox covering the same region but with bigger pixels (i.e. lower resolution)
    """

    H, W = (max(1, math.ceil(s / factor)) for s in gbox.shape)
    A = gbox.affine * Affine.scale(factor, factor)
    return GeoBox(W, H, A, gbox.crs)


def zoom_to(gbox: GeoBox, shape: Tuple[int, int]) -> GeoBox:
    """
    :returns: GeoBox covering the same region but with different number of pixels
              and therefore resolution.
    """
    H, W = gbox.shape
    h, w = shape

    sx, sy = W / float(w), H / float(h)
    A = gbox.affine * Affine.scale(sx, sy)
    return GeoBox(w, h, A, gbox.crs)


def rotate(gbox: GeoBox, deg: float) -> GeoBox:
    """
    Rotate GeoBox around the center.

    It's as if you stick a needle through the center of the GeoBox footprint
    and rotate it counter clock wise by supplied number of degrees.

    Note that from pixel point of view image rotates the other way. If you have
    source image with an arrow pointing right, and you rotate GeoBox 90 degree,
    in that view arrow should point down (this is assuming usual case of inverted
    y-axis)
    """
    h, w = gbox.shape
    c0 = gbox.transform * (w * 0.5, h * 0.5)
    A = Affine.rotation(deg, c0) * gbox.transform
    return GeoBox(w, h, A, gbox.crs)


def affine_transform_pix(gbox: GeoBox, transform: Affine) -> GeoBox:
    """
    Apply affine transform on pixel side.

    :param transform: Affine matrix mapping from new pixel coordinate space to
    pixel coordinate space of input gbox

    :returns: GeoBox of the same pixel shape but covering different region,
    pixels in the output gbox relate to input geobox via `transform`

    X_old_pix = transform * X_new_pix

    """
    H, W = gbox.shape
    A = gbox.affine * transform
    return GeoBox(W, H, A, gbox.crs)


class GeoboxTiles:
    """Partition GeoBox into sub geoboxes"""

    def __init__(self, box: GeoBox, tile_shape: Tuple[int, int]):
        """Construct from a ``GeoBox``.

        :param box: source :class:`~odc.geo.GeoBox`
        :param tile_shape: Shape of sub-tiles in pixels (rows, cols)
        """
        self._gbox = box
        self._tile_shape = tile_shape
        self._shape = tuple(
            math.ceil(float(N) / n) for N, n in zip(box.shape, tile_shape)
        )
        self._cache: Dict[Tuple[int, int], GeoBox] = {}

    @property
    def base(self) -> GeoBox:
        return self._gbox

    @property
    def shape(self):
        """Number of tiles along each dimension"""
        return self._shape

    def _idx_to_slice(self, idx: Tuple[int, int]) -> Tuple[slice, slice]:
        def _slice(i, N, n) -> slice:
            _in = i * n
            if 0 <= _in < N:
                return slice(_in, min(_in + n, N))
            raise IndexError(f"Index ({idx[0]},{idx[1]})is out of range")

        ir, ic = (
            _slice(i, N, n) for i, N, n in zip(idx, self._gbox.shape, self._tile_shape)
        )
        return (ir, ic)

    def chunk_shape(self, idx: Tuple[int, int]) -> Tuple[int, int]:
        """Chunk shape for a given chunk index.

        :param idx: (row, col) index
        :returns: (nrow, ncols) shape of a tile (edge tiles might be smaller)
        :raises: IndexError when index is outside of [(0,0) -> .shape)
        """

        def _sz(i: int, n: int, tile_sz: int, total_sz: int) -> int:
            if 0 <= i < n - 1:  # not edge tile
                return tile_sz
            if i == n - 1:  # edge tile
                return total_sz - (i * tile_sz)
            # out of index case
            raise IndexError(f"Index ({idx[0]},{idx[1]}) is out of range")

        n1, n2 = map(_sz, idx, self._shape, self._tile_shape, self._gbox.shape)
        return (n1, n2)

    def __getitem__(self, idx: Tuple[int, int]) -> GeoBox:
        """Lookup tile by index, index is in matrix access order: (row, col)

        :param idx: (row, col) index
        :returns: GeoBox of a tile
        :raises: IndexError when index is outside of [(0,0) -> .shape)
        """
        sub_gbox = self._cache.get(idx, None)
        if sub_gbox is not None:
            return sub_gbox

        roi = self._idx_to_slice(idx)
        return self._cache.setdefault(idx, self._gbox[roi])

    def range_from_bbox(self, bbox: BoundingBox) -> Tuple[range, range]:
        """Compute rows and columns overlapping with a given ``BoundingBox``"""

        def clamped_range(v1: float, v2: float, N: int) -> range:
            _in = clamp(math.floor(v1), 0, N)
            _out = clamp(math.ceil(v2), 0, N)
            return range(_in, _out)

        sy, sx = self._tile_shape
        A = Affine.scale(1.0 / sx, 1.0 / sy) * (~self._gbox.transform)
        # A maps from X,Y in meters to chunk index
        bbox = bbox.transform(A)

        NY, NX = self.shape
        xx = clamped_range(bbox.left, bbox.right, NX)
        yy = clamped_range(bbox.bottom, bbox.top, NY)
        return (yy, xx)

    def tiles(self, polygon: Geometry) -> Iterable[Tuple[int, int]]:
        """Return tile indexes overlapping with a given geometry."""
        target_crs = self._gbox.crs
        poly = polygon
        if target_crs is not None and poly.crs != target_crs:
            poly = poly.to_crs(target_crs)

        yy, xx = self.range_from_bbox(poly.boundingbox)
        for idx in itertools.product(yy, xx):
            gbox = self[idx]
            if gbox.extent.intersects(poly):
                yield idx
