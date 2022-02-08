# This file is part of the Open Data Cube, see https://opendatacube.org for more information
#
# Copyright (c) 2015-2020 ODC Contributors
# SPDX-License-Identifier: Apache-2.0
import itertools
import math
from collections import OrderedDict, namedtuple
from typing import Dict, Iterable, List, Optional, Tuple

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
from .math import clamp, is_affine_st, is_almost_int
from .roi import align_up, polygon_path, roi_normalise, roi_shape
from .types import (
    XY,
    Index2d,
    MaybeInt,
    Resolution,
    SomeIndex2d,
    SomeResolution,
    SomeShape,
    iyx_,
    res_,
    resxy_,
    xy_,
    yx_,
)

# pylint: disable=invalid-name
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
    including it's :py:class:`~odc.geo.crs.CRS`.

    :param shape: Shape in pixels ``(ny, nx)``
    :param crs: Coordinate Reference System
    :param affine: Affine transformation defining the location of the geobox
    """

    def __init__(self, shape: SomeShape, affine: Affine, crs: MaybeCRS):
        assert is_affine_st(
            affine
        ), "Only axis-aligned geoboxes are currently supported"
        shape = yx_(shape)

        self._shape = shape
        self.affine = affine

        ny, nx = shape.yx
        self.extent = polygon_from_transform(nx, ny, affine, crs=crs)

    @staticmethod
    def from_geopolygon(
        geopolygon: Geometry,
        resolution: SomeResolution,
        crs: MaybeCRS = None,
        align: Optional[XY[float]] = None,
    ) -> "GeoBox":
        """
        Construct :py:class:`~odc.geo.geobox.GeoBox` from a polygon.

        :param resolution:
           Either a single number or a :py:class:`~odc.geo.types.Resolution` object.
        :param crs:
           CRS to use, if different from the geopolygon
        :param align:
           Align geobox such that point 'align' lies on the pixel boundary.
        """
        resolution = res_(resolution)
        if align is None:
            align = xy_(0.0, 0.0)

        assert (
            0.0 <= align.x <= abs(resolution.x)
        ), "X align must be in [0, abs(x_resolution)] range"
        assert (
            0.0 <= align.y <= abs(resolution.y)
        ), "Y align must be in [0, abs(y_resolution)] range"

        if crs is None:
            crs = geopolygon.crs
        else:
            geopolygon = geopolygon.to_crs(crs)

        bbox = geopolygon.boundingbox
        rx, ry = resolution.xy
        offx, nx = _align_pix(bbox.left, bbox.right, rx, align.x)
        offy, ny = _align_pix(bbox.bottom, bbox.top, ry, align.y)
        affine = Affine.translation(offx, offy) * Affine.scale(rx, ry)
        return GeoBox((ny, nx), crs=crs, affine=affine)

    def buffered(self, xbuff: float, ybuff: Optional[float] = None) -> "GeoBox":
        """
        Produce a tile buffered by ``xbuff, ybuff`` (in CRS units).
        """
        if ybuff is None:
            ybuff = xbuff

        by, bx = (
            _round_to_res(buf, res)
            for buf, res in zip((ybuff, xbuff), self.resolution.yx)
        )
        affine = self.affine * Affine.translation(-bx, -by)

        ny, nx = (sz + 2 * b for sz, b in zip(self.shape, (by, bx)))

        return GeoBox(
            (ny, nx),
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
        ny, nx = roi_shape(roi)

        affine = self.affine * Affine.translation(tx, ty)

        return GeoBox(shape=(ny, nx), affine=affine, crs=self.crs)

    def __or__(self, other) -> "GeoBox":
        """A geobox that encompasses both self and other."""
        return geobox_union_conservative([self, other])

    def __and__(self, other) -> "GeoBox":
        """A geobox that is contained in both self and other."""
        return geobox_intersection_conservative([self, other])

    def is_empty(self) -> bool:
        """Check if geobox is "empty"."""
        return self._shape.shape == (0, 0)

    def __bool__(self) -> bool:
        return not self.is_empty()

    def __hash__(self):
        return hash((*self.shape, self.crs, self.affine))

    @property
    def transform(self) -> Affine:
        """Linear mapping from pixel space to CRS."""
        return self.affine

    @property
    def width(self) -> int:
        """Width in pixels (nx)."""
        return self._shape.x

    @property
    def height(self) -> int:
        """Height in pixels (ny)."""
        return self._shape.y

    @property
    def shape(self) -> Tuple[int, int]:
        """Shape in pixels ``(height, width)``."""
        return self._shape.shape

    @property
    def crs(self) -> Optional[CRS]:
        """Coordinate Reference System of the GeoBox."""
        return self.extent.crs

    @property
    def dimensions(self) -> Tuple[str, str]:
        """List of dimension names of the GeoBox."""
        crs = self.crs
        if crs is None:
            return ("y", "x")
        return crs.dimensions

    @property
    def resolution(self) -> Resolution:
        """Resolution, pixel size in CRS units."""
        rx, _, _, _, ry, *_ = self.affine
        return resxy_(rx, ry)

    @property
    def alignment(self) -> XY[float]:
        """
        Alignment of pixel boundaries in CRS units.

        This is usally ``(0,0)``.
        """
        rx, _, tx, _, ry, ty, *_ = self.affine
        return xy_(tx % abs(rx), ty % abs(ry))

    @property
    def coordinates(self) -> Dict[str, Coordinate]:
        """
        Query coordinates.

        :return:
          Mapping from coordinate name to :py:class:`~odc.geo.geobox.Coordinate`.
        """
        rx, _, tx, _, ry, ty, *_ = self.affine
        ny, nx = self.shape

        xs = numpy.arange(nx) * rx + (tx + rx / 2)
        ys = numpy.arange(ny) * ry + (ty + ry / 2)

        units = self.crs.units if self.crs is not None else ("1", "1")

        return OrderedDict(
            (dim, Coordinate(labels, units, res))
            for dim, labels, units, res in zip(
                self.dimensions, (ys, xs), units, (ry, rx)
            )
        )

    @property
    def geographic_extent(self) -> Geometry:
        """GeoBox extent in EPSG:4326."""
        if self.crs is None or self.crs.geographic:
            return self.extent
        return self.extent.to_crs(CRS("EPSG:4326"))

    coords = coordinates
    dims = dimensions

    def __str__(self):
        return f"GeoBox({self.geographic_extent})"

    def __repr__(self):
        return f"GeoBox(({self._shape.y, self._shape.x}), {self.affine!r}, {self.crs})"

    def __eq__(self, other):
        if not isinstance(other, GeoBox):
            return False

        return (
            self._shape == other._shape
            and self.affine == other.affine
            and self.crs == other.crs
        )


def gbox_boundary(gbox: GeoBox, pts_per_side: int = 16) -> numpy.ndarray:
    """
    Boundary of a :py:class:`~odc.geo.geobox.GeoBox`.

    :return:
      Points in pixel space along the perimeter of a GeoBox as a 2d array.
    """
    ny, nx = gbox.shape[:2]
    xx = numpy.linspace(0, nx, pts_per_side, dtype="float32")
    yy = numpy.linspace(0, ny, pts_per_side, dtype="float32")

    return polygon_path(xx, yy).T[:-1]


def bounding_box_in_pixel_domain(
    geobox: GeoBox, reference: GeoBox, tol: float = 1e-8
) -> BoundingBox:
    """
    Bounding box of ``geobox`` in pixel space of ``reference``.

    :return:
      The bounding box of ``geobox`` with respect to the pixel grid defined by ``reference`` when
      their coordinate grids are compatible. Two geoboxes are compatible when they have the same
      CRS, same pixel size and orientation, and are related by whole pixel translation.

    :raises:
      :py:class:`ValueError` when two geoboxes are not pixel-aligned.
    """
    if reference.crs != geobox.crs:
        raise ValueError("Cannot combine geoboxes in different CRSs")

    # compute pixel-to-pixel transform
    # expect it to be a pure, pixel aligned translation
    #    1  0  tx
    #    0  1  ty
    #    0  0   1
    # Such that tx,ty are almost integer.
    sx, z1, tx, z2, sy, ty, *_ = ~reference.affine * geobox.affine

    if not (
        numpy.isclose(sx, 1)
        and numpy.isclose(z1, 0)
        and is_almost_int(tx, tol)
        and numpy.isclose(z2, 0)
        and numpy.isclose(sy, 1)
        and is_almost_int(ty, tol)
    ):
        raise ValueError("Incompatible grids")

    tx, ty = round(tx), round(ty)
    ny, nx = geobox.shape
    return BoundingBox(tx, ty, tx + nx, ty + ny)


def geobox_union_conservative(geoboxes: List[GeoBox]) -> GeoBox:
    """
    Union of geoboxes as a geobox.

    Fails whenever incompatible grids are encountered.
    """
    if len(geoboxes) == 0:
        raise ValueError("No geoboxes supplied")

    reference, *_ = geoboxes

    bbox = bbox_union(
        bounding_box_in_pixel_domain(geobox, reference=reference) for geobox in geoboxes
    )

    affine = reference.affine * Affine.translation(*bbox[:2])
    return GeoBox(shape=bbox.shape, affine=affine, crs=reference.crs)


def geobox_intersection_conservative(geoboxes: List[GeoBox]) -> GeoBox:
    """
    Intersection of geoboxes.

    Fails whenever incompatible grids are encountered.
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

    return GeoBox(shape=bbox.shape, affine=affine, crs=reference.crs)


def scaled_down_geobox(src_geobox: GeoBox, scaler: int) -> GeoBox:
    """
    Compute :py:class:`~odc.geo.geobox.GeoBox` of a zoomed image.

    Given a source geobox and an integer scaler compute geobox of a scaled down image.

    Output geobox will be padded when shape is not a multiple of scaler.
    Example: ``5x4, scaler=2 -> 3x2``

    .. note::

       We assume that pixel coordinates are ``0,0`` at the top-left corner of a top-left pixel.

    """
    assert scaler > 1

    ny, nx = (X // scaler + (1 if X % scaler else 0) for X in src_geobox.shape)

    # Since 0,0 is at the corner of a pixel, not center, there is no
    # translation between pixel plane coords due to scaling
    A = src_geobox.transform * Affine.scale(scaler, scaler)

    return GeoBox((ny, nx), A, src_geobox.crs)


def _round_to_res(value: float, res: float) -> int:
    res = abs(res)
    return int(math.ceil((value - 0.1 * res) / res))


def flipy(gbox: GeoBox) -> GeoBox:
    """
    Flip along Y axis.

    :returns: GeoBox covering the same region but with Y-axis flipped
    """
    ny, _ = gbox.shape
    A = Affine.translation(0, ny) * Affine.scale(1, -1)
    A = gbox.affine * A
    return GeoBox(gbox.shape, A, gbox.crs)


def flipx(gbox: GeoBox) -> GeoBox:
    """
    Flip along X axis.

    :returns: GeoBox covering the same region but with X-axis flipped
    """
    _, nx = gbox.shape
    A = Affine.translation(nx, 0) * Affine.scale(-1, 1)
    A = gbox.affine * A
    return GeoBox(gbox.shape, A, gbox.crs)


def translate_pix(gbox: GeoBox, tx: float, ty: float) -> GeoBox:
    """
    Shift GeoBox in pixel plane.

    ``(0,0)`` of the new GeoBox will be at the same location as pixel ``(tx, ty)`` in the original
    GeoBox.
    """
    A = gbox.affine * Affine.translation(tx, ty)
    return GeoBox(gbox.shape, A, gbox.crs)


def pad(gbox: GeoBox, padx: int, pady: MaybeInt = None) -> GeoBox:
    """
    Pad geobox.

    Expand GeoBox by fixed number of pixels on each side
    """
    # false positive for -pady, it's never None by the time it runs
    # pylint: disable=invalid-unary-operand-type

    pady = padx if pady is None else pady

    ny, nx = gbox.shape
    A = gbox.affine * Affine.translation(-padx, -pady)
    shape = (ny + pady * 2, nx + padx * 2)
    return GeoBox(shape, A, gbox.crs)


def pad_wh(gbox: GeoBox, alignx: int = 16, aligny: MaybeInt = None) -> GeoBox:
    """
    Expand GeoBox such that width and height are multiples of supplied number.
    """
    aligny = alignx if aligny is None else aligny
    ny, nx = (align_up(sz, n) for sz, n in zip(gbox.shape, (aligny, alignx)))

    return GeoBox((ny, nx), gbox.affine, gbox.crs)


def zoom_out(gbox: GeoBox, factor: float) -> GeoBox:
    """
    Compute :py:class:`~odc.geo.geobox.GeoBox` with changed resolution.

    - ``factor > 1`` implies smaller width/height, fewer but bigger pixels
    - ``factor < 1`` implies bigger width/height, more but smaller pixels

    :returns:
       GeoBox covering the same region but with different pixels (i.e. lower or higher resolution)
    """

    ny, nx = (max(1, math.ceil(s / factor)) for s in gbox.shape)
    A = gbox.affine * Affine.scale(factor, factor)
    return GeoBox((ny, nx), A, gbox.crs)


def zoom_to(gbox: GeoBox, shape: SomeShape) -> GeoBox:
    """
    Change GeoBox shape.

    :returns:
      GeoBox covering the same region but with different number of pixels and therefore resolution.
    """
    shape = yx_(shape)
    sy, sx = (N / float(n) for N, n in zip(gbox.shape, shape.shape))
    A = gbox.affine * Affine.scale(sx, sy)
    return GeoBox(shape, A, gbox.crs)


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
    ny, nx = gbox.shape
    c0 = gbox.transform * (nx * 0.5, ny * 0.5)
    A = Affine.rotation(deg, c0) * gbox.transform
    return GeoBox(gbox.shape, A, gbox.crs)


def affine_transform_pix(gbox: GeoBox, transform: Affine) -> GeoBox:
    """
    Apply affine transform on pixel side.

    :param transform:
       Affine matrix mapping from new pixel coordinate space to pixel coordinate space of input gbox

    :returns:
      GeoBox of the same pixel shape but covering different region, pixels in the output gbox relate
      to input geobox via ``transform``

    ``X_old_pix = transform * X_new_pix``

    """
    A = gbox.affine * transform
    return GeoBox(gbox.shape, A, gbox.crs)


class GeoboxTiles:
    """Partition GeoBox into sub geoboxes."""

    def __init__(self, box: GeoBox, tile_shape: SomeShape):
        """
        Construct from a :py:class:`~odc.geo.GeoBox`.

        :param box: source :py:class:`~odc.geo.GeoBox`
        :param tile_shape: Shape of sub-tiles in pixels ``(rows, cols)``
        """
        tile_shape = yx_(tile_shape)
        self._gbox = box
        self._tile_shape = tile_shape
        self._shape = yx_(
            int(math.ceil(float(N) / n)) for N, n in zip(box.shape, tile_shape.shape)
        )
        self._cache: Dict[Index2d, GeoBox] = {}

    @property
    def base(self) -> GeoBox:
        """Access base Geobox"""
        return self._gbox

    @property
    def shape(self):
        """Number of tiles along each dimension."""
        return self._shape.shape

    def _idx_to_slice(self, idx: Index2d) -> Tuple[slice, slice]:
        def _slice(i, N, n) -> slice:
            _in = i * n
            if 0 <= _in < N:
                return slice(_in, min(_in + n, N))
            raise IndexError(f"Index {idx} is out of range")

        ir, ic = (
            _slice(i, N, n)
            for i, N, n in zip(idx.yx, self._gbox.shape, self._tile_shape.yx)
        )
        return (ir, ic)

    def chunk_shape(self, idx: SomeIndex2d) -> Tuple[int, int]:
        """
        Query chunk shape for a given chunk.

        :param idx: ``(row, col)`` chunk index
        :returns: ``(nrows, ncols)`` shape of a tile (edge tiles might be smaller)
        :raises: :py:class:`IndexError` when index is outside of ``[(0,0) -> .shape)``.
        """
        idx = iyx_(idx)

        def _sz(i: int, n: int, tile_sz: int, total_sz: int) -> int:
            if 0 <= i < n - 1:  # not edge tile
                return tile_sz
            if i == n - 1:  # edge tile
                return total_sz - (i * tile_sz)
            # out of index case
            raise IndexError(f"Index {idx} is out of range")

        n1, n2 = map(_sz, idx.yx, self._shape.yx, self._tile_shape.yx, self._gbox.shape)
        return (n1, n2)

    def __getitem__(self, idx: SomeIndex2d) -> GeoBox:
        """Lookup tile by index, index is in matrix access order: (row, col)

        :param idx: (row, col) index
        :returns: GeoBox of a tile
        :raises: IndexError when index is outside of [(0,0) -> .shape)
        """
        idx = iyx_(idx)
        sub_gbox = self._cache.get(idx, None)
        if sub_gbox is not None:
            return sub_gbox

        roi = self._idx_to_slice(idx)
        return self._cache.setdefault(idx, self._gbox[roi])

    def range_from_bbox(self, bbox: BoundingBox) -> Tuple[range, range]:
        """
        Intersect with a bounding box.

        Compute rows and columns overlapping with a given :py:class:`~odc.geo.geom.BoundingBox`.
        """

        def clamped_range(v1: float, v2: float, N: int) -> range:
            _in = clamp(math.floor(v1), 0, N)
            _out = clamp(math.ceil(v2), 0, N)
            return range(_in, _out)

        sy, sx = self._tile_shape.yx
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
