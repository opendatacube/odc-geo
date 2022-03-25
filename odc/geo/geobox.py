# This file is part of the Open Data Cube, see https://opendatacube.org for more information
#
# Copyright (c) 2015-2020 ODC Contributors
# SPDX-License-Identifier: Apache-2.0
import itertools
import math
from collections import OrderedDict, namedtuple
from textwrap import dedent
from typing import Dict, Iterable, List, Literal, Optional, Tuple, Union

import numpy
from affine import Affine

from .crs import CRS, MaybeCRS, norm_crs
from .geom import (
    BoundingBox,
    Geometry,
    bbox_intersection,
    bbox_union,
    line,
    multigeom,
    multiline,
    point,
    polygon_from_transform,
)
from .math import clamp, is_affine_st, is_almost_int
from .roi import align_up, polygon_path, roi_normalise, roi_shape
from .types import (
    XY,
    Index2d,
    MaybeInt,
    Resolution,
    Shape2d,
    SomeIndex2d,
    SomeResolution,
    SomeShape,
    iyx_,
    res_,
    resxy_,
    shape_,
    xy_,
)

OutlineMode = Union[
    Literal["native"], Literal["pixel"], Literal["geo"], Literal["auto"]
]

# pylint: disable=invalid-name,too-many-public-methods,too-many-lines
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


def _align_pix_tight(left: float, right: float, res: float) -> Tuple[float, int]:
    if res < 0:
        res = -res
        return right, max(1, int(math.ceil((right - left - 0.1 * res) / res)))

    return left, max(1, int(math.ceil((right - left - 0.1 * res) / res)))


class GeoBox:
    """
    Defines the location and resolution of a rectangular grid of data,
    including it's :py:class:`~odc.geo.crs.CRS`.

    :param shape: Shape in pixels ``(ny, nx)``
    :param crs: Coordinate Reference System
    :param affine: Affine transformation defining the location of the geobox
    """

    __slots__ = ("_shape", "_affine", "_crs", "_extent")

    def __init__(self, shape: SomeShape, affine: Affine, crs: MaybeCRS):
        shape = shape_(shape)

        self._shape = shape
        self._affine = affine
        self._crs = norm_crs(crs)
        self._extent: Optional[Geometry] = None

    @staticmethod
    def from_bbox(
        bbox: Union[BoundingBox, Tuple[float, float, float, float]],
        crs: MaybeCRS = None,
        *,
        tight: bool = False,
        shape: Optional[SomeShape] = None,
        resolution: Optional[SomeResolution] = None,
        align: Optional[XY[float]] = None,
    ) -> "GeoBox":
        """
        Construct :py:class:`~odc.geo.geobox.GeoBox` from a bounding box.

        :param bbox: Bounding box in CRS units, lonlat is assumed when ``crs`` is not supplied
        :param crs: CRS of the bounding box (defaults to EPSG:4326)
        :param tight: Supplying ``tight=True`` turns off pixel snapping.
        :param shape: Span that many pixels.
        :param resolution: Use specified resolution
        :param align:
        :return: :py:class:`~odc.geo.geobox.GeoBox` that covers supplied bounding box.
        """

        if align is None and tight is False:
            align = xy_(0, 0)
        if crs is None:
            crs = "EPSG:4326"

        if not isinstance(bbox, BoundingBox):
            bbox = BoundingBox(*bbox)

        if resolution is not None:
            rx, ry = res_(resolution).xy
            if align is None:
                offx, nx = _align_pix_tight(bbox.left, bbox.right, rx)
                offy, ny = _align_pix_tight(bbox.bottom, bbox.top, ry)
            else:
                offx, nx = _align_pix(bbox.left, bbox.right, rx, align.x)
                offy, ny = _align_pix(bbox.bottom, bbox.top, ry, align.y)

            affine = Affine.translation(offx, offy) * Affine.scale(rx, ry)
            return GeoBox((ny, nx), crs=crs, affine=affine)

        if shape is None:
            raise ValueError("Must supply shape or resolution")

        shape = shape_(shape)
        nx, ny = shape.wh
        rx = bbox.span_x / nx
        ry = -bbox.span_y / ny

        if align is None:
            offx, offy = bbox.left, bbox.top
        else:
            offx, _ = _align_pix(bbox.left, bbox.right, rx, align.x)
            offy, _ = _align_pix(bbox.bottom, bbox.top, ry, align.y)

        affine = Affine.translation(offx, offy) * Affine.scale(rx, ry)
        return GeoBox((ny, nx), crs=crs, affine=affine)

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

        return GeoBox.from_bbox(
            geopolygon.boundingbox, crs, resolution=resolution, align=align
        )

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
        affine = self._affine * Affine.translation(-bx, -by)

        ny, nx = (sz + 2 * b for sz, b in zip(self._shape, (by, bx)))

        return GeoBox(
            (ny, nx),
            affine=affine,
            crs=self._crs,
        )

    def __getitem__(self, roi) -> "GeoBox":
        if isinstance(roi, int):
            roi = (slice(roi, roi + 1), slice(None, None))

        if isinstance(roi, slice):
            roi = (roi, slice(None, None))

        if len(roi) > 2:
            raise ValueError("Expect 2d slice")

        roi = roi_normalise(roi, self._shape.shape)

        if not all(s.step is None or s.step == 1 for s in roi):
            raise NotImplementedError("scaling not implemented, yet")

        ty, tx = (s.start for s in roi)
        ny, nx = roi_shape(roi)

        affine = self._affine * Affine.translation(tx, ty)

        return GeoBox(shape=(ny, nx), affine=affine, crs=self._crs)

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
        return hash((*self._shape, self._crs, self._affine))

    @property
    def transform(self) -> Affine:
        """Linear mapping from pixel space to CRS."""
        return self._affine

    @property
    def affine(self) -> Affine:
        """
        Linear mapping from pixel space to CRS.

        alias for :py:attr:`~odc.geo.geobox.GeoBox.transform`
        """
        return self._affine

    @property
    def width(self) -> int:
        """Width in pixels (nx)."""
        return self._shape.x

    @property
    def height(self) -> int:
        """Height in pixels (ny)."""
        return self._shape.y

    @property
    def shape(self) -> Shape2d:
        """Shape in pixels ``(height, width)``."""
        return self._shape

    @property
    def crs(self) -> Optional[CRS]:
        """Coordinate Reference System of the GeoBox."""
        return self._crs

    @property
    def dimensions(self) -> Tuple[str, str]:
        """List of dimension names of the GeoBox."""
        crs = self._crs
        if crs is None:
            return ("y", "x")
        return crs.dimensions

    @property
    def resolution(self) -> Resolution:
        """Resolution, pixel size in CRS units."""
        rx, _, _, _, ry, *_ = self._affine
        return resxy_(rx, ry)

    @property
    def alignment(self) -> XY[float]:
        """
        Alignment of pixel boundaries in CRS units.

        This is usally ``(0,0)``.
        """
        rx, _, tx, _, ry, ty, *_ = self._affine
        return xy_(tx % abs(rx), ty % abs(ry))

    @property
    def coordinates(self) -> Dict[str, Coordinate]:
        """
        Query coordinates.

        This method only works with axis-aligned boxes. It will raise :py:class:`ValueError` if called
        on non-axis aligned :py:class:`~odc.geo.geobox.GeoBox`.

        :raises: :py:class:`ValueError` if not axis aligned.

        :return:
          Mapping from coordinate name to :py:class:`~odc.geo.geobox.Coordinate`.
        """
        self._confirm_axis_aligned("Only axis aligned GeoBox can do this.")
        rx, _, tx, _, ry, ty, *_ = self._affine
        ny, nx = self._shape

        xs = numpy.arange(nx) * rx + (tx + rx / 2)
        ys = numpy.arange(ny) * ry + (ty + ry / 2)

        units = self._crs.units if self._crs is not None else ("1", "1")

        return OrderedDict(
            (dim, Coordinate(labels, units, res))
            for dim, labels, units, res in zip(
                self.dimensions, (ys, xs), units, (ry, rx)
            )
        )

    @property
    def extent(self) -> Geometry:
        """GeoBox footprint in native CRS."""
        if self._extent is not None:
            return self._extent
        _extent = polygon_from_transform(self._shape, self._affine, crs=self._crs)
        self._extent = _extent
        return _extent

    def _reproject_resolution(self, npoints: int = 100):
        bbox = self.extent.boundingbox
        span = max(bbox.span_x, bbox.span_y)
        return span / npoints

    @property
    def geographic_extent(self) -> Geometry:
        """GeoBox extent in EPSG:4326."""
        if self._crs is None or self._crs.geographic:
            return self.extent

        return self.extent.to_crs(
            CRS("EPSG:4326"), resolution=self._reproject_resolution()
        )

    coords = coordinates
    dims = dimensions

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f"GeoBox({self._shape.yx!r}, {self._affine!r}, {self._crs!r})"

    def __eq__(self, other):
        if not isinstance(other, GeoBox):
            return False

        return (
            self._shape == other._shape
            and self._affine == other._affine
            and self._crs == other._crs
        )

    def __rmul__(self, transform: Affine) -> "GeoBox":
        """
        Apply affine transform on CRS side.

        This has effect of transforming footprint of the source via ``transform``.

        :param transform:
           Affine matrix that shifts footprint of the source geobox.

        :return:
           :py:class:`~odc.geo.gebox.GeoBox` of the same pixel shape but
           covering different region.
        """
        return GeoBox(self._shape, transform * self._affine, self._crs)

    def __mul__(self, transform: Affine) -> "GeoBox":
        """
        Apply affine transform on pixel side.

        ``X_old_pix = transform * X_new_pix``

        :param transform:
           Affine matrix mapping from new pixel coordinate space to pixel coordinate
           space of input geobox.

        :returns:
          :py:class:`~odc.geo.gebox.GeoBox` of the same pixel shape but covering different
          region. Pixel coordinates in the output relate to input coordinates via ``transform``.
        """
        return GeoBox(self._shape, self._affine * transform, self._crs)

    def pad(self, padx: int, pady: MaybeInt = None) -> "GeoBox":
        """
        Pad geobox.

        Expand GeoBox by fixed number of pixels on each side
        """
        # false positive for -pady, it's never None by the time it runs
        # pylint: disable=invalid-unary-operand-type

        pady = padx if pady is None else pady

        ny, nx = self._shape.yx
        A = self._affine * Affine.translation(-padx, -pady)
        shape = (ny + pady * 2, nx + padx * 2)
        return GeoBox(shape, A, self._crs)

    def pad_wh(self, alignx: int = 16, aligny: MaybeInt = None) -> "GeoBox":
        """
        Possibly expand :py:class:`~odc.geo.geobox.GeoBox` by a few pixels.

        Find nearest ``width``/``height`` that are multiples of the desired factor. And return a new
        geobox that is slighly taller and/or wider covering roughly the same region. The new geobox
        will have the same CRS and transform but possibly larger shape.
        """
        aligny = alignx if aligny is None else aligny
        ny, nx = (align_up(sz, n) for sz, n in zip(self._shape.yx, (aligny, alignx)))

        return GeoBox((ny, nx), self._affine, self._crs)

    def zoom_out(self, factor: float) -> "GeoBox":
        """
        Compute :py:class:`~odc.geo.geobox.GeoBox` with changed resolution.

        - ``factor > 1`` implies smaller width/height, fewer but bigger pixels
        - ``factor < 1`` implies bigger width/height, more but smaller pixels

        :returns:
           GeoBox covering the same region but with different pixels (i.e. lower or higher resolution)
        """

        ny, nx = (max(1, math.ceil(s / factor)) for s in self.shape)
        A = self._affine * Affine.scale(factor, factor)
        return GeoBox((ny, nx), A, self._crs)

    def zoom_to(self, shape: SomeShape) -> "GeoBox":
        """
        Change GeoBox shape.

        :returns:
          GeoBox covering the same region but with different number of pixels and therefore resolution.
        """
        shape = shape_(shape)
        sy, sx = (N / float(n) for N, n in zip(self._shape, shape.shape))
        A = self._affine * Affine.scale(sx, sy)
        return GeoBox(shape, A, self._crs)

    def flipy(self) -> "GeoBox":
        """
        Flip along Y axis.

        :returns: GeoBox covering the same region but with Y-axis flipped
        """
        ny, _ = self._shape
        A = Affine.translation(0, ny) * Affine.scale(1, -1)
        return self * A

    def flipx(self) -> "GeoBox":
        """
        Flip along X axis.

        :returns: GeoBox covering the same region but with X-axis flipped
        """
        _, nx = self._shape
        A = Affine.translation(nx, 0) * Affine.scale(-1, 1)
        return self * A

    def translate_pix(self, tx: float, ty: float) -> "GeoBox":
        """
        Shift GeoBox in pixel plane.

        ``(0,0)`` of the new GeoBox will be at the same location as pixel ``(tx, ty)`` in the original
        GeoBox.
        """
        return self * Affine.translation(tx, ty)

    def rotate(self, deg: float) -> "GeoBox":
        """
        Rotate GeoBox around the center.

        It's as if you stick a needle through the center of the GeoBox footprint
        and rotate it counter clock wise by supplied number of degrees.

        Note that from the pixel point of view image rotates the other way. If you have
        source image with an arrow pointing right, and you rotate GeoBox 90 degrees,
        in that view arrow should point down (this is assuming usual case of inverted
        y-axis)
        """
        ny, nx = self._shape
        c0 = self._affine * (nx * 0.5, ny * 0.5)
        return Affine.rotation(deg, c0) * self

    def boundary(self, pts_per_side: int = 16) -> numpy.ndarray:
        """
        Boundary of a :py:class:`~odc.geo.geobox.GeoBox`.

        Construct a ring of points in pixel space along the edge of the geobox.

        :param pts_per_side: Number of points per side, default is 16.

        :return:
          Points in pixel space along the perimeter of a GeoBox as a ``Nx2`` array
          in pixel coordinates.
        """
        ny, nx = self._shape.yx
        xx = numpy.linspace(0, nx, pts_per_side, dtype="float32")
        yy = numpy.linspace(0, ny, pts_per_side, dtype="float32")

        return polygon_path(xx, yy).T[:-1]

    def _confirm_axis_aligned(self, raise_error: Optional[str] = None) -> bool:
        if is_affine_st(self._affine):
            return True
        if raise_error is not None:
            raise ValueError(raise_error)
        return False

    @property
    def center_pixel(self) -> "GeoBox":
        """
        GeoBox of a center pixel.
        """
        return self[self.shape.map(lambda x: x // 2).yx]

    def svg(
        self,
        scale_factor: float = 1.0,
        mode: OutlineMode = "auto",
        notch: float = 0.0,
        grid_stroke: str = "pink",
    ) -> str:
        """
        Produce SVG paths.

        :param mode: One of pixel, native, geo (default is geo)
        :return: SVG path
        """
        if mode == "auto":
            mode = "native" if self._crs is None else "geo"

        grids = self.grid_lines(mode=mode)
        outline = self.outline(mode, notch=notch)

        grid_svg = (
            '<path fill="none" opacity="0.8"'
            f' stroke-width="{0.8*scale_factor}"'
            f' stroke="{grid_stroke}"'
            f' d="{grids.svg_path()}" />'
        )

        return outline.svg(scale_factor) + grid_svg

    def grid_lines(self, step: int = 0, mode: OutlineMode = "native") -> Geometry:
        """
        Construct pixel edge aligned grid lines.
        """
        from .ui import pick_grid_step  # pylint: disable=import-outside-toplevel

        nx, ny = self._shape.xy
        if step == 0:
            step = pick_grid_step(max(nx, ny))
        xx = [*range(0, nx, step), nx]
        yy = [*range(0, ny, step), ny]
        vertical = [list(itertools.product([x], yy)) for x in xx[1:-1]]
        horizontal = [list(itertools.product(xx, [y])) for y in yy[1:-1]]
        lines = multiline(vertical + horizontal, self._crs)

        if mode == "pixel":
            return lines

        lines = lines.transform(self._affine)
        if mode == "native":
            return lines

        dx, dy = self._affine * (step, 0)
        res = math.sqrt(dx * dx + dy * dy) / 5
        return lines.to_crs("epsg:4326", resolution=res)

    def outline(self, mode: OutlineMode = "native", notch: float = 0.1) -> Geometry:
        """
        Produce Line Geometry around perimeter.

        .. code-block:: txt

             +---+-------------+
             |   |             |
             +---+             |
             |                 |
             |                 |
             +-----------------+
        """

        assert notch < 1
        w, h = self._shape.wh
        if notch > 0:
            nn = min(notch * max(w, h), w, h)
            pix = line(
                [
                    (0, nn),
                    (0, 0),
                    (nn, 0),
                    (w, 0),
                    (w, h),
                    (0, h),
                    (0, nn),
                    (nn, nn),
                    (nn, 0),
                ],
                self._crs,
            )
        else:
            pix = multigeom(
                [
                    line([(0, 0), (w, 0), (w, h), (0, h), (0, 0)], self._crs),
                    point(0, 0, self._crs),
                ]
            )
        if mode == "pixel":
            return pix

        native = pix.transform(self._affine)
        if mode == "native":
            return native

        # about 100 pts per side
        bbox = native.boundingbox
        res = max(bbox.span_x, bbox.span_y) / 100

        return native.to_crs("EPSG:4326", resolution=res)

    def _display_bbox(self, pad_fraction: float = 0.1):
        bbox = self.geographic_extent.boundingbox
        pad_deg = max(bbox.span_x, bbox.span_y) * pad_fraction
        return bbox.buffered(pad_deg)

    def _render_svg(self, sz=360):
        # pylint: disable=import-outside-toplevel
        from .ui import make_svg, svg_base_map

        if self._crs is None:
            bbox = self.extent.boundingbox
            margin = 0.1 * max(bbox.span_x, bbox.span_y)
            bbox = bbox.buffered(margin)
            return make_svg(
                self,
                bbox=bbox,
                sz=sz,
            )

        return svg_base_map(self, bbox=self._display_bbox(), sz=sz)

    def _repr_svg_(self):
        return self._render_svg()

    def _repr_html_(self):
        # pylint: disable=import-outside-toplevel,too-many-locals
        from .data import gbox_css
        from .ui import pick_grid_step, svg_base_map

        unit_remaps = {"metre": "m", "meter": "m"}

        W, H = self._shape.wh
        grid_step = pick_grid_step(max(W, H))
        svg_zoomed_txt = self._render_svg(sz=320)

        crs = self._crs
        if crs is None:
            epsg = "not set"
            wkt = "not set"
            units = ""
            svg_global_txt = ""
        else:
            epsg = "undefined" if crs.epsg is None else f"{crs.epsg:d}"
            wkt = crs.to_wkt(pretty=True).replace("\n", "<br/>").replace(" ", "&nbsp;")
            units = crs.units[0]
            svg_global_txt = svg_base_map(
                sz=200, target=self.geographic_extent.centroid.coords[0]
            )

        units = unit_remaps.get(units, units)

        pix_sz = max(*self.resolution.map(abs).xy)

        info = [
            ("Dimensions", f"{W:,d}x{H:,d}"),
            ("EPSG", f"{epsg}"),
            ("Resolution", f"{pix_sz:g}{units}"),
            ("Cell", f"{grid_step:,d}px"),
        ]

        info_html = "\n".join(
            [
                (
                    f'<div class="row"><div class="column">{hdr}</div>'
                    f'<div class="column value">{val}</div></div>'
                )
                for hdr, val in info
            ]
        )

        return dedent(
            f"""\
        <style>{gbox_css()}</style>
        <div class="gbox-info">
        <h4>GeoBox</h4>
        <div class="row">
            <div class="column">
                <div class="info-box">
                    {info_html}
                    <div>{svg_global_txt}</div>
                </div>
            </div>
            <div class="column svg-zoomed">{svg_zoomed_txt}</div>
        </div>
        <details>
            <summary>WKT</summary>
            <div class="wkt">{wkt}</div>
        </details>
        </div>"""
        )


def gbox_boundary(gbox: GeoBox, pts_per_side: int = 16) -> numpy.ndarray:
    """Alias for :py:meth:`odc.geo.geobox.GeoBox.boundary`."""
    return gbox.boundary(pts_per_side)


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
    """Alias for :py:meth:`odc.geo.geobox.flipy`."""
    return gbox.flipy()


def flipx(gbox: GeoBox) -> GeoBox:
    """Alias for :py:meth:`odc.geo.geobox.flipx`."""
    return gbox.flipx()


def translate_pix(gbox: GeoBox, tx: float, ty: float) -> GeoBox:
    """Alias for :py:meth:`odc.geo.geobox.GeoBox.translate_pix`."""
    return gbox.translate_pix(tx, ty)


def pad(gbox: GeoBox, padx: int, pady: MaybeInt = None) -> GeoBox:
    """Alias for :py:meth:`odc.geo.geobox.GeoBox.pad`."""
    return gbox.pad(padx, pady)


def pad_wh(gbox: GeoBox, alignx: int = 16, aligny: MaybeInt = None) -> GeoBox:
    """Alias for :py:meth:`odc.geo.geobox.GeoBox.pad_wh`."""
    return gbox.pad_wh(alignx, aligny)


def zoom_out(gbox: GeoBox, factor: float) -> GeoBox:
    """Alias for :py:meth:`odc.geo.geobox.GeoBox.zoom_out`."""
    return gbox.zoom_out(factor)


def zoom_to(gbox: GeoBox, shape: SomeShape) -> GeoBox:
    """Alias for :py:meth:`odc.geo.geobox.GeoBox.zoom_to`."""
    return gbox.zoom_to(shape)


def rotate(gbox: GeoBox, deg: float) -> GeoBox:
    """Alias for :py:meth:`odc.geo.geobox.GeoBox.`."""
    return gbox.rotate(deg)


def affine_transform_pix(gbox: GeoBox, transform: Affine) -> GeoBox:
    """Alias for :py:meth:`odc.geo.geobox.GeoBox.__mul__`."""
    return gbox * transform


class GeoboxTiles:
    """Partition GeoBox into sub geoboxes."""

    def __init__(self, box: GeoBox, tile_shape: SomeShape):
        """
        Construct from a :py:class:`~odc.geo.GeoBox`.

        :param box: source :py:class:`~odc.geo.GeoBox`
        :param tile_shape: Shape of sub-tiles in pixels ``(rows, cols)``
        """
        tile_shape = shape_(tile_shape)
        self._gbox = box
        self._tile_shape = tile_shape
        ny, nx = (
            int(math.ceil(float(N) / n)) for N, n in zip(box.shape, tile_shape.shape)
        )
        self._shape = shape_((ny, nx))
        self._cache: Dict[Index2d, GeoBox] = {}

    @property
    def base(self) -> GeoBox:
        """Access base Geobox"""
        return self._gbox

    @property
    def shape(self):
        """Number of tiles along each dimension."""
        return self._shape

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

    def chunk_shape(self, idx: SomeIndex2d) -> Shape2d:
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
        return shape_((n1, n2))

    def __getitem__(self, idx: SomeIndex2d) -> GeoBox:
        """
        Lookup tile by index, index is in matrix access order: ``(row, col)``.

        :param idx: ``(row, col)`` index
        :returns: GeoBox of a tile
        :raises: IndexError when index is outside of ``[(0,0) -> .shape)``
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

        NY, NX = self._shape
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
