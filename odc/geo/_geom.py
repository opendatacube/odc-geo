# This file is part of the Open Data Cube, see https://opendatacube.org for more information
#
# Copyright (c) 2015-2020 ODC Contributors
# SPDX-License-Identifier: Apache-2.0
import array
import functools
import itertools
import math
from collections import namedtuple
from collections.abc import Sequence
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union

import numpy
from affine import Affine
from shapely import geometry, ops
from shapely.geometry import base

from .crs import CRS, CRSMismatchError, MaybeCRS, SomeCRS, norm_crs, norm_crs_or_error

_BoundingBox = namedtuple("_BoundingBox", ("left", "bottom", "right", "top"))
CoordList = List[Tuple[float, float]]


class BoundingBox(_BoundingBox):
    """Bounding box, defining extent in cartesian coordinates."""

    def buffered(self, xbuff: float, ybuff: Optional[float] = None) -> "BoundingBox":
        """
        Return a new BoundingBox, buffered in the x and y dimensions.

        :param xbuff: X dimension buffering amount
        :param ybuff: Y dimension buffering amount
        :return: new BoundingBox
        """
        if ybuff is None:
            ybuff = xbuff

        return BoundingBox(
            left=self.left - xbuff,
            bottom=self.bottom - ybuff,
            right=self.right + xbuff,
            top=self.top + ybuff,
        )

    @property
    def span_x(self) -> float:
        return self.right - self.left

    @property
    def span_y(self) -> float:
        return self.top - self.bottom

    @property
    def width(self) -> int:
        return int(self.right - self.left)

    @property
    def height(self) -> int:
        return int(self.top - self.bottom)

    @property
    def range_x(self) -> Tuple[float, float]:
        return (self.left, self.right)

    @property
    def range_y(self) -> Tuple[float, float]:
        return (self.bottom, self.top)

    @property
    def points(self) -> CoordList:
        """Extract four corners of the bounding box"""
        x0, y0, x1, y1 = self
        return list(itertools.product((x0, x1), (y0, y1)))

    def transform(self, transform: Affine) -> "BoundingBox":
        """Transform bounding box through a linear transform

        Apply linear transform on 4 points of the bounding box and compute
        bounding box of these four points.
        """
        pts = [transform * pt for pt in self.points]
        xx = [x for x, _ in pts]
        yy = [y for _, y in pts]
        return BoundingBox(min(xx), min(yy), max(xx), max(yy))

    @staticmethod
    def from_xy(x: Tuple[float, float], y: Tuple[float, float]) -> "BoundingBox":
        """BoundingBox from x and y ranges

        :param x: (left, right)
        :param y: (bottom, top)
        """
        x1, x2 = sorted(x)
        y1, y2 = sorted(y)
        return BoundingBox(x1, y1, x2, y2)

    @staticmethod
    def from_points(p1: Tuple[float, float], p2: Tuple[float, float]) -> "BoundingBox":
        """BoundingBox from 2 points
        :param p1: (x, y)
        :param p2: (x, y)
        """
        return BoundingBox.from_xy((p1[0], p2[0]), (p1[1], p2[1]))


def wrap_shapely(method):
    """
    Takes a method that expects shapely geometry arguments
    and converts it to a method that operates on `Geometry`
    objects that carry their CRSs.
    """

    @functools.wraps(method, assigned=("__doc__",))
    def wrapped(*args):
        first = args[0]
        for arg in args[1:]:
            if first.crs != arg.crs:
                raise CRSMismatchError((first.crs, arg.crs))

        result = method(*[arg.geom for arg in args])
        if isinstance(result, base.BaseGeometry):
            return Geometry(result, first.crs)
        return result

    return wrapped


def force_2d(geojson: Dict[str, Any]) -> Dict[str, Any]:
    assert "type" in geojson
    assert "coordinates" in geojson

    def is_scalar(x):
        return isinstance(x, (int, float))

    def go(x):
        if is_scalar(x):
            return x

        if isinstance(x, Sequence):
            if all(is_scalar(y) for y in x):
                return x[:2]
            return [go(y) for y in x]

        raise ValueError(f"invalid coordinate {x}")

    return {"type": geojson["type"], "coordinates": go(geojson["coordinates"])}


def densify(coords: CoordList, resolution: float) -> CoordList:
    """
    Adds points so they are at most `resolution` units apart.
    """
    d2 = resolution ** 2

    def short_enough(p1, p2):
        return (p1[0] ** 2 + p2[0] ** 2) < d2

    new_coords = [coords[0]]
    for p1, p2 in zip(coords[:-1], coords[1:]):
        if not short_enough(p1, p2):
            segment = geometry.LineString([p1, p2])
            segment_length = segment.length
            d = resolution
            while d < segment_length:
                (pt,) = segment.interpolate(d).coords
                new_coords.append(pt)
                d += resolution

        new_coords.append(p2)

    return new_coords


def _clone_shapely_geom(geom: base.BaseGeometry) -> base.BaseGeometry:
    return type(geom)(geom)


class Geometry:
    """
    2D Geometry with CRS

    Instantiate with a GeoJSON structure

    If 3D coordinates are supplied, they are converted to 2D by dropping the Z points.
    """

    # pylint: disable=protected-access, too-many-public-methods

    def __init__(
        self,
        geom: Union[base.BaseGeometry, Dict[str, Any], "Geometry"],
        crs: MaybeCRS = None,
    ):
        if isinstance(geom, Geometry):
            assert crs is None
            self.crs: Optional[CRS] = geom.crs
            self.geom: base.BaseGeometry = _clone_shapely_geom(geom.geom)
            return

        crs = norm_crs(crs)
        self.crs = crs
        if isinstance(geom, base.BaseGeometry):
            self.geom = geom
        elif isinstance(geom, dict):
            self.geom = geometry.shape(force_2d(geom))
        else:
            raise ValueError(f"Unexpected type {type(geom)}")

    def clone(self) -> "Geometry":
        return Geometry(self)

    @wrap_shapely
    def contains(self, other: "Geometry") -> bool:
        return self.contains(other)

    @wrap_shapely
    def crosses(self, other: "Geometry") -> bool:
        return self.crosses(other)

    @wrap_shapely
    def disjoint(self, other: "Geometry") -> bool:
        return self.disjoint(other)

    @wrap_shapely
    def intersects(self, other: "Geometry") -> bool:
        return self.intersects(other)

    @wrap_shapely
    def touches(self, other: "Geometry") -> bool:
        return self.touches(other)

    @wrap_shapely
    def within(self, other: "Geometry") -> bool:
        return self.within(other)

    @wrap_shapely
    def overlaps(self, other: "Geometry") -> bool:
        return self.overlaps(other)

    @wrap_shapely
    def difference(self, other: "Geometry") -> "Geometry":
        return self.difference(other)

    @wrap_shapely
    def intersection(self, other: "Geometry") -> "Geometry":
        return self.intersection(other)

    @wrap_shapely
    def symmetric_difference(self, other: "Geometry") -> "Geometry":
        return self.symmetric_difference(other)

    @wrap_shapely
    def union(self, other: "Geometry") -> "Geometry":
        return self.union(other)

    @wrap_shapely
    def __and__(self, other: "Geometry") -> "Geometry":
        return self.__and__(other)

    @wrap_shapely
    def __or__(self, other: "Geometry") -> "Geometry":
        return self.__or__(other)

    @wrap_shapely
    def __xor__(self, other: "Geometry") -> "Geometry":
        return self.__xor__(other)

    @wrap_shapely
    def __sub__(self, other: "Geometry") -> "Geometry":
        return self.__sub__(other)

    def svg(self) -> str:
        return self.geom.svg()

    def _repr_svg_(self) -> str:
        return self.geom._repr_svg_()

    @property
    def type(self) -> str:
        return self.geom.type

    @property
    def is_empty(self) -> bool:
        return self.geom.is_empty

    @property
    def is_valid(self) -> bool:
        return self.geom.is_valid

    @property
    def boundary(self) -> "Geometry":
        return Geometry(self.geom.boundary, self.crs)

    @property
    def exterior(self) -> "Geometry":
        return Geometry(self.geom.exterior, self.crs)

    @property
    def interiors(self) -> List["Geometry"]:
        return [Geometry(g, self.crs) for g in self.geom.interiors]

    @property
    def centroid(self) -> "Geometry":
        return Geometry(self.geom.centroid, self.crs)

    @property
    def coords(self) -> CoordList:
        return list(self.geom.coords)

    @property
    def points(self) -> CoordList:
        return self.coords

    @property
    def length(self) -> float:
        return self.geom.length

    @property
    def area(self) -> float:
        return self.geom.area

    @property
    def xy(self) -> Tuple[array.array, array.array]:
        return self.geom.xy

    @property
    def convex_hull(self) -> "Geometry":
        return Geometry(self.geom.convex_hull, self.crs)

    @property
    def envelope(self) -> "Geometry":
        return Geometry(self.geom.envelope, self.crs)

    @property
    def boundingbox(self) -> BoundingBox:
        minx, miny, maxx, maxy = self.geom.bounds
        return BoundingBox(left=minx, right=maxx, bottom=miny, top=maxy)

    @property
    def wkt(self) -> str:
        return self.geom.wkt

    @property
    def __array_interface__(self):
        return self.geom.__array_interface__

    @property
    def __geo_interface__(self):
        return self.geom.__geo_interface__

    @property
    def json(self):
        return self.__geo_interface__

    def segmented(self, resolution: float) -> "Geometry":
        """
        Possibly add more points to the geometry so that no edge is longer than `resolution`.
        """

        def segmentize_shapely(geom: base.BaseGeometry) -> base.BaseGeometry:
            if geom.type in ["Point", "MultiPoint"]:
                return type(geom)(geom)  # clone without changes

            if geom.type in ["GeometryCollection", "MultiPolygon", "MultiLineString"]:
                return type(geom)([segmentize_shapely(g) for g in geom.geoms])

            if geom.type in ["LineString", "LinearRing"]:
                return type(geom)(densify(list(geom.coords), resolution))

            if geom.type == "Polygon":
                return geometry.Polygon(
                    densify(list(geom.exterior.coords), resolution),
                    [densify(list(i.coords), resolution) for i in geom.interiors],
                )

            raise ValueError(f"unknown geometry type {geom.type}")  # pragma: no cover

        return Geometry(segmentize_shapely(self.geom), self.crs)

    def interpolate(self, distance: float) -> "Geometry":
        """
        Returns a point distance units along the line.
        Raises TypeError if geometry doesn't support this operation.
        """
        return Geometry(self.geom.interpolate(distance), self.crs)

    def buffer(self, distance: float, resolution: float = 30) -> "Geometry":
        return Geometry(self.geom.buffer(distance, resolution=resolution), self.crs)

    def simplify(self, tolerance: float, preserve_topology: bool = True) -> "Geometry":
        return Geometry(
            self.geom.simplify(tolerance, preserve_topology=preserve_topology), self.crs
        )

    def transform(self, func) -> "Geometry":
        """Applies func to all coordinates of Geometry and returns a new Geometry
        of the same type and in the same projection from the transformed coordinates.

        func maps x, y, and optionally z to output xp, yp, zp. The input
        parameters may be iterable types like lists or arrays or single values.
        The output shall be of the same type: scalars in, scalars out; lists
        in, lists out.
        """
        return Geometry(ops.transform(func, self.geom), self.crs)

    def _to_crs(self, crs: CRS) -> "Geometry":
        assert self.crs is not None
        return Geometry(ops.transform(self.crs.transformer_to_crs(crs), self.geom), crs)

    def to_crs(
        self,
        crs: SomeCRS,
        resolution: Optional[float] = None,
        wrapdateline: bool = False,
    ) -> "Geometry":
        """
        Convert geometry to a different Coordinate Reference System

        :param crs: CRS to convert to

        :param resolution: Subdivide the geometry such it has no segment longer then the given distance.
                           Defaults to 1 degree for geographic and 100km for projected. To disable
                           completely use Infinity float('+inf')

        :param wrapdateline: Attempt to gracefully handle geometry that intersects the dateline
                                  when converting to geographic projections.
                                  Currently only works in few specific cases (source CRS is smooth over the dateline).
        """
        crs = norm_crs_or_error(crs)
        if self.crs == crs:
            return self

        if self.crs is None:
            raise ValueError("Cannot project geometries without CRS")

        if resolution is None:
            resolution = 1 if self.crs.geographic else 100000

        geom = self.segmented(resolution) if math.isfinite(resolution) else self

        eps = 1e-4
        if wrapdateline and crs.geographic:
            # TODO: derive precision from resolution by converting to degrees
            precision = 0.1
            chopped = chop_along_antimeridian(geom, precision)
            chopped_lonlat = chopped._to_crs(crs)
            return clip_lon180(chopped_lonlat, eps)

        return geom._to_crs(crs)

    def split(self, splitter: "Geometry") -> Iterable["Geometry"]:
        """shapely.ops.split"""
        if splitter.crs != self.crs:
            raise CRSMismatchError(self.crs, splitter.crs)

        for g in ops.split(self.geom, splitter.geom).geoms:
            yield Geometry(g, self.crs)

    def __iter__(self) -> Iterator["Geometry"]:
        for geom in self.geom.geoms:
            yield Geometry(geom, self.crs)

    def __nonzero__(self) -> bool:
        return not self.is_empty

    def __bool__(self) -> bool:
        return not self.is_empty

    def __eq__(self, other: Any) -> bool:
        return (
            hasattr(other, "crs")
            and self.crs == other.crs
            and hasattr(other, "geom")
            and self.geom == other.geom
        )

    def __str__(self):
        return f"Geometry({self.__geo_interface__}, {self.crs!r})"

    def __repr__(self):
        return f"Geometry({self.geom}, {self.crs})"

    # Implement pickle/unpickle
    # It does work without these two methods, but gdal/ogr prints 'ERROR 1: Empty geometries cannot be constructed'
    # when unpickling, which is quite unpleasant.
    def __getstate__(self):
        return {"geom": self.json, "crs": self.crs}

    def __setstate__(self, state):
        self.__init__(**state)


def common_crs(geoms: Iterable[Geometry]) -> Optional[CRS]:
    """Return CRS common across geometries, or raise CRSMismatchError"""
    all_crs = [g.crs for g in geoms]
    if len(all_crs) == 0:
        return None
    ref = all_crs[0]
    for crs in all_crs[1:]:
        if crs != ref:
            raise CRSMismatchError()
    return ref


def projected_lon(
    crs: MaybeCRS,
    lon: float,
    lat: Tuple[float, float] = (-90.0, 90.0),
    step: float = 1.0,
) -> Geometry:
    """Project vertical line along some longitude into given CRS."""
    crs = norm_crs_or_error(crs)
    yy = numpy.arange(lat[0], lat[1], step, dtype="float32")
    xx = numpy.full_like(yy, lon)
    tr = CRS("EPSG:4326").transformer_to_crs(crs)
    xx_, yy_ = tr(xx, yy)
    pts = [
        (float(x), float(y))
        for x, y in zip(xx_, yy_)
        if math.isfinite(x) and math.isfinite(y)
    ]
    return line(pts, crs)


def clip_lon180(geom: Geometry, tol=1e-6) -> Geometry:
    """For every point in the ``lon=180|-180`` band clip to either 180 or -180
    180|-180 is decided based on where the majority of other points lie.

    NOTE: this will only do "right thing" for chopped geometries,
          expectation is that all the points are to one side of lon=180
          line, or in the the capture zone of lon=(+/-)180
    """
    thresh = 180 - tol

    def _clip_180(xx, clip):
        return [x if abs(x) < thresh else clip for x in xx]

    def _pick_clip(xx: List[float]):
        cc = 0
        for x in xx:
            if abs(x) < thresh:
                cc += 1 if x > 0 else -1
        return 180 if cc >= 0 else -180

    def transformer(xx, yy):
        clip = _pick_clip(xx)
        return _clip_180(xx, clip), yy

    if geom.type.startswith("Multi"):
        return multigeom(g.transform(transformer) for g in geom)

    return geom.transform(transformer)


def chop_along_antimeridian(geom: Geometry, precision: float = 0.1) -> Geometry:
    """
    Chop a geometry along the antimeridian

    :param geom: Geometry to maybe partition
    :param precision: in degrees
    :returns: either the same geometry if it doesn't intersect the antimeridian,
              or multi-geometry that has been split.
    """
    if geom.crs is None:
        raise ValueError("Expect geometry with CRS defined")

    l180 = projected_lon(geom.crs, 180, step=precision)
    if geom.intersects(l180):
        return multigeom(geom.split(l180))

    return geom


###########################################
# Helper constructor functions a la shapely
###########################################


def point(x: float, y: float, crs: MaybeCRS) -> Geometry:
    """
    Create a 2D Point

    >>> point(10, 10, crs=None)
    Geometry(POINT (10 10), None)
    """
    return Geometry({"type": "Point", "coordinates": [float(x), float(y)]}, crs=crs)


def multipoint(coords: CoordList, crs: MaybeCRS) -> Geometry:
    """
    Create a 2D MultiPoint Geometry

    >>> multipoint([(10, 10), (20, 20)], None)
    Geometry(MULTIPOINT (10 10, 20 20), None)

    :param coords: list of x,y coordinate tuples
    """
    return Geometry({"type": "MultiPoint", "coordinates": coords}, crs=crs)


def line(coords: CoordList, crs: MaybeCRS) -> Geometry:
    """
    Create a 2D LineString (Connected set of lines)

    >>> line([(10, 10), (20, 20), (30, 40)], None)
    Geometry(LINESTRING (10 10, 20 20, 30 40), None)

    :param coords: list of x,y coordinate tuples
    """
    return Geometry({"type": "LineString", "coordinates": coords}, crs=crs)


def multiline(coords: List[CoordList], crs: MaybeCRS) -> Geometry:
    """
    Create a 2D MultiLineString (Multiple disconnected sets of lines)

    >>> multiline([[(10, 10), (20, 20), (30, 40)], [(50, 60), (70, 80), (90, 99)]], None)
    Geometry(MULTILINESTRING ((10 10, 20 20, 30 40), (50 60, 70 80, 90 99)), None)

    :param coords: list of lists of x,y coordinate tuples
    """
    return Geometry({"type": "MultiLineString", "coordinates": coords}, crs=crs)


def polygon(outer, crs: MaybeCRS, *inners) -> Geometry:
    """
    Create a 2D Polygon

    >>> polygon([(10, 10), (20, 20), (20, 10), (10, 10)], None)
    Geometry(POLYGON ((10 10, 20 20, 20 10, 10 10)), None)

    :param coords: list of 2d x,y coordinate tuples
    """
    return Geometry({"type": "Polygon", "coordinates": (outer,) + inners}, crs=crs)


def multipolygon(coords: List[CoordList], crs: MaybeCRS) -> Geometry:
    """
    Create a 2D MultiPolygon

    :param coords: list of lists of x,y coordinate tuples
    """
    return Geometry(
        {"type": "MultiPolygon", "coordinates": [[poly] for poly in coords]}, crs=crs
    )


def box(
    left: float, bottom: float, right: float, top: float, crs: MaybeCRS
) -> Geometry:
    """
    Create a 2D Box (Polygon)

    >>> box(10, 10, 20, 20, None)
    Geometry(POLYGON ((10 10, 10 20, 20 20, 20 10, 10 10)), None)
    """
    points = [
        (left, bottom),
        (left, top),
        (right, top),
        (right, bottom),
        (left, bottom),
    ]
    return polygon(points, crs=crs)


def polygon_from_transform(
    width: float, height: float, transform: Affine, crs: MaybeCRS
) -> Geometry:
    """
    Create a 2D Polygon from an affine transform

    :param width:
    :param height:
    :param transform:
    :param crs: CRS
    """
    points = [(0, 0), (0, height), (width, height), (width, 0), (0, 0)]
    transform.itransform(points)
    return polygon(points, crs=crs)


def sides(poly: Geometry) -> Iterable[Geometry]:
    """Returns a sequence of Geometry[Line] objects.

    One for each side of the exterior ring of the input polygon.
    """
    XY = poly.exterior.points
    crs = poly.crs
    for p1, p2 in zip(XY[:-1], XY[1:]):
        yield line([p1, p2], crs)


def multigeom(geoms: Iterable[Geometry]) -> Geometry:
    """Construct Multi{Polygon|LineString|Point}"""
    geoms = list(geoms)  # force into list
    src_type = {g.type for g in geoms}
    if len(src_type) > 1:
        raise ValueError("All Geometries must be of the same type")

    crs = common_crs(geoms)  # will raise if some differ
    raw_geoms = [g.geom for g in geoms]
    src_type = src_type.pop()
    if src_type == "Polygon":
        return Geometry(geometry.MultiPolygon(raw_geoms), crs)
    if src_type == "Point":
        return Geometry(geometry.MultiPoint(raw_geoms), crs)
    if src_type == "LineString":
        return Geometry(geometry.MultiLineString(raw_geoms), crs)

    raise ValueError("Only understand Polygon|LineString|Point")


###########################################
# Multi-geometry operations
###########################################


def unary_union(geoms: Iterable[Geometry]) -> Optional[Geometry]:
    """
    compute union of multiple (multi)polygons efficiently
    """
    geoms = list(geoms)
    if len(geoms) == 0:
        return None

    first = geoms[0]
    crs = first.crs
    for g in geoms[1:]:
        if crs != g.crs:
            raise CRSMismatchError((crs, g.crs))

    return Geometry(ops.unary_union([g.geom for g in geoms]), crs)


def unary_intersection(geoms: Iterable[Geometry]) -> Geometry:
    """
    compute intersection of multiple (multi)polygons
    """
    return functools.reduce(Geometry.intersection, geoms)


def intersects(a: Geometry, b: Geometry) -> bool:
    """Returns True if geometries intersect, else False"""
    return a.intersects(b) and not a.touches(b)


def bbox_union(bbs: Iterable[BoundingBox]) -> BoundingBox:
    """Given a stream of bounding boxes compute enclosing BoundingBox"""
    # pylint: disable=invalid-name

    L = B = float("+inf")
    R = T = float("-inf")

    for bb in bbs:
        l, b, r, t = bb
        L = min(l, L)
        B = min(b, B)
        R = max(r, R)
        T = max(t, T)

    return BoundingBox(L, B, R, T)


def bbox_intersection(bbs: Iterable[BoundingBox]) -> BoundingBox:
    """Given a stream of bounding boxes compute the overlap BoundingBox"""
    # pylint: disable=invalid-name

    L = B = float("-inf")
    R = T = float("+inf")

    for bb in bbs:
        l, b, r, t = bb
        L = max(l, L)
        B = max(b, B)
        R = min(r, R)
        T = min(t, T)

    return BoundingBox(L, B, R, T)


def lonlat_bounds(
    geom: Geometry, mode: str = "safe", resolution: Optional[float] = None
) -> BoundingBox:
    """
    Return the bounding box of a geometry

    :param geom: Geometry in any projection
    :param mode: safe|quick
    :param resolution: If supplied will first segmentize input geometry to have no segment longer than ``resolution``,
                       this increases accuracy at the cost of computation
    """
    assert mode in ("safe", "quick")
    if geom.crs is None:
        raise ValueError("lonlat_bounds can only operate on Geometry with CRS defined")

    if geom.crs.geographic:
        return geom.boundingbox

    if resolution is not None and math.isfinite(resolution):
        geom = geom.segmented(resolution)

    bbox = geom.to_crs("EPSG:4326", resolution=math.inf).boundingbox

    xx_range = bbox.range_x
    if mode == "safe":
        # If range in Longitude is more than 180 then it's probably wrapped
        # around 180 (X-360 for X > 180), so we add back 360 but only for X<0
        # values. This only works if input geometry doesn't span more than half
        # a globe, so we need to check for that too, but this is not yet
        # implemented...

        if bbox.span_x > 180:
            # TODO: check the case when input geometry spans >180 region.
            #       For now we assume "smaller" geometries not too close
            #       to poles.
            xx_ = [x + 360 if x < 0 else x for x in bbox.range_x]
            xx_range_ = min(xx_), max(xx_)
            span_x_ = xx_range_[1] - xx_range_[0]
            if span_x_ < bbox.span_x:
                xx_range = xx_range_

    return BoundingBox.from_xy(xx_range, bbox.range_y)


def mid_longitude(geom: Geometry) -> float:
    """
    Compute longitude of the center point of a geometry
    """
    ((lon,), _) = geom.centroid.to_crs("epsg:4326").xy
    return lon
