# This file is part of the Open Data Cube, see https://opendatacube.org for more information
#
# Copyright (c) 2015-2020 ODC Contributors
# SPDX-License-Identifier: Apache-2.0
import array
import functools
import itertools
import math
import warnings
from collections import abc
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy
from affine import Affine
from pyproj.aoi import AreaOfInterest
from shapely import geometry, ops
from shapely.geometry import base

from .crs import CRS, CRSMismatchError, MaybeCRS, SomeCRS, norm_crs, norm_crs_or_error
from .math import edge_index, quasi_random_r2
from .types import SomeShape, SupportsCoords, Unset, shape_

CoordList = List[Tuple[float, float]]

# pylint: disable=too-many-lines,too-many-public-methods


class BoundingBox(Sequence[float]):
    """Bounding box, defining extent in cartesian coordinates."""

    __slots__ = "_crs", "_box"

    def __init__(
        self, left: float, bottom: float, right: float, top: float, crs: MaybeCRS = None
    ):
        self._box = (left, bottom, right, top)
        self._crs = norm_crs(crs)

    @property
    def left(self):
        return self._box[0]

    @property
    def bottom(self):
        return self._box[1]

    @property
    def right(self):
        return self._box[2]

    @property
    def top(self):
        return self._box[3]

    @property
    def crs(self) -> Optional[CRS]:
        return self._crs

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        return self._box

    def __iter__(self):
        return self._box.__iter__()

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, BoundingBox):
            return self._crs == other._crs and self._box == other._box
        return self._box == other

    def __hash__(self) -> int:
        return hash((self._crs, self._box))

    def __len__(self) -> int:
        return 4

    def __getitem__(self, idx):
        return self._box[idx]

    def __repr__(self) -> str:
        if self.crs is None:
            return f"BoundingBox(left={self.left}, bottom={self.bottom}, right={self.right}, top={self.top})"
        return (
            f"BoundingBox(left={self.left}, bottom={self.bottom},"
            f" right={self.right}, top={self.top}, crs={repr(self.crs)})"
        )

    def __str__(self) -> str:
        return self.__repr__()

    def __and__(self, other: "BoundingBox") -> "BoundingBox":
        return bbox_intersection([self, other])

    def __or__(self, other: "BoundingBox") -> "BoundingBox":
        return bbox_union([self, other])

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
            crs=self._crs,
        )

    @property
    def span_x(self) -> float:
        """Span of the bounding box along x axis."""
        return self.right - self.left

    @property
    def span_y(self) -> float:
        """Span of the bounding box along y axis."""
        return self.top - self.bottom

    @property
    def aspect(self) -> float:
        """Aspect ratio."""
        return self.span_x / self.span_y

    @property
    def width(self) -> int:
        """``int(span_x)``"""
        return int(self.right - self.left)

    @property
    def height(self) -> int:
        """``int(span_y)``"""
        return int(self.top - self.bottom)

    @property
    def shape(self) -> Tuple[int, int]:
        """``(int(span_y), int(span_x))``."""
        return (self.height, self.width)

    @property
    def range_x(self) -> Tuple[float, float]:
        """``left, right``"""
        return (self.left, self.right)

    @property
    def range_y(self) -> Tuple[float, float]:
        """``bottom, top``"""
        return (self.bottom, self.top)

    @property
    def points(self) -> CoordList:
        """Extract four corners of the bounding box."""
        x0, y0, x1, y1 = self._box
        return list(itertools.product((x0, x1), (y0, y1)))

    def transform(self, transform: Affine) -> "BoundingBox":
        """
        Map bounding box through a linear transform.

        Apply linear transform on four points of the bounding box and compute bounding box of these
        four points.
        """
        pts = [transform * pt for pt in self.points]
        xx = [x for x, _ in pts]
        yy = [y for _, y in pts]
        return BoundingBox(min(xx), min(yy), max(xx), max(yy), self._crs)

    def to_crs(self, crs: SomeCRS, **kw) -> "BoundingBox":
        """
        Compute bounding box in other CRS.
        """
        return self.polygon.to_crs(crs, **kw).boundingbox

    def map_bounds(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Convert to bounds in folium/ipyleaflet style.

        Returns SW, and NE corners in lat/lon order.
        ``((lat_w, lon_s), (lat_e, lon_n))``.
        """
        x0, y0, x1, y1 = self._box
        if self._crs == "epsg:4326" or self._crs is None:
            return (y0, x0), (y1, x1)
        (x0, y0), _, (x1, y1) = self.polygon.exterior.to_crs("epsg:4326").points[:3]
        return (y0, x0), (y1, x1)

    @property
    def polygon(self) -> "Geometry":
        return box(*self.bbox, self.crs)

    @staticmethod
    def from_xy(
        x: Tuple[float, float], y: Tuple[float, float], crs: MaybeCRS = None
    ) -> "BoundingBox":
        """
        Construct :py:class:`~odc.geo.geom.BoundingBox` from x and y ranges.

        :param x: (left, right)
        :param y: (bottom, top)
        :param crs: CRS
        """
        x1, x2 = sorted(x)
        y1, y2 = sorted(y)
        return BoundingBox(x1, y1, x2, y2, crs)

    @staticmethod
    def from_points(
        p1: Tuple[float, float], p2: Tuple[float, float], crs: MaybeCRS = None
    ) -> "BoundingBox":
        """
        Construct :py:class:`~odc.geo.geom.BoundingBox` from two points.

        :param p1: (x, y)
        :param p2: (x, y)
        :param crs: CRS
        """
        return BoundingBox.from_xy((p1[0], p2[0]), (p1[1], p2[1]), crs)

    @staticmethod
    def from_transform(
        shape: SomeShape, transform: Affine, crs: MaybeCRS = None
    ) -> "BoundingBox":
        """
        Construct :py:class:`~odc.geo.geom.BoundingBox` from image shape and transform.

        :param shape: image dimensions
        :param transform: Affine mapping from pixel to world
        :param crs: CRS
        """
        p1 = transform * (0, 0)
        p2 = transform * shape_(shape).xy
        return BoundingBox.from_points(p1, p2, crs=crs)

    @property
    def aoi(self) -> AreaOfInterest:
        """
        Interop with pyproj.
        """
        if self._crs is None or self._crs == "epsg:4326":
            return AreaOfInterest(*self._box)
        return AreaOfInterest(*self.to_crs("epsg:4326").bbox)

    def round(self) -> "BoundingBox":
        """
        Expand bounding box to nearest integer on all sides.
        """
        x0, y0, x1, y1 = self._box
        return BoundingBox(
            math.floor(x0), math.floor(y0), math.ceil(x1), math.ceil(y1), crs=self._crs
        )

    def boundary(self, pts_per_side: int = 2) -> "Geometry":
        """
        Points along the perimeter as a linear ring.
        """
        x0, y0, x1, y1 = self._box
        xx = numpy.linspace(x0, x1, pts_per_side, dtype="float32")
        yy = numpy.linspace(y0, y1, pts_per_side, dtype="float32")
        return line(
            [
                (float(xx[ix]), float(yy[iy]))
                for iy, ix in edge_index((pts_per_side, pts_per_side), closed=True)
            ],
            self.crs,
        )

    def qr2sample(
        self,
        n: int,
        padding: Optional[float] = None,
        with_edges: bool = False,
        offset: int = 0,
    ) -> "Geometry":
        """
        Generate quasi-random sample of points within this boundingbox.

        :param n:
           Number of points

        :param padding:
           Minimal distance to the edge

        :param offset:
           Offset into quasi-random sequence from where to start

        :param edges:
           Also include samples along the edge and corners

        References: http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/
        """
        # pylint: disable=too-many-locals
        x0, y0, x1, y1 = self._box
        nx = x1 - x0
        ny = y1 - y0
        pts = quasi_random_r2(n, offset=offset)
        s = numpy.asarray([nx, ny], dtype="float32")
        edge_pts = []

        if with_edges:
            sample_density = numpy.sqrt(n / (nx * ny))
            n_side = int(numpy.round(sample_density * min(nx, ny))) + 1
            n_side = max(2, n_side)
            edge_pts = self.boundary(n_side).coords[:-1]
            if padding is None:
                padding = 0.3 * min(nx, ny) / (n_side - 1)

        if padding is None:
            pts = pts * s
        else:
            pts = pts * (s - 2 * padding) + padding

        pts[:, 0] += x0
        pts[:, 1] += y0

        return multipoint(pts.tolist() + edge_pts, self.crs)


def wrap_shapely(method):
    """
    Takes a method that expects shapely geometry arguments and converts it to a method that operates
    on :py:class:`odc.geo.geom.Geometry` objects that carry their CRSs.
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

        if isinstance(x, abc.Sequence):
            if all(is_scalar(y) for y in x):
                return x[:2]
            return [go(y) for y in x]

        raise ValueError(f"invalid coordinate {x}")

    return {"type": geojson["type"], "coordinates": go(geojson["coordinates"])}


def _geojson_to_shapely(xx: Any) -> base.BaseGeometry:
    _type = xx.get("type", None)

    def to_geom(x) -> base.BaseGeometry:
        return geometry.shape(force_2d(x))

    if _type is None:
        raise ValueError("Not a valid GeoJSON")

    _type = _type.lower()
    if _type == "featurecollection":
        features = xx.get("features", [])
        if len(features) == 1:
            return to_geom(features[0]["geometry"])

        return _multigeom([to_geom(feature["geometry"]) for feature in features])

    if _type == "feature":
        return to_geom(xx["geometry"])

    return to_geom(xx)


def densify(coords: CoordList, resolution: float) -> CoordList:
    """
    Adds points so they are at most `resolution` units apart.
    """
    d2 = resolution**2

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
    return geometry.shape(geom)


class Geometry(SupportsCoords[float]):
    """
    2D Geometry with CRS.

    This is a wrapper around :py:class:`shapely.geometry.BaseGeometry` that adds projection information.

    Instantiate with a GeoJSON structure. If 3D coordinates are supplied, they are converted to 2D
    by dropping the Z points.
    """

    # pylint: disable=protected-access, multiple-statements

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

        if crs is None:
            if isinstance(geom, dict) and geom.get("type", "").lower().startswith(
                "feature"
            ):
                # Assume 4326 for geojson inputs
                crs = "epsg:4326"

        crs = norm_crs(crs)
        self.crs = crs
        if isinstance(geom, base.BaseGeometry):
            self.geom = geom
        elif isinstance(geom, dict):
            self.geom = _geojson_to_shapely(geom)
        else:
            raise ValueError(f"Unexpected type {type(geom)}")

    def clone(self) -> "Geometry":
        return Geometry(self)

    # fmt: off
    @wrap_shapely
    def contains(self, other: "Geometry") -> bool: return self.contains(other)
    @wrap_shapely
    def covers(self, other: "Geometry") -> bool: return self.covers(other)
    @wrap_shapely
    def crosses(self, other: "Geometry") -> bool: return self.crosses(other)
    @wrap_shapely
    def disjoint(self, other: "Geometry") -> bool: return self.disjoint(other)
    @wrap_shapely
    def intersects(self, other: "Geometry") -> bool: return self.intersects(other)
    @wrap_shapely
    def touches(self, other: "Geometry") -> bool: return self.touches(other)
    @wrap_shapely
    def within(self, other: "Geometry") -> bool: return self.within(other)
    @wrap_shapely
    def overlaps(self, other: "Geometry") -> bool: return self.overlaps(other)
    @wrap_shapely
    def difference(self, other: "Geometry") -> "Geometry": return self.difference(other)
    @wrap_shapely
    def intersection(self, other: "Geometry") -> "Geometry": return self.intersection(other)
    @wrap_shapely
    def symmetric_difference(self, other: "Geometry") -> "Geometry": return self.symmetric_difference(other)
    @wrap_shapely
    def union(self, other: "Geometry") -> "Geometry": return self.union(other)
    @wrap_shapely
    def __and__(self, other: "Geometry") -> "Geometry": return self.__and__(other)
    @wrap_shapely
    def __or__(self, other: "Geometry") -> "Geometry": return self.__or__(other)
    @wrap_shapely
    def __xor__(self, other: "Geometry") -> "Geometry": return self.__xor__(other)
    @wrap_shapely
    def __sub__(self, other: "Geometry") -> "Geometry": return self.__sub__(other)

    def _repr_svg_(self) -> str: return self.geom._repr_svg_()

    @property
    def geom_type(self) -> str: return self.geom.geom_type
    @property
    def type(self) -> str: return self.geom.type
    @property
    def is_empty(self) -> bool: return self.geom.is_empty
    @property
    def is_valid(self) -> bool: return self.geom.is_valid
    @property
    def is_ring(self) -> bool: return self.geom.is_ring
    @property
    def boundary(self) -> "Geometry": return Geometry(self.geom.boundary, self.crs)
    @property
    def exterior(self) -> "Geometry": return Geometry(self.geom.exterior, self.crs)
    @property
    def interiors(self) -> List["Geometry"]: return [Geometry(g, self.crs) for g in self.geom.interiors]
    @property
    def centroid(self) -> "Geometry": return Geometry(self.geom.centroid, self.crs)
    @property
    def coords(self) -> CoordList: return list(self.geom.coords)
    @property
    def points(self) -> CoordList: return self.coords
    @property
    def length(self) -> float: return self.geom.length
    @property
    def area(self) -> float: return self.geom.area
    @property
    def xy(self) -> Tuple[array.array, array.array]: return self.geom.xy
    @property
    def convex_hull(self) -> "Geometry": return Geometry(self.geom.convex_hull, self.crs)
    @property
    def envelope(self) -> "Geometry": return Geometry(self.geom.envelope, self.crs)
    @property
    def boundingbox(self) -> BoundingBox:
        minx, miny, maxx, maxy = self.geom.bounds
        return BoundingBox(left=minx, right=maxx, bottom=miny, top=maxy, crs=self.crs)
    @property
    def wkt(self) -> str: return self.geom.wkt
    @property
    def __geo_interface__(self): return self.geom.__geo_interface__
    # fmt: on

    @property
    def json(self):
        return self.__geo_interface__

    def segmented(self, resolution: float) -> "Geometry":
        """
        Increase resolution of the geometry.

        Possibly add more points to the geometry so that no edge is longer than ``resolution``.
        """

        def segmentize_shapely(geom: base.BaseGeometry) -> base.BaseGeometry:
            if geom.geom_type in ["Point", "MultiPoint"]:
                return _clone_shapely_geom(geom)  # clone without changes

            if geom.geom_type in [
                "GeometryCollection",
                "MultiPolygon",
                "MultiLineString",
            ]:
                return type(geom)([segmentize_shapely(g) for g in geom.geoms])

            if geom.geom_type in ["LineString", "LinearRing"]:
                return type(geom)(densify(list(geom.coords), resolution))

            if geom.geom_type == "Polygon":
                return geometry.Polygon(
                    densify(list(geom.exterior.coords), resolution),
                    [densify(list(i.coords), resolution) for i in geom.interiors],
                )

            raise ValueError(
                f"unknown geometry type {geom.geom_type}"
            )  # pragma: no cover

        return Geometry(segmentize_shapely(self.geom), self.crs)

    def interpolate(self, distance: float) -> "Geometry":
        """
        Returns a point ``distance`` units along the line.

        @raises :py:class:`TypeError` if geometry doesn't support this operation.
        """
        return Geometry(self.geom.interpolate(distance), self.crs)

    def buffer(self, distance: float, resolution: float = 30) -> "Geometry":
        return Geometry(self.geom.buffer(distance, resolution=resolution), self.crs)

    def simplify(self, tolerance: float, preserve_topology: bool = True) -> "Geometry":
        return Geometry(
            self.geom.simplify(tolerance, preserve_topology=preserve_topology), self.crs
        )

    def transform(self, func, *, crs: MaybeCRS = Unset()) -> "Geometry":
        """
        Map through arbitrary transform.

        Applies ``func`` to all coordinates of :py:class:`~odc.geo.geom.Geometry` and returns a new
        Geometry of the same type and in the same projection from the transformed coordinates.

        ``func`` maps ``x, y``, and optionally ``z`` to output ``xp, yp, zp``. The input parameters
        may be iterable types like lists or arrays or single values. The output shall be of the same
        type: scalars in, scalars out; lists in, lists out.

        :param crs: If supplied override output CRS with that value, ``crs=None`` will remove CRS
        for the output, default is to keep the original projection.
        """
        if isinstance(func, Affine):

            def _tr(A, x, y):
                _xy = [A * xy for xy in zip(x, y)]
                return [x for x, _ in _xy], [y for _, y in _xy]

            _geom = ops.transform(functools.partial(_tr, func), self.geom)
        else:
            _geom = ops.transform(func, self.geom)

        if isinstance(crs, Unset):
            crs = self.crs

        return Geometry(_geom, crs)

    def _to_crs(self, crs: CRS) -> "Geometry":
        assert self.crs is not None
        return Geometry(ops.transform(self.crs.transformer_to_crs(crs), self.geom), crs)

    def to_crs(
        self,
        crs: SomeCRS,
        resolution: Union[float, Literal["auto"], None] = None,
        wrapdateline: bool = False,
        *,
        check_and_fix: bool = False,
    ) -> "Geometry":
        """
        Convert geometry to a different Coordinate Reference System.

        :param crs:
          CRS to convert to

        :param resolution:
          When supplied, increase resolution of the geometry before projecting.
          This is done by adding extra points such that no segment is longer
          than ``resolution`` units in the source CRS of the geometry.
          Reasonable value for the parameter depends on the application. Lower
          value will result in higher precision, but larger output geometry and
          slower processing.

          Consider using :py:meth:`~odc.geo.geom.Geometry.simplify` on the
          output of this function to reduce the size of the ouptut, especially
          when using high precision transform.

          If not supplied project the geometry as is without adding any extra points.

        :param wrapdateline:
           Attempt to gracefully handle geometry that intersects the dateline when converting to
           geographic projections. Currently only works in few specific cases (source CRS is smooth
           over the dateline).

        :param check_and_fix:
           When ``True`` remove vertices that didn't project cleanly and apply ``.buffer(0)`` is geometry
           is not valid after projection.
        """
        crs = norm_crs_or_error(crs, self)
        if self.crs == crs:
            return self

        if self.crs is None:
            raise ValueError("Cannot project geometries without CRS")

        if resolution == "auto":
            resolution = _auto_resolution(self)

        if resolution is not None and math.isfinite(resolution):
            geom = self.segmented(resolution)
        else:
            geom = self

        def maybe_fix(g: Geometry) -> Geometry:
            if not check_and_fix or g.is_valid:
                return g
            g = g.dropna()
            if g.geom_type in ["Polygon", "MultiPolygon"] and not g.is_valid:
                g = g.buffer(0)
            return g

        eps = 1e-4
        if wrapdateline and crs.geographic:
            # TODO: derive precision from resolution by converting to degrees
            precision = 0.1
            chopped = chop_along_antimeridian(geom, precision)
            chopped_lonlat = maybe_fix(chopped._to_crs(crs))
            return clip_lon180(chopped_lonlat, eps)

        return maybe_fix(geom._to_crs(crs))

    def assign_crs(self, crs: MaybeCRS) -> "Geometry":
        """
        Same geometry but with crs changed.
        """
        return Geometry(self.geom, crs=crs)

    def geojson(
        self,
        properties: Optional[Dict[str, Any]] = None,
        simplify: float = 0.05,
        resolution: Optional[float] = None,
        wrapdateline: bool = False,
        **props,
    ) -> Dict[str, Any]:
        """
        Render geometry to GeoJSON.

        Convert geometry to ``ESPG:4326`` and wrap it in GeoJSON Feature with supplied properties.

        :param properties:
            Properties to include in the GeoJSON output.

        :param simplify:
            Tolerance in degrees for simplifying geometry after changing to lon/lat. Larger number
            will result in a smaller (fewer points) and hence faster to display, but less precise
            geometry. Default is ``0.05`` of a degree. To disable set to ``0``.

        :param resolution:
           When supplied, extra points will be added to the original geometry such that no segment
           is longer than ``resolution`` units. Passed on to
           :py:meth:`~odc.geo.geom.Geometry.to_crs`.

        :param wrapdateline: Passed on to :py:meth:`~odc.geo.geom.Geometry.to_crs`

        :return: GeoJSON Feature dictionary
        """
        if self.geom_type == "GeometryCollection":
            return {
                "type": "FeatureCollection",
                "features": [
                    g.geojson(
                        properties=properties,
                        simplify=simplify,
                        resolution=resolution,
                        wrapdateline=wrapdateline,
                        **props,
                    )
                    for g in self.geoms
                ],
            }

        if self.crs is not None:
            gg = self.to_crs(
                "epsg:4326", resolution=resolution, wrapdateline=wrapdateline
            )
        else:
            gg = self

        if simplify > 0:
            gg = gg.simplify(simplify)
        if properties is None:
            properties = {**props}
        return {"type": "Feature", "geometry": gg.json, "properties": properties}

    def split(self, splitter: "Geometry") -> Iterable["Geometry"]:
        """shapely.ops.split"""
        if splitter.crs != self.crs:
            raise CRSMismatchError(self.crs, splitter.crs)

        for g in ops.split(self.geom, splitter.geom).geoms:
            yield Geometry(g, self.crs)

    @property
    def geoms(self) -> Iterator["Geometry"]:
        sub_geoms = getattr(self.geom, "geoms", [])
        for geom in sub_geoms:
            yield Geometry(geom, self.crs)

    def __iter__(self) -> Iterator["Geometry"]:
        warnings.warn(
            "Iteration interface is deprecated, please use `.geoms`",
            category=DeprecationWarning,
        )
        yield from self.geoms

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

    @property
    def is_multi(self) -> bool:
        """True for multi-geometry types."""
        t = self.geom_type
        return t.startswith("Multi") or t == "GeometryCollection"

    def __rmul__(self, A: Affine) -> "Geometry":
        """
        Map geometry points through linear transform ``A*g``.
        """
        return self.transform(A)

    def svg(self, *args, **kw) -> str:
        """
        Returns SVG path element (wraps shapely).

        This method will drop parameters not supported by the underlying shapely geometry.

        :param scale_factor: Multiplication factor for the SVG stroke-width.  Default is 1
        :param stroke_color: Color for lines and rings
        :param fill_color: Color for poinits and polygons
        :param opacity: Float number between 0 and 1 for color opacity. Default value is 0.6
        """
        from inspect import signature  # pylint: disable=import-outside-toplevel

        valid_params = signature(self.geom.svg).parameters
        opts = {k: v for k, v in kw.items() if k in valid_params}
        return self.geom.svg(*args, **opts)

    def svg_path(self, ndecimal: int = 6) -> str:
        """
        Produce SVG path text.

        This should work for polygons/rings/lines, doesn't really
        makes sense for points.

        Example usage: ``f'<path d="{g.svg_path(0)}" ... />'``
        """

        def render_ring(g, close):
            return (
                "M"
                + "L".join(f"{x:.{ndecimal}f},{y:.{ndecimal}f}" for x, y in g.coords)
                + ("z" if close else "")
            )

        if self.is_multi:
            return "".join([g.svg_path(ndecimal) for g in self.geoms])

        if self.geom_type == "Polygon":
            return "".join(
                [render_ring(g, close=True) for g in [self.exterior, *self.interiors]]
            )

        return render_ring(self, close=self.is_ring)

    def filter(self, pred: Callable[[float, float], bool]) -> "Geometry":
        """
        Keep only those points for which `pred(x,y) is True`.
        """

        if self.geom_type == "Polygon":
            pts = [(x, y) for x, y in self.exterior.points if pred(x, y)]
            # prune inner polygon points
            inners = [
                [(x, y) for x, y in ring.points if pred(x, y)]
                for ring in self.interiors
            ]
            # prune inner polygons that don't have enough points
            inners = [ring for ring in inners if len(ring) >= 3]
            return polygon(pts, self.crs, *inners)

        if self.geom_type in ("LinearRing", "LineString"):
            pts = [(x, y) for x, y in self.points if pred(x, y)]
            if len(pts) == 1:
                pts = []  # need at least 2 points for this type
            _geom = type(self.geom)(pts)
            return Geometry(_geom, self.crs)

        if self.geom_type == "Point":
            x, y = self.coords[0]
            if pred(x, y):
                return self
            return Geometry(geometry.Point(), self.crs)

        if self.geom_type == "MultiPoint":
            pts = [(x, y) for x, y in [pt.points[0] for pt in self.geoms] if pred(x, y)]
            return multipoint(pts, self.crs)

        if self.is_multi:
            _filtered = [g.filter(pred).geom for g in self.geoms]
            _filtered = [g for g in _filtered if not g.is_empty]
            _geom = type(self.geom)(_filtered)
            return Geometry(_geom, self.crs)

        raise AssertionError("Unhandled geometry type detected")  # pragma: no cover

    def dropna(self) -> "Geometry":
        """
        Only keep finite points.
        """
        return self.filter(lambda x, y: math.isfinite(x) and math.isfinite(y))


def common_crs(geoms: Iterable[Geometry]) -> Optional[CRS]:
    """
    Compute common CRS.

    :return: CRS common across geometries.
    :return: ``None`` when input was empty

    :raises: :py:class:`odc.geo.crs.CRSMismatchError` when there are multiple.
    """
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
    """
    Project vertical line along some longitude into a given CRS.

    :param crs: Destination CRS
    :param lon: Longitude to project
    :param lat: Optionally limit range of the line
    :param step: Line "resolution" in degrees
    """
    crs = norm_crs_or_error(crs, lon)
    yy = numpy.arange(lat[0], lat[1], step, dtype="float32")
    xx = numpy.full_like(yy, lon)
    tr = CRS("EPSG:4326").transformer_to_crs(crs)
    xx_, yy_ = tr(xx, yy)
    pts = [
        (float(x), float(y))
        for x, y in zip(xx_, yy_)
        if math.isfinite(x) and math.isfinite(y)
    ]
    if len(pts) < 2:
        return line([], crs)
    return line(pts, crs)


def clip_lon180(geom: Geometry, tol=1e-6) -> Geometry:
    """
    Tweak Geometry in the vicinity of longitude discontinuity.

    For every point within ``tol`` degress of ``lon=180|-180``, clip to either ``lon=180`` or
    ``lon=-180``. Which one is decided based on where the majority of other points lie.

    .. note::

      This will only do the "right thing" for "chopped" geometries. Expectation is that all the
      points are to one side of ``lon=180`` line, but some might have moved just beyond due to
      numeric tolerance issues.

    .. seealso:: :py:meth:`odc.geo.geom.chop_along_antimeridian`
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

    if geom.geom_type.startswith("Multi"):
        return multigeom(g.transform(transformer) for g in geom.geoms)

    return geom.transform(transformer)


def chop_along_antimeridian(geom: Geometry, precision: float = 0.1) -> Geometry:
    """
    Chop a geometry along the antimeridian.

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
    Create a 2D Point.

    >>> point(10, 10, crs=None)
    Geometry(POINT (10 10), None)
    """
    return Geometry({"type": "Point", "coordinates": [float(x), float(y)]}, crs=crs)


def multipoint(coords: CoordList, crs: MaybeCRS) -> Geometry:
    """
    Create a 2D MultiPoint Geometry.

    >>> multipoint([(10, 10), (20, 20)], None)
    Geometry(MULTIPOINT (10 10, 20 20), None)

    :param coords: list of x,y coordinate tuples
    """
    return Geometry({"type": "MultiPoint", "coordinates": coords}, crs=crs)


def line(coords: CoordList, crs: MaybeCRS) -> Geometry:
    """
    Create a 2D LineString (Connected set of lines).

    >>> line([(10, 10), (20, 20), (30, 40)], None)
    Geometry(LINESTRING (10 10, 20 20, 30 40), None)

    :param coords: list of x,y coordinate tuples
    """
    return Geometry({"type": "LineString", "coordinates": coords}, crs=crs)


def multiline(coords: List[CoordList], crs: MaybeCRS) -> Geometry:
    """
    Create a 2D MultiLineString (Multiple disconnected sets of lines).

    >>> multiline([[(10, 10), (20, 20), (30, 40)], [(50, 60), (70, 80), (90, 99)]], None)
    Geometry(MULTILINESTRING ((10 10, 20 20, 30 40), (50 60, 70 80, 90 99)), None)

    :param coords: list of lists of x,y coordinate tuples
    """
    return Geometry({"type": "MultiLineString", "coordinates": coords}, crs=crs)


def polygon(outer, crs: MaybeCRS, *inners) -> Geometry:
    """
    Create a 2D Polygon.

    >>> polygon([(10, 10), (20, 20), (20, 10), (10, 10)], None)
    Geometry(POLYGON ((10 10, 20 20, 20 10, 10 10)), None)

    :param coords: list of 2d x,y coordinate tuples
    """
    return Geometry({"type": "Polygon", "coordinates": (outer,) + inners}, crs=crs)


def multipolygon(coords: List[List[CoordList]], crs: MaybeCRS) -> Geometry:
    """
    Create a 2D MultiPolygon.

    :param coords:
       List of polygons. Where each "polygon" is itself a list of "rings". Each "ring" is a list of
       ``(x, y)`` tuples. First ring is an outer layer of the polygon, the rest, if present, define
       holes in the outer layer.
    """
    return Geometry({"type": "MultiPolygon", "coordinates": coords}, crs=crs)


def box(
    left: float, bottom: float, right: float, top: float, crs: MaybeCRS
) -> Geometry:
    """
    Create a 2D Box (Polygon).

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
    shape: SomeShape, transform: Affine, crs: MaybeCRS
) -> Geometry:
    """
    Create a 2D Polygon from an affine transform and shape.

    Useful for computing footprints of a geo-registered raster images.

    :param shape: Shape of the raster in pixels
    :param transform: Affine transfrom from pixel to CRS units
    :param crs: CRS
    """
    x1, y1 = shape_(shape).xy
    points = [(0, 0), (0, y1), (x1, y1), (x1, 0), (0, 0)]
    transform.itransform(points)
    return polygon(points, crs=crs)


def sides(poly: Geometry) -> Iterable[Geometry]:
    """
    Returns a sequence of Geometry[Line] objects.

    One for each side of the exterior ring of the input polygon.
    """
    XY = poly.exterior.points
    crs = poly.crs
    for p1, p2 in zip(XY[:-1], XY[1:]):
        yield line([p1, p2], crs)


def _multigeom(geoms: List[base.BaseGeometry]) -> base.BaseGeometry:
    src_type = {g.geom_type for g in geoms}
    if len(src_type) > 1:
        return geometry.GeometryCollection(geoms)

    geoms = [g for g in geoms if not g.is_empty]
    src_type = src_type.pop()
    if src_type == "Polygon":
        return geometry.MultiPolygon(geoms)
    if src_type == "Point":
        return geometry.MultiPoint(geoms)
    if src_type == "LineString":
        return geometry.MultiLineString(geoms)
    return geometry.GeometryCollection(geoms)


def multigeom(geoms: Iterable[Geometry]) -> Geometry:
    """Construct ``Multi{Polygon|LineString|Point}``."""
    geoms = list(geoms)  # force into list
    crs = common_crs(geoms)  # will raise if some differ
    return Geometry(_multigeom([g.geom for g in geoms]), crs)


###########################################
# Multi-geometry operations
###########################################


def unary_union(geoms: Iterable[Geometry]) -> Optional[Geometry]:
    """
    Compute union of multiple (multi)polygons efficiently.
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
    Compute intersection of multiple (multi)polygons.
    """
    return functools.reduce(Geometry.intersection, geoms)


def triangulate(pts: Geometry, **kw) -> Geometry:
    """
     Creates the Delaunay triangulation and returns a geometry collection.

     The source may be any geometry type. All vertices of the geometry will be
     used as the points of the triangulation.

    :param tolerance:
        From the GEOS documentation:
        tolerance is the snapping tolerance used to improve the robustness of
        the triangulation computation. A tolerance of 0.0 specifies that no
        snapping will take place.

     :param edges:
        If edges is False, a geometry collection of Polygons (triangles) will be returned (default).
        Otherwise LineString edges are returned.
    """
    geom = geometry.GeometryCollection(ops.triangulate(pts.geom, **kw))
    return Geometry(geom, pts.crs)


def intersects(a: Geometry, b: Geometry) -> bool:
    """
    Check for intersection.

    :return: ``True`` if geometries intersect, else ``False``
    """
    return a.intersects(b) and not a.touches(b)


def bbox_union(bbs: Iterable[BoundingBox]) -> BoundingBox:
    """
    Compute union of bounding boxes.

    Given a stream of bounding boxes compute enclosing :py:class:`~odc.geo.geom.BoundingBox`.
    """
    # pylint: disable=invalid-name
    try:
        bb, *bbs = bbs
    except ValueError:
        raise ValueError("Union of empty stream is undefined") from None

    L, B, R, T = bb
    crs = bb.crs

    for bb in bbs:
        l, b, r, t = bb
        L = min(l, L)
        B = min(b, B)
        R = max(r, R)
        T = max(t, T)

        if crs != bb.crs:
            raise CRSMismatchError((crs, bb.crs))

    return BoundingBox(L, B, R, T, crs)


def bbox_intersection(bbs: Iterable[BoundingBox]) -> BoundingBox:
    """
    Compute intersection of boudning boxes.

    Given a stream of bounding boxes compute the overlap :py:class:`~odc.geo.geom.BoundingBox`.
    """
    # pylint: disable=invalid-name
    try:
        bb, *bbs = bbs
    except ValueError:
        raise ValueError("Intersection of empty stream is undefined") from None

    L, B, R, T = bb
    crs = bb.crs

    for bb in bbs:
        l, b, r, t = bb
        L = max(l, L)
        B = max(b, B)
        R = min(r, R)
        T = min(t, T)

        if crs != bb.crs:
            raise CRSMismatchError((crs, bb.crs))

    return BoundingBox(L, B, R, T, crs)


def lonlat_bounds(
    geom: Geometry,
    mode: str = "safe",
    resolution: Union[float, None, Literal["auto"]] = None,
) -> BoundingBox:
    """
    Return the bounding box of a geometry in lon/lat.

    :param geom:
       Geometry in any projection
    :param mode:
       safe|quick
    :param resolution:
       If supplied will first segmentize input geometry to have no segment longer than
       ``resolution``, this increases accuracy at the cost of computation
    """
    assert mode in ("safe", "quick")
    if geom.crs is None:
        raise ValueError("lonlat_bounds can only operate on Geometry with CRS defined")

    if geom.crs.geographic:
        return geom.boundingbox

    if resolution == "auto":
        resolution = _auto_resolution(geom)

    if resolution is not None and math.isfinite(resolution):
        geom = geom.segmented(resolution)

    bbox = geom.to_crs("EPSG:4326", check_and_fix=True).boundingbox

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

    return BoundingBox.from_xy(xx_range, bbox.range_y, crs=4326)


def mid_longitude(geom: Geometry) -> float:
    """
    Compute longitude of the center point of a geometry.
    """
    ((lon,), _) = geom.centroid.to_crs("epsg:4326").xy
    return lon


def _auto_resolution(g: Geometry) -> float:
    # aim for ~100 points per side of a square
    return math.sqrt(g.area) * 4 / 100
