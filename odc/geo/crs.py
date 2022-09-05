# This file is part of the Open Data Cube, see https://opendatacube.org for more information
#
# Copyright (c) 2015-2020 ODC Contributors
# SPDX-License-Identifier: Apache-2.0
import warnings
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    overload,
)

import cachetools
import numpy
from pyproj.crs import CRS as _CRS
from pyproj.enums import WktVersion
from pyproj.exceptions import CRSError
from pyproj.transformer import Transformer

from .types import XY

SomeCRS = Union[str, int, "CRS", _CRS, Dict[str, Any]]
MaybeCRS = Optional[SomeCRS]

_crs_cache: Dict[str, Tuple[_CRS, str, Optional[int]]] = {}


def _make_crs_key(crs_spec: Union[int, str, _CRS]) -> str:
    if isinstance(crs_spec, str):
        normed_epsg = crs_spec.upper()
        if normed_epsg.startswith("EPSG:"):
            return normed_epsg
        return crs_spec
    if isinstance(crs_spec, int):
        return f"EPSG:{crs_spec}"
    return crs_spec.to_wkt()


@cachetools.cached(_crs_cache, key=_make_crs_key)
def _make_crs(crs: Union[str, int, _CRS]) -> Tuple[_CRS, str, Optional[int]]:
    if isinstance(crs, str):
        crs = _CRS.from_user_input(crs)
    if isinstance(crs, int):
        crs = _CRS.from_epsg(crs)
    epsg = crs.to_epsg()
    if epsg is not None:
        crs_str = f"EPSG:{epsg}"
    else:
        crs_str = crs.to_wkt()
    return (crs, crs_str, epsg)


def _make_crs_transform_key(from_crs, to_crs, always_xy):
    return (id(from_crs), id(to_crs), always_xy)


@cachetools.cached({}, key=_make_crs_transform_key)
def _make_crs_transform(from_crs: _CRS, to_crs: _CRS, always_xy: bool) -> Transformer:
    return Transformer.from_crs(from_crs, to_crs, always_xy=always_xy)


class CRS:
    """
    Wrapper around :py:class:`pyproj.crs.CRS` for backwards compatibility.
    """

    DEFAULT_WKT_VERSION = WktVersion.WKT2_2019
    """Default version for WKT: WKT2_2019"""

    __slots__ = ("_crs", "_epsg", "_str")

    def __init__(self, crs_spec: Any):
        """
        Construct CRS object from *something*.

        :param crs_spec:
           String representation of a CRS, often an EPSG code like ``'EPSG:4326'``. Can also be any
           object that implements ``.to_epsg()`` or ``.to_wkt()``.

        :raises: :py:class:`pyproj.exceptions.CRSError`
        """

        if isinstance(crs_spec, (str, int, _CRS)):
            self._crs, self._str, self._epsg = _make_crs(crs_spec)
        elif isinstance(crs_spec, CRS):
            self._crs = crs_spec._crs
            self._epsg = crs_spec._epsg
            self._str = crs_spec._str
        elif isinstance(crs_spec, dict):
            self._crs, self._str, self._epsg = _make_crs(_CRS.from_dict(crs_spec))
        else:
            try:
                epsg = crs_spec.to_epsg()
            except AttributeError:
                epsg = None
            if epsg is not None:
                self._crs, self._str, self._epsg = _make_crs(f"EPSG:{epsg}")
                return
            try:
                wkt = crs_spec.to_wkt()
            except AttributeError:
                wkt = None
            if wkt is not None:
                self._crs, self._str, self._epsg = _make_crs(wkt)
                return

            raise CRSError(
                "Expect string or any object with `.to_epsg()` or `.to_wkt()` methods"
            )

    def __getstate__(self):
        return {"crs_str": self._str}

    def __setstate__(self, state):
        self.__init__(state["crs_str"])

    def to_wkt(self, pretty: bool = False, version: Optional[WktVersion] = None) -> str:
        """
        Generate WKT representation of this CRS.

        :param pretty: If ``True`` generate multi-line WKT.
        :param version: Specify WKT version.
        """
        if version is None:
            version = self.DEFAULT_WKT_VERSION

        return self._crs.to_wkt(pretty=pretty, version=version)

    @property
    def wkt(self) -> str:
        """WKT representation of this CRS."""
        return self.to_wkt()

    def to_epsg(self) -> Optional[int]:
        """
        EPSG Code of the CRS or ``None``.
        """
        return self._epsg

    @property
    def epsg(self) -> Optional[int]:
        """
        EPSG Code of the CRS or ``None``.
        """
        return self._epsg

    @property
    def semi_major_axis(self):
        """Semi-major axis of the ellipsoid."""
        return self._crs.ellipsoid.semi_major_metre

    @property
    def semi_minor_axis(self):
        """Semi-minor axis of the ellipsoid."""
        return self._crs.ellipsoid.semi_minor_metre

    @property
    def inverse_flattening(self):
        """Inverse flattening of the ellipsoid."""
        return self._crs.ellipsoid.inverse_flattening

    @property
    def geographic(self) -> bool:
        """True if CRS is geographic."""
        return self._crs.is_geographic

    @property
    def projected(self) -> bool:
        """True if CRS is projected."""
        return self._crs.is_projected

    @property
    def dimensions(self) -> Tuple[str, str]:
        """
        List of dimension names of the CRS.

        The ordering of the names is intended to reflect the :py:class:`numpy.ndarray` axis order of the
        loaded raster.
        """
        if self.geographic:
            return "latitude", "longitude"

        if self.projected:
            return "y", "x"

        raise ValueError("Neither projected nor geographic")  # pragma: no cover

    @property
    def units(self) -> Tuple[str, str]:
        """
        List of dimension units of the CRS.

        The ordering of the units is intended to reflect the :py:class:`numpy.ndarray` axis order of the
        loaded raster.
        """
        if self.geographic:
            return "degrees_north", "degrees_east"

        if self.projected:
            x, y = self._crs.axis_info
            return x.unit_name, y.unit_name

        raise ValueError("Neither projected nor geographic")  # pragma: no cover

    @property
    def authority(self) -> Tuple[str, Union[str, int]]:
        """
        Get ``(authority_name, code)`` tuple.

        :returns: ``("", "")`` when not available
        """
        if self._epsg is not None:
            return ("EPSG", self._epsg)

        if (r := self._crs.to_authority()) is not None:
            name, code = r
            try:
                return (name, int(code))
            except ValueError:  # pragma: nocover
                return (name, code)

        return ("", "")

    def __str__(self) -> str:
        return self._str

    def __hash__(self) -> int:
        return hash(self._str)

    def __repr__(self) -> str:
        return f"CRS('{self._str}')"

    def __eq__(self, other) -> bool:
        if not isinstance(other, CRS):
            try:
                other = CRS(other)
            except Exception:  # pylint: disable=broad-except
                return False

        if self._crs is other._crs:
            return True

        if self.epsg is not None and other.epsg is not None:
            return self.epsg == other.epsg

        return self._crs == other._crs

    def __ne__(self, other) -> bool:
        return not self == other

    @property
    def proj(self) -> _CRS:
        """Access :py:class:`pyproj.crs.CRS` object that this wraps."""
        return self._crs

    @property
    def valid_region(self) -> Optional["geom.Geometry"]:
        """
        Return valid region of this CRS.

        :returns: Bounding box in Lon/Lat as a 4 point Polygon in EPSG:4326.
        :returns: ``None`` if valid region is not defined for this CRS
        """

        from . import geom  # pylint: disable=import-outside-toplevel

        aou = self._crs.area_of_use
        if aou is None:
            return None
        return geom.box(aou.west, aou.south, aou.east, aou.north, "EPSG:4326")

    @property
    def crs_str(self) -> str:
        """DEPRECATED"""
        warnings.warn(
            "Please use `str(crs)` instead of `crs.crs_str`",
            category=DeprecationWarning,
        )
        return self._str

    def transformer_to_crs(
        self, other: "CRS", always_xy: bool = True
    ) -> Callable[[Any, Any], Tuple[Any, Any]]:
        """
        Build coordinate transformer to other projection.

        Returns a function that maps ``x, y -> x', y'`` where ``x, y`` are coordinates in this CRS,
        stored either as scalars or :py:class:`numpy.ndarray` objects, and ``x', y'`` are the same
        points in the ``other`` CRS.

        :param other:
              Destination CRS

        :param always_xy:
              If true, the transform method will accept as input and return as output coordinates
              using the traditional GIS order, that is longitude, latitude for geographic CRS and
              easting, northing for most projected CRS.
        """

        # pylint: disable=protected-access
        tr = _make_crs_transform(self._crs, other._crs, always_xy=always_xy)

        def result(x, y, **kw):
            rx, ry = tr.transform(x, y, **kw)  # pylint: disable=unpacking-non-sequence

            if not isinstance(rx, numpy.ndarray) or not isinstance(ry, numpy.ndarray):
                return (rx, ry)

            missing = numpy.isnan(rx) | numpy.isnan(ry)
            rx[missing] = numpy.nan
            ry[missing] = numpy.nan
            return (rx, ry)

        return result

    def __dask_tokenize__(self):
        return ("odc.geo.crs.CRS", str(self))

    @staticmethod
    def utm(
        x: Union[float, int, XY[float], "geom.Geometry", "geom.BoundingBox"],
        y: Optional[float] = None,
        /,
        datum_name: str = "WGS 84",
    ) -> "CRS":
        """
        Construct appropriate UTM CRS for a given point.

        Uses CRS database query methods from :py:mod:`pyproj` to locate
        appropriate UTM CRS.

        :params datum_name: The name of the datum in the CRS name ('NAD27', 'NAD83', 'WGS 84', ...)
        """
        # pylint: disable=import-outside-toplevel,no-name-in-module
        from pyproj.database import query_utm_crs_info

        from . import geom

        if isinstance(x, geom.BoundingBox):
            _bbox = x
        elif isinstance(x, geom.Geometry):
            if x.crs is not None:
                _bbox = x.to_crs("epsg:4326").boundingbox
            else:
                # assume already in lon/lat
                _bbox = x.boundingbox
        elif isinstance(x, (float, int)):
            if y is None:
                y = 0.0
            _bbox = geom.BoundingBox(x, y, x, y)
        else:
            x, y = x.xy
            _bbox = geom.BoundingBox(x, y, x, y)

        return _pick_best_crs(
            _bbox.polygon,
            [
                CRS(f"{info.auth_name}:{info.code}")
                for info in query_utm_crs_info(
                    datum_name=datum_name,
                    area_of_interest=_bbox.aoi,
                )
            ],
        )


class CRSMismatchError(ValueError):
    """
    CRS Mismatch Error.

    Raised when geometry operation is attempted on geometries in different
    coordinate references.
    """


# fmt: off
@overload
def norm_crs(crs: SomeCRS) -> CRS: ...
@overload
def norm_crs(crs: SomeCRS, ctx: Any) -> CRS: ...
@overload
def norm_crs(crs: None) -> None: ...
@overload
def norm_crs(crs: None, ctx: Any) -> None: ...
# fmt: on


def norm_crs(crs: MaybeCRS, ctx=None) -> Optional[CRS]:
    """Normalise CRS representation."""
    if isinstance(crs, CRS):
        return crs
    if crs is None:
        return None
    if isinstance(crs, str):
        _txt = crs.lower()
        if _txt.startswith("utm"):
            assert ctx is not None

            utm_crs = CRS.utm(ctx)
            if _txt == "utm":
                return utm_crs

            utm_zone = utm_crs.proj.utm_zone
            epsg = utm_crs.epsg
            assert utm_zone is not None
            assert epsg is not None
            if _txt == "utm-n" and utm_zone.endswith("S"):
                utm_crs = CRS(epsg - 100)
            elif _txt == "utm-s" and utm_zone.endswith("N"):
                utm_crs = CRS(epsg + 100)
            return utm_crs
    return CRS(crs)


def norm_crs_or_error(crs: MaybeCRS, ctx=None) -> CRS:
    """Normalise CRS representation, raise error if input is ``None``."""
    if isinstance(crs, CRS):
        return crs
    if crs is None:
        raise ValueError("Expect valid CRS")
    if isinstance(crs, str):
        crs = norm_crs(crs, ctx)
        assert crs is not None
        return crs
    return CRS(crs)


def crs_units_per_degree(
    crs: SomeCRS,
    lon: Union[float, Tuple[float, float]],
    lat: float = 0,
    step: float = 0.1,
) -> float:
    """
    Helper method for converting resolution between meters/degrees.

    Compute number of CRS units per degree for a projected CRS at a given location
    in lon/lat.

    Location can be supplied as a tuple or as two arguments.

    :param crs: CRS
    :param lon: Either longitude or ``(lon, lat)`` tuple
    :param lat: Latitude or ignored if ``lon`` was a tuple
    :param step: Length of the segment in degrees used to estimate relative scale change

    :returns:  A floating number ``S`` such that ``S*degrees -> meters``
    """

    from . import geom  # pylint: disable=import-outside-toplevel

    if isinstance(lon, tuple):
        lon, lat = lon

    lon2 = lon + step
    if lon2 > 180:
        lon2 = lon - step

    ll = geom.line([(lon, lat), (lon2, lat)], "EPSG:4326")
    xy = ll.to_crs(crs)

    return xy.length / step


def _pick_best_crs(poly: "geom.Geometry", crs_candidates: List[CRS]) -> CRS:
    # pylint: disable=import-outside-toplevel
    from . import geom

    def overlap_pct(crs: CRS) -> float:
        crs_region = crs.valid_region
        if crs_region is None:
            return 1  # pragma: nocover
        return (crs_region & poly).area / poly.area

    if len(crs_candidates) < 1:
        raise ValueError("No candidate CRSs found")

    if len(crs_candidates) > 1 and poly.area > 1e-9:
        if poly.crs is None:
            poly = geom.Geometry(poly.geom, "epsg:4326")

        crs_candidates = sorted(crs_candidates, key=overlap_pct, reverse=True)

    return crs_candidates[0]


if TYPE_CHECKING:
    from . import geom  # pragma: no cover
