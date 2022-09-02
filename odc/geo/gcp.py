# This file is part of the Open Data Cube, see https://opendatacube.org for more information
#
# Copyright (c) 2015-2022 ODC Contributors
# SPDX-License-Identifier: Apache-2.0
from typing import List, Optional, Tuple, Union

import numpy as np
from affine import Affine

from .crs import CRS, MaybeCRS, SomeCRS, norm_crs
from .geobox import GeoBox, GeoBoxBase
from .geom import Geometry, multipoint
from .math import Poly2d, affine_from_pts, align_up, resolution_from_affine, unstack_xy
from .types import XY, MaybeInt, Resolution, SomeShape, wh_

SomePointSet = Union[np.ndarray, Geometry, List[Geometry], List[XY[float]]]


def _points_to_array(pts: SomePointSet) -> Tuple[np.ndarray, Optional[CRS]]:
    if isinstance(pts, np.ndarray):
        return pts, None
    if isinstance(pts, Geometry):
        return np.asarray([pt.coords[0] for pt in pts.geoms]), pts.crs

    def _xy(pt: Union[Geometry, XY[float]]) -> Tuple[float, float]:
        if isinstance(pt, Geometry):
            ((x, y),) = pt.coords
        else:
            x, y = pt.xy
        return (x, y)

    crs = getattr(pts[0], "crs", None)
    return np.asarray([_xy(pt) for pt in pts]), crs


class GCPMapping:
    """
    Shared state for GCPGeoBox.
    """

    def __init__(
        self,
        pix: SomePointSet,
        wld: SomePointSet,
        crs: MaybeCRS = None,
    ):
        pix, _ = _points_to_array(pix)
        wld, _crs = _points_to_array(wld)

        if crs is None:
            crs = _crs

        # Nx2
        assert pix.shape == wld.shape
        assert pix.shape[1] == 2

        self._pix = pix
        self._wld = wld
        self._crs = norm_crs(crs)
        self._approx_affine: Optional[Affine] = None

        self._p2w: Optional[Poly2d] = None
        self._w2p: Optional[Poly2d] = None

    @property
    def p2w(self) -> Poly2d:
        """Pixel to world."""
        if self._p2w is not None:
            return self._p2w

        self._p2w = Poly2d.fit(self._pix, self._wld)
        return self._p2w

    @property
    def w2p(self) -> Poly2d:
        """World to pixel."""
        if self._w2p is not None:
            return self._w2p

        self._w2p = Poly2d.fit(self._wld, self._pix)
        return self._w2p

    @property
    def crs(self) -> Optional[CRS]:
        return self._crs

    @property
    def approx(self) -> Affine:
        if self._approx_affine is not None:
            return self._approx_affine

        pix = unstack_xy(self._pix)
        wld = unstack_xy(self._wld)
        self._approx_affine = affine_from_pts(pix, wld)

        return self._approx_affine

    @property
    def resolution(self) -> Resolution:
        """Resolution, pixel size in CRS units."""
        return resolution_from_affine(self.approx)

    def points(self) -> Tuple[Geometry, Geometry]:
        """Return multipoint geometries for (Pixel, World)."""
        return (
            multipoint(self._pix.tolist(), None),
            multipoint(self._wld.tolist(), self.crs),
        )

    def __dask_tokenize__(self):
        return (
            "odc.geo._gcp.GCPMapping",
            str(self._crs),
            self._wld,
            self._pix,
        )

    @staticmethod
    def from_rio(src, output_crs: MaybeCRS = None) -> "GCPMapping":
        """
        Construct from an open rasterio file.
        """
        # pylint: disable=import-outside-toplevel
        from .converters import extract_gcps

        return GCPMapping(*extract_gcps(src, output_crs))


class GCPGeoBox(GeoBoxBase):
    """
    Ground Control Point based GeoBox.

    """

    __slots__ = ("_mapping",)

    def __init__(
        self, shape: SomeShape, mapping: GCPMapping, affine: Optional[Affine] = None
    ):
        if affine is None:
            affine = Affine.identity()
        GeoBoxBase.__init__(self, shape, affine, mapping.crs)
        self._mapping = mapping

    def __getitem__(self, roi) -> "GCPGeoBox":
        _shape, _affine = self.compute_crop(roi)
        return GCPGeoBox(shape=_shape, mapping=self._mapping, affine=_affine)

    @property
    def approx(self) -> GeoBox:
        """
        Compute best fitting linear GeoBox.
        """
        return GeoBox(self.shape, self._mapping.approx * self._affine, self.crs)

    @property
    def resolution(self) -> Resolution:
        """Resolution, pixel size in CRS units."""
        return self.approx.resolution

    def wld2pix(self, x, y):
        x, y = self._mapping.w2p(x, y)
        return (~self._affine) * (x, y)

    def pix2wld(self, x, y):
        x, y = self._affine * (x, y)
        wx, wy = self._mapping.p2w(x, y)
        return (wx, wy)

    def __hash__(self):
        return hash((*self._shape, self._affine, self._crs, id(self._mapping)))

    @property
    def linear(self) -> bool:
        return False

    @property
    def axis_aligned(self):
        return False

    @property
    def center_pixel(self) -> "GCPGeoBox":
        """
        GeoBox of a center pixel.
        """
        return self[self.shape.map(lambda x: x // 2).yx]

    def map_bounds(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Query bounds in folium/ipyleaflet style.

        Returns SW, and NE corners in lat/lon order.
        ``((lat_w, lon_s), (lat_e, lon_n))``.
        """
        if self._crs is not None:
            x0, y0, x1, y1 = self.geographic_extent.boundingbox.bbox
        else:
            x0, y0, x1, y1 = self.extent.boundingbox.bbox
        return (y0, x0), (y1, x1)

    def to_crs(self, crs: SomeCRS, **_) -> "GCPGeoBox":
        """
        Project GCPs to a different CRS.

        extra arguments undesrtood by GeoBox.to_crs are simply ignored, but we do
        accept them to make generic code easier to write.
        """
        assert self._crs is not None
        pix, wld = self._mapping.points()
        if not self._affine.is_identity:
            pix = pix.transform(~self._affine)
        wld = wld.to_crs(crs)
        mapping = GCPMapping(pix, wld, wld.crs)
        return GCPGeoBox(self._shape, mapping)

    def pad(self, padx: int, pady: MaybeInt = None) -> "GCPGeoBox":
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
        return GCPGeoBox(shape, self._mapping, A)

    def pad_wh(self, alignx: int = 16, aligny: MaybeInt = None) -> "GCPGeoBox":
        """
        Possibly expand :py:class:`~odc.geo.geobox.GeoBox` by a few pixels.

        Find nearest ``width``/``height`` that are multiples of the desired factor. And return a new
        geobox that is slighly taller and/or wider covering roughly the same region. The new geobox
        will have the same CRS and transform but possibly larger shape.
        """
        aligny = alignx if aligny is None else aligny
        ny, nx = (align_up(sz, n) for sz, n in zip(self._shape.yx, (aligny, alignx)))

        return GCPGeoBox((ny, nx), self._mapping, self._affine)

    def zoom_out(self, factor: float) -> "GCPGeoBox":
        """
        Compute :py:class:`~odc.geo.geobox.GCPGeoBox` with changed resolution.

        - ``factor > 1`` implies smaller width/height, fewer but bigger pixels
        - ``factor < 1`` implies bigger width/height, more but smaller pixels

        :returns:
           GCPGeoBox covering the same region but with different pixels (i.e. lower or higher resolution)
        """
        _shape, _affine = self.compute_zoom_out(factor)
        return GCPGeoBox(_shape, self._mapping, _affine)

    def zoom_to(self, shape: Union[SomeShape, int, float]) -> "GCPGeoBox":
        """
        Compute :py:class:`~odc.geo.geobox.GCPGeoBox` with changed resolution.

        When supplied a single integer scale longest dimension to match that.

        :returns:
          GCPGeoBox covering the same region but with different number of pixels and therefore resolution.
        """
        _shape, _affine = self.compute_zoom_to(shape)
        return GCPGeoBox(_shape, self._mapping, _affine)

    def __str__(self):
        return self.__repr__()

    def __repr__(self) -> str:
        return f"GCPGeoBox({self._shape.xy!r}, {self.crs!r})"

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, GCPGeoBox):
            return False

        return (
            self._shape == __o.shape
            and self._mapping is __o._mapping
            and self._affine == __o._affine
        )

    def gcps(self):
        """Extract GCPs in rasterio format."""
        # pylint: disable=import-outside-toplevel
        from rasterio.control import GroundControlPoint

        pix, wld = self._mapping.points()

        a_inv = ~self._affine

        def to_gcp(pix: Geometry, wld: Geometry, _id) -> GroundControlPoint:
            ((wx, wy),) = wld.coords

            ((px, py),) = pix.coords
            px, py = a_inv * (px, py)
            return GroundControlPoint(row=py, col=px, x=wx, y=wy, id=_id)

        return [
            to_gcp(p, w, idx) for idx, (p, w) in enumerate(zip(pix.geoms, wld.geoms))
        ]

    def __dask_tokenize__(self):
        return (
            "odc.geo._gcp.GCPGeoBox",
            *self._mapping.__dask_tokenize__()[1:],
            *self._shape.yx,
            *self._affine[:6],
        )

    @staticmethod
    def from_rio(src, output_crs: MaybeCRS = None) -> "GCPGeoBox":
        """
        Construct from an open rasterio file.
        """
        return GCPGeoBox(
            wh_(src.width, src.height), GCPMapping.from_rio(src, output_crs)
        )
