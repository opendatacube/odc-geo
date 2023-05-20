# This file is part of the Open Data Cube, see https://opendatacube.org for more information
#
# Copyright (c) 2015-2020 ODC Contributors
# SPDX-License-Identifier: Apache-2.0
"""GridSpec class."""
import math
from typing import Any, Dict, Iterator, Optional, Tuple

from affine import Affine

from . import geom
from .crs import SomeCRS, norm_crs_or_error
from .geobox import GeoBox
from .geom import BoundingBox, Geometry
from .math import Bin1D
from .types import (
    XY,
    Index2d,
    Shape2d,
    SomeIndex2d,
    SomeResolution,
    SomeShape,
    ixy_,
    res_,
    resyx_,
    shape_,
    xy_,
    yx_,
)


class GridSpec:
    """
    Definition for a regular spatial grid.

    :param crs:
       Coordinate System used to define the grid
    :param tile_shape:
       Size of each tile in pixels
    :param resolution:
       Size of each data point in the grid, in CRS units. ``Y`` will usually be negative.
    :param origin:
       Coordinate of a bottom-left corner of the ``(0,0)`` tile in CRS units. Default is
       ``xy_(0.0, 0.0)``
    :param flipx: when ``True`` grid index for X axis increments left to right
    :param flipy: when ``True`` grid index for Y axis increments top to bottom
    """

    def __init__(
        self,
        crs: SomeCRS,
        tile_shape: SomeShape,
        resolution: SomeResolution,
        origin: Optional[XY[float]] = None,
        flipx: bool = False,
        flipy: bool = False,
    ):
        tile_shape = shape_(tile_shape)
        resolution = res_(resolution)

        if origin is None:
            origin = xy_(0.0, 0.0)
        else:
            assert isinstance(origin, XY)

        self.crs = norm_crs_or_error(crs)
        self._shape = tile_shape
        self.resolution = resolution
        self.tile_size = xy_(
            tile_shape.x * abs(resolution.x),
            tile_shape.y * abs(resolution.y),
        )
        self.origin = origin

        ox, oy = origin.xy
        self._ybin = Bin1D(self.tile_size.y, oy, -1 if flipy else 1)
        self._xbin = Bin1D(self.tile_size.x, ox, -1 if flipx else 1)

    def __eq__(self, other):
        if not isinstance(other, GridSpec):
            return False

        return (
            self._shape == other._shape
            and self._ybin == other._ybin
            and self._xbin == other._xbin
            and self.crs == other.crs
        )

    @property
    def dimensions(self) -> Tuple[str, str]:
        """List of dimension names of the grid spec."""
        return self.crs.dimensions

    @property
    def alignment(self) -> XY[float]:
        """Pixel boundary alignment."""
        y, x = (
            orig % abs(res) for orig, res in zip(self.origin.yx, self.resolution.yx)
        )
        return yx_(y, x)

    @property
    def tile_shape(self) -> Shape2d:
        """Tile shape in pixels (Y,X order, like numpy)."""
        return self._shape

    def pt2idx(self, x: float, y: float) -> Index2d:
        """
        Compute tile index from a point.

        :param x: X coordinate of a point in CRS units
        :param y: Y coordinate of a point in CRS units
        :return:
          ``(ix, iy)``, index of a tile containing given point
        """
        return ixy_(self._xbin.bin(x), self._ybin.bin(y))

    def _tile_txy(self, tile_index: Index2d) -> XY[float]:
        """Location of 0,0 pixel in CRS units."""

        ix, iy = tile_index.xy
        rx, ry = self.resolution.xy

        x0, x1 = self._xbin[ix]
        tx = x0 if rx > 0 else x1

        y0, y1 = self._ybin[iy]
        ty = y0 if ry > 0 else y1

        return xy_(tx, ty)

    def tile_geobox(self, tile_index: SomeIndex2d) -> GeoBox:
        """
        Tile geobox.

        :param tile_index: ``(ix, iy)``
        """
        tile_index = ixy_(tile_index)
        tx, ty = self._tile_txy(tile_index).xy
        rx, ry = self.resolution.xy
        return GeoBox(self._shape, crs=self.crs, affine=Affine(rx, 0, tx, 0, ry, ty))

    def __getitem__(self, idx: SomeIndex2d) -> GeoBox:
        """Lookup :py:class:`~odc.geo.geobox.GeoBox` of a given tile."""
        return self.tile_geobox(idx)

    def idx_bounds(self, bounds: BoundingBox) -> Tuple[int, int, int, int]:
        """
        Convert bounds from CRS to index space.

        :param bounds: Query bounding box
        :return: ``x1, y1, x2, y2``, with closed/open range, i.e. ``[x1, x2), [y1, y2)``
        """
        assert self.crs == bounds.crs
        tol = 1e-8

        x1, y1, x2, y2 = bounds
        ix1, iy1 = self.pt2idx(x1 + tol, y1 + tol).xy
        ix2, iy2 = self.pt2idx(x2 - tol, y2 - tol).xy

        ix1, ix2 = sorted([ix1, ix2])
        iy1, iy2 = sorted([iy1, iy2])
        return ix1, iy1, ix2 + 1, iy2 + 1

    def tiles(
        self, bounds: BoundingBox, geobox_cache: Optional[dict] = None
    ) -> Iterator[Tuple[Tuple[int, int], GeoBox]]:
        """
        Query tiles overlapping with bounding box.

        Output is a sequence of ``tile_index``, :py:class:`odc.geo.geobox.GeoBox` tuples.

        .. note::

           Grid cells are referenced by coordinates ``(x, y)``, which is the opposite to the usual
           CRS dimension order.

        :param bounds:
           Boundary coordinates of the required grid
        :param geobox_cache:
           Optional cache to re-use geoboxes instead of creating new one each time
        :return:
          Iterator of tuples of grid index and corresponding :py:class:`odc.geo.geobox.GeoBox`
        """

        def geobox(tile_index):
            if geobox_cache is None:
                return self.tile_geobox(tile_index)

            gbox = geobox_cache.get(tile_index)
            if gbox is None:
                gbox = self.tile_geobox(tile_index)
                geobox_cache[tile_index] = gbox
            return gbox

        ix1, iy1, ix2, iy2 = map(int, self.idx_bounds(bounds))

        for iy in range(iy1, iy2):
            for ix in range(ix1, ix2):
                tile_index = (ix, iy)
                yield tile_index, geobox(tile_index)

    def tiles_from_geopolygon(
        self,
        geopolygon: Geometry,
        geobox_cache: Optional[dict] = None,
    ) -> Iterator[Tuple[Tuple[int, int], GeoBox]]:
        """
        Query tiles overlapping with a given polygon.

        Output is a sequence of ``tile_index``, :py:class:`odc.geo.geobox.GeoBox` tuples.

        :param geopolygon:
          Polygon to tile
        :param geobox_cache:
          Optional cache to re-use geoboxes instead of creating new one each turn: iterator of grid
          cells with :py:class:`odc.geo.geobox.GeoBox` tiles
        """
        geopolygon = geopolygon.to_crs(self.crs, check_and_fix=True)
        bbox = geopolygon.boundingbox

        for tile_index, tile_geobox in self.tiles(bbox, geobox_cache):
            if not geopolygon.disjoint(tile_geobox.extent):
                yield (tile_index, tile_geobox)

    def __str__(self) -> str:
        return f"GridSpec(crs={self.crs}, tile_shape={self._shape}, resolution={self.resolution})"

    def __repr__(self) -> str:
        return self.__str__()

    def geojson(
        self,
        *,
        bbox: Optional[BoundingBox] = None,
        geopolygon: Optional[Geometry] = None,
    ) -> Dict[str, Any]:
        """
        Render to GeoJSON.

        :param bbox:
           Limit output to tiles overlapping with the given bounding box (native CRS of the grid).

        :param geopolygon:
           Limit output to tiles overlapping with the given geometry (any CRS)

        :return: GeoJSON representation of the grid spec
        """
        if geopolygon is not None:
            _tiles = self.tiles_from_geopolygon(geopolygon)
        elif bbox is not None:
            _tiles = self.tiles(bbox)
        else:
            valid_region = self.crs.valid_region
            if valid_region is None:
                valid_region = geom.box(-180, -90, 180, 90, "epsg:4326")

            _tiles = self.tiles(
                valid_region.buffer(-0.05).to_crs(self.crs, resolution=0.5).boundingbox
            )

        props = {
            "native_crs": str(self.crs),
            "tile_shape": self._shape,
            "resolution": self.resolution,
        }

        features = []
        for (ix, iy), geobox in _tiles:
            features.append(geobox.extent.geojson(idx=f"{ix},{iy}"))
        return {"type": "FeatureCollection", "features": features, "properties": props}

    @staticmethod
    def from_sample_tile(
        box: Geometry,
        *,
        shape: SomeShape = (-1, -1),
        idx: SomeIndex2d = (0, 0),
        flipx: bool = False,
        flipy: bool = False,
    ) -> "GridSpec":
        """
        Construct :py:class:`odc.geo.gridspec.GridSpec` from a sample tile.

        Bounding box of one tile, it's index, and a shape of the tile in pixels fully define a grid.
        This method could be more convenient than canonical representation.

        :param box: Geometry of the tile in some CRS
        :param idx: ``ix, iy`` index of the sample tile, default ``(0, 0)``
        :param shape: ``height, width`` of the tile, must be supplied
        :param flipx: when ``True`` grid index for X axis increments left to right
        :param flipy: when ``True`` grid index for Y axis increments top to bottom
        """
        if shape == (-1, -1):
            raise ValueError("Must specify shape of the tile in pixels")
        shape = shape_(shape)
        idx = ixy_(idx)

        crs = norm_crs_or_error(box.crs)
        ix, iy = idx.xy
        nx, ny = shape.xy
        bbox = box.boundingbox

        xbin = Bin1D.from_sample_bin(ix, bbox.range_x, -1 if flipx else 1)
        ybin = Bin1D.from_sample_bin(iy, bbox.range_y, -1 if flipy else 1)

        origin = xy_(xbin.origin, ybin.origin)
        resolution = resyx_(-ybin.sz / ny, xbin.sz / nx)
        return GridSpec(
            crs, shape, resolution=resolution, origin=origin, flipx=flipx, flipy=flipy
        )

    @staticmethod
    def web_tiles(zoom: int, npix: int = 256) -> "GridSpec":
        """
        Construct :py:class:`~odc.geo.gridspec.GridSpec` that matches slippy tiles.

        Tile with index ``(0, 0)`` is at the top left corner of the map, and tile with index
        ``(2^zoom - 1, 2^zoom - 1)`` is at the bottom right.

        :param zoom:
           Zoom level, ``0`` is one single tile, ``1`` is ``2x2``, ``3`` is ``8x8``...

        :param npix:
           Usually tiles are ``256x256`` pixels wide. But you can override that.

        :return: Grid spec that encodes slippy tiles scheme in ``EPSG:3857``.
        """
        R = 6_378_137
        pi = math.pi
        tsz = pi * R * (2 ** (1 - zoom))  # in meters
        x, y = -pi * R, pi * R  # top-left corner of tile 0,0
        tile0 = geom.box(x, y - tsz, x + tsz, y, "epsg:3857")
        shape = (npix, npix)

        return GridSpec.from_sample_tile(tile0, shape=shape, idx=(0, 0), flipy=True)
