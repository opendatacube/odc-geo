# This file is part of the Open Data Cube, see https://opendatacube.org for more information
#
# Copyright (c) 2015-2020 ODC Contributors
# SPDX-License-Identifier: Apache-2.0
"""GridSpec class."""
import math
from typing import Iterator, Optional, Tuple

from affine import Affine

from .crs import CRS
from .geobox import GeoBox
from .geom import BoundingBox, Geometry, intersects


class GridSpec:
    """
    Definition for a regular spatial grid.

    :param CRS crs: Coordinate System used to define the grid
    :param [float,float] tile_size: (Y, X) size of each tile, in CRS units
    :param [float,float] resolution: (Y, X) size of each data point in the grid, in CRS units. Y will
                                   usually be negative.
    :param [float,float] origin: (Y, X) coordinates of a corner of the (0,0) tile in CRS units. default is (0.0, 0.0)
    """

    def __init__(
        self,
        crs: CRS,
        tile_size: Tuple[float, float],
        resolution: Tuple[float, float],
        origin: Optional[Tuple[float, float]] = None,
    ):
        self.crs = crs
        self.tile_size = tile_size
        self.resolution = resolution
        self.origin = origin or (0.0, 0.0)

    def __eq__(self, other):
        if not isinstance(other, GridSpec):
            return False

        return (
            self.crs == other.crs
            and self.tile_size == other.tile_size
            and self.resolution == other.resolution
            and self.origin == other.origin
        )

    @property
    def dimensions(self) -> Tuple[str, str]:
        """
        List of dimension names of the grid spec
        """
        return self.crs.dimensions

    @property
    def alignment(self) -> Tuple[float, float]:
        """
        Pixel boundary alignment
        """
        y, x = (orig % abs(res) for orig, res in zip(self.origin, self.resolution))
        return (y, x)

    @property
    def tile_shape(self) -> Tuple[int, int]:
        """
        Tile shape in pixels in Y,X order, like numpy
        """
        y, x = (int(abs(ts / res)) for ts, res in zip(self.tile_size, self.resolution))
        return (y, x)

    def tile_coords(self, tile_index: Tuple[int, int]) -> Tuple[float, float]:
        """
        Coordinate of the top-left corner of the tile in (Y,X) order

        :param tile_index: in X,Y order
        """

        def coord(index: int, resolution: float, size: float, origin: float) -> float:
            return (index + (1 if resolution < 0 < size else 0)) * size + origin

        y, x = (
            coord(index, res, size, origin)
            for index, res, size, origin in zip(
                tile_index[::-1], self.resolution, self.tile_size, self.origin
            )
        )
        return (y, x)

    def tile_geobox(self, tile_index: Tuple[int, int]) -> GeoBox:
        """
        Tile geobox.

        :param (int,int) tile_index:
        """
        res_y, res_x = self.resolution
        y, x = self.tile_coords(tile_index)
        h, w = self.tile_shape
        geobox = GeoBox(
            crs=self.crs, affine=Affine(res_x, 0.0, x, 0.0, res_y, y), width=w, height=h
        )
        return geobox

    def tiles(
        self, bounds: BoundingBox, geobox_cache: Optional[dict] = None
    ) -> Iterator[Tuple[Tuple[int, int], GeoBox]]:
        """
        Returns an iterator of tile_index, :py:class:`GeoBox` tuples across
        the grid and overlapping with the specified `bounds` rectangle.

        .. note::

           Grid cells are referenced by coordinates `(x, y)`, which is the opposite to the usual CRS
           dimension order.

        :param BoundingBox bounds: Boundary coordinates of the required grid
        :param dict geobox_cache: Optional cache to re-use geoboxes instead of creating new one each time
        :return: iterator of grid cells with :py:class:`GeoBox` tiles
        """

        def geobox(tile_index):
            if geobox_cache is None:
                return self.tile_geobox(tile_index)

            gbox = geobox_cache.get(tile_index)
            if gbox is None:
                gbox = self.tile_geobox(tile_index)
                geobox_cache[tile_index] = gbox
            return gbox

        tile_size_y, tile_size_x = self.tile_size
        tile_origin_y, tile_origin_x = self.origin
        for y in GridSpec._grid_range(
            bounds.bottom - tile_origin_y, bounds.top - tile_origin_y, tile_size_y
        ):
            for x in GridSpec._grid_range(
                bounds.left - tile_origin_x, bounds.right - tile_origin_x, tile_size_x
            ):
                tile_index = (x, y)
                yield tile_index, geobox(tile_index)

    def tiles_from_geopolygon(
        self,
        geopolygon: Geometry,
        tile_buffer: Optional[Tuple[float, float]] = None,
        geobox_cache: Optional[dict] = None,
    ) -> Iterator[Tuple[Tuple[int, int], GeoBox]]:
        """
        Returns an iterator of tile_index, :py:class:`GeoBox` tuples across
        the grid and overlapping with the specified `geopolygon`.

        .. note::

           Grid cells are referenced by coordinates `(x, y)`, which is the opposite to the usual CRS
           dimension order.

        :param Geometry geopolygon: Polygon to tile
        :param tile_buffer: Optional <float,float> tuple, (extra padding for the query
                            in native units of this GridSpec)
        :param dict geobox_cache: Optional cache to re-use geoboxes instead of creating new one each time
        :return: iterator of grid cells with :py:class:`GeoBox` tiles
        """
        geopolygon = geopolygon.to_crs(self.crs)
        bbox = geopolygon.boundingbox
        bbox = bbox.buffered(*tile_buffer) if tile_buffer else bbox

        for tile_index, tile_geobox in self.tiles(bbox, geobox_cache):
            tile_geobox = (
                tile_geobox.buffered(*tile_buffer) if tile_buffer else tile_geobox
            )

            if intersects(tile_geobox.extent, geopolygon):
                yield (tile_index, tile_geobox)

    @staticmethod
    def _grid_range(lower: float, upper: float, step: float) -> range:
        """
        Returns the indices along a 1D scale.

        Used for producing 2D grid indices.
        """
        if step < 0.0:
            lower, upper, step = -upper, -lower, -step
        assert step > 0.0
        return range(int(math.floor(lower / step)), int(math.ceil(upper / step)))

    def __str__(self) -> str:
        return f"GridSpec(crs={self.crs}, tile_size={self.tile_size}, resolution={self.resolution})"

    def __repr__(self) -> str:
        return self.__str__()
