# This file is part of the Open Data Cube, see https://opendatacube.org for more information
#
# Copyright (c) 2015-2020 ODC Contributors
# SPDX-License-Identifier: Apache-2.0
import numpy

from odc.geo import CRS, BoundingBox, polygon
from odc.geo.gridspec import GridSpec


def test_gridspec():
    gs = GridSpec(
        crs=CRS("EPSG:4326"),
        tile_size=(1, 1),
        resolution=(-0.1, 0.1),
        origin=(10, 10),
    )
    poly = polygon(
        [(10, 12.2), (10.8, 13), (13, 10.8), (12.2, 10), (10, 12.2)],
        crs=CRS("EPSG:4326"),
    )
    cells = {index: geobox for index, geobox in list(gs.tiles_from_geopolygon(poly))}
    assert set(cells.keys()) == {(0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1)}
    assert numpy.isclose(
        cells[(2, 0)].coordinates["longitude"].values,
        numpy.linspace(12.05, 12.95, num=10),
    ).all()
    assert numpy.isclose(
        cells[(2, 0)].coordinates["latitude"].values,
        numpy.linspace(10.95, 10.05, num=10),
    ).all()

    # check geobox_cache
    cache = {}
    poly = gs.tile_geobox((3, 4)).extent
    ((c1, gbox1),) = list(gs.tiles_from_geopolygon(poly, geobox_cache=cache))
    ((c2, gbox2),) = list(gs.tiles_from_geopolygon(poly, geobox_cache=cache))

    assert c1 == (3, 4) and c2 == c1
    assert gbox1 is gbox2

    assert "4326" in str(gs)
    assert "4326" in repr(gs)
    assert gs == gs
    assert (gs == {}) is False
    assert gs.dimensions == ("latitude", "longitude")

    assert GridSpec(CRS("epsg:3857"), (100, 100), (1, 1)).alignment == (0, 0)
    assert GridSpec(CRS("epsg:3857"), (100, 100), (1, 1)).dimensions == ("y", "x")


def test_gridspec_upperleft():
    """Test to ensure grid indexes can be counted correctly from bottom left or top left"""
    tile_bbox = BoundingBox(
        left=1934400.0, top=2414800.0, right=2084400.000, bottom=2264800.000
    )
    bbox = BoundingBox(left=1934615, top=2379460, right=1937615, bottom=2376460)
    # Upper left - validated against WELD product tile calculator
    # http://globalmonitoring.sdstate.edu/projects/weld/tilecalc.php
    gs = GridSpec(
        crs=CRS("EPSG:5070"),
        tile_size=(-150000, 150000),
        resolution=(-30, 30),
        origin=(3314800.0, -2565600.0),
    )
    cells = {index: geobox for index, geobox in list(gs.tiles(bbox))}
    assert set(cells.keys()) == {(30, 6)}
    assert cells[(30, 6)].extent.boundingbox == tile_bbox

    gs = GridSpec(
        crs=CRS("EPSG:5070"),
        tile_size=(150000, 150000),
        resolution=(-30, 30),
        origin=(14800.0, -2565600.0),
    )
    cells = {index: geobox for index, geobox in list(gs.tiles(bbox))}
    assert set(cells.keys()) == {
        (30, 15)
    }  # WELD grid spec has 21 vertical cells -- 21 - 6 = 15
    assert cells[(30, 15)].extent.boundingbox == tile_bbox


def test_grid_range():
    assert list(GridSpec._grid_range(-4.0, -1.0, 3.0)) == [-2, -1]
    assert list(GridSpec._grid_range(+1.0, 4.0, -3.0)) == [-2, -1]
    assert list(GridSpec._grid_range(-3.0, 0.0, 3.0)) == [-1]
    assert list(GridSpec._grid_range(-2.0, 1.0, 3.0)) == [-1, 0]
    assert list(GridSpec._grid_range(-1.0, 2.0, 3.0)) == [-1, 0]
    assert list(GridSpec._grid_range(+0.0, 3.0, 3.0)) == [0]
    assert list(GridSpec._grid_range(+1.0, 4.0, 3.0)) == [0, 1]
