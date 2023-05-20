# This file is part of the Open Data Cube, see https://opendatacube.org for more information
#
# Copyright (c) 2015-2020 ODC Contributors
# SPDX-License-Identifier: Apache-2.0
import itertools
import math

import numpy
import pytest
from pytest import approx

from odc.geo import CRS, res_, resyx_, xy_, yx_
from odc.geo.geom import polygon
from odc.geo.gridspec import GridSpec
from odc.geo.testutils import SAMPLE_WKT_WITHOUT_AUTHORITY

# pylint: disable=protected-access,use-implicit-booleaness-not-comparison
# pylint: disable=comparison-with-itself,unnecessary-comprehension


def test_gridspec():
    gs = GridSpec(
        crs=CRS("EPSG:4326"),
        tile_shape=(10, 10),
        resolution=0.1,
        origin=xy_(10, 10),
    )
    assert gs.tile_shape == (10, 10)
    assert gs.tile_size == xy_(1, 1)

    assert gs == GridSpec(
        gs.crs, gs.tile_shape, resolution=resyx_(-0.1, 0.1), origin=gs.origin
    )
    assert gs == GridSpec(gs.crs, gs.tile_shape, resolution=res_(0.1), origin=gs.origin)

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
    poly = gs[3, 4].extent
    ((c1, gbox1),) = list(gs.tiles_from_geopolygon(poly, geobox_cache=cache))
    ((c2, gbox2),) = list(gs.tiles_from_geopolygon(poly, geobox_cache=cache))

    assert c1 == (3, 4) and c2 == c1
    assert gbox1 is gbox2

    assert "4326" in str(gs)
    assert "4326" in repr(gs)
    assert gs == gs
    assert (gs == {}) is False
    assert gs.dimensions == ("latitude", "longitude")

    assert GridSpec("epsg:3857", (100, 100), 1).alignment == xy_(0, 0)
    assert GridSpec("epsg:3857", (100, 100), resyx_(1, 1)).dimensions == ("y", "x")

    assert GridSpec("epsg:3857", (10, 20), 11.0) == GridSpec(
        "epsg:3857", (10, 20), resyx_(-11, 11)
    )
    assert GridSpec("epsg:3857", (10, 20), 11) == GridSpec(
        "epsg:3857", (10, 20), resyx_(-11, 11)
    )

    # missing shape parameter
    with pytest.raises(ValueError):
        GridSpec.from_sample_tile(poly)


def test_non_square():
    gs = GridSpec(
        crs=CRS("EPSG:4326"),
        tile_shape=(30, 10),
        resolution=0.1,
        origin=xy_(0.05, 0.0),
    )
    assert gs.tile_shape == (30, 10)
    assert gs.tile_size == yx_(3, 1)
    assert gs.alignment == xy_(0.05, 0)


def test_web_tiles():
    TSZ0 = 6_378_137 * 2 * math.pi
    epsg3857 = CRS("epsg:3857")

    gs = GridSpec.web_tiles(0)
    assert gs.crs == epsg3857
    assert gs.tile_shape == (256, 256)
    assert gs.tile_size.xy == approx((TSZ0, TSZ0))

    gs = GridSpec.web_tiles(1)
    assert gs.crs == epsg3857
    assert gs.tile_shape == (256, 256)
    assert gs.tile_size.xy == approx((TSZ0 / 2, TSZ0 / 2))


@pytest.mark.parametrize(
    "gs, tiles_to_check",
    [
        (GridSpec.web_tiles(3), itertools.product(range(8), range(8))),
        (GridSpec(4326, (360, 360), 0.01), [(0, 0), (1, 3), (2, 1)]),
    ],
)
def test_tiles_tight_query_issue_97(gs: GridSpec, tiles_to_check):
    for idx in tiles_to_check:
        bbox = gs[idx].boundingbox
        tiles = list(gs.tiles(bbox))
        assert len(tiles) == 1
        ((_idx, gbox),) = tiles
        assert idx == _idx
        assert gs[idx] == gbox


def test_geojson():
    gs = GridSpec.web_tiles(3)
    gjson = gs.geojson()

    assert gjson["type"] == "FeatureCollection"
    assert len(gjson["features"]) == (2**3) ** 2

    gjson = gs.geojson(geopolygon=gs.crs.valid_region.buffer(-0.1))
    assert len(gjson["features"]) == (2**3) ** 2

    gjson = gs.geojson(
        bbox=gs.crs.valid_region.buffer(-0.1).to_crs("epsg:3857").boundingbox
    )
    assert len(gjson["features"]) == (2**3) ** 2

    crs = CRS(SAMPLE_WKT_WITHOUT_AUTHORITY)
    gs = GridSpec(crs, (10, 10), resolution=6_378_137 * 2 * math.pi)
    gjson = gs.geojson()
    assert len(gjson["features"]) > 0
