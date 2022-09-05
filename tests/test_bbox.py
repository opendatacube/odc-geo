import pytest
from affine import Affine
from pyproj.aoi import AreaOfInterest

from odc.geo import BoundingBox, wh_
from odc.geo.geom import bbox_intersection, bbox_union
from odc.geo.testutils import epsg3857


def test_boundingbox():
    bb = BoundingBox(0, 3, 2, 4)
    assert bb.width == 2
    assert bb.height == 1
    assert bb.width == bb.span_x
    assert bb.height == bb.span_y
    assert bb.shape == (bb.height, bb.width)
    assert bb.crs is None
    assert bb.bbox == (0, 3, 2, 4)
    assert "crs=" not in str(bb)
    assert "crs=" not in repr(bb)
    assert bb.aoi == AreaOfInterest(0, 3, 2, 4)

    bb = BoundingBox(0, 3, 2.1, 4, "epsg:4326")
    assert bb.width == 2
    assert bb.height == 1
    assert bb.span_x == 2.1
    assert bb.width != bb.span_x
    assert bb.height == bb.span_y
    assert bb.crs.epsg == 4326
    assert "crs=" in str(bb)
    assert "crs=" in repr(bb)
    assert bb.aoi == AreaOfInterest(0, 3, 2.1, 4)

    assert BoundingBox.from_xy(bb.range_x, bb.range_y, bb.crs) == bb

    assert BoundingBox.from_xy((1, 2), (10, 20)) == (1, 10, 2, 20)
    assert BoundingBox.from_xy((2, 1), (20, 10)) == (1, 10, 2, 20)
    assert BoundingBox.from_points((1, 11), (2, 22)) == (1, 11, 2, 22)
    assert BoundingBox.from_points((1, 22), (2, 11)) == (1, 11, 2, 22)

    assert BoundingBox(1, 3, 10, 20).buffered(1, 3) == (0, 0, 11, 23)
    assert BoundingBox(1, 3, 10, 20).buffered(1) == (0, 2, 11, 21)

    assert BoundingBox.from_transform(
        wh_(10, 12), Affine.translation(1, 3), "epsg:3857"
    ) == (1, 3, 11, 15)
    assert (
        BoundingBox.from_transform(
            wh_(10, 12), Affine.translation(1, 3), "epsg:3857"
        ).crs
        == "epsg:3857"
    )

    assert bb.polygon.boundingbox == bb

    bb = BoundingBox(0, 3, 2, 4, "epsg:3857")
    assert bb.aoi == bb.to_crs("epsg:4326").aoi
    assert bb.aoi != AreaOfInterest(0, 3, 2, 4)


@pytest.mark.parametrize("crs", [None, "epsg:4326", epsg3857])
def test_bbox_union(crs):
    b1 = BoundingBox(0, 1, 10, 20, crs)
    b2 = BoundingBox(5, 6, 11, 22, crs)

    assert bbox_union([b1]) == b1
    assert bbox_union([b2]) == b2

    bb = bbox_union(iter([b1, b2]))
    assert bb == BoundingBox(0, 1, 11, 22, crs)

    bb = bbox_union(iter([b2, b1] * 10))
    assert bb == BoundingBox(0, 1, 11, 22, crs)

    assert b2 | b1 == bbox_union([b2, b1])
    assert b1 | b2 == bbox_union([b2, b1])


@pytest.mark.parametrize("crs", [None, "epsg:4326", epsg3857])
def test_bbox_intersection(crs):
    b1 = BoundingBox(0, 1, 10, 20, crs)
    b2 = BoundingBox(3, -10, 12, 8, crs)

    assert b1 & b2 == bbox_intersection([b1, b2])
    assert b2 & b1 == bbox_intersection([b1, b2])
    assert b1 & b1 == b1
    assert b2 & b2 == b2
    assert bbox_intersection([b1, b2]) == BoundingBox(3, 1, 10, 8, crs)
    assert b1 & b2 == BoundingBox(3, 1, 10, 8, crs)


@pytest.mark.parametrize(
    "crss",
    [
        (None, "epsg:4326", epsg3857),
        ("epsg:4326", "epsg:3857"),
        (*["epsg:4326"] * 4, "epsg:3857"),
    ],
)
def test_bbox_crs_mismatch(crss):
    with pytest.raises(ValueError):
        _ = bbox_union(BoundingBox(0, 0, 1, 1, crs) for crs in crss)

    with pytest.raises(ValueError):
        _ = bbox_intersection(BoundingBox(0, 0, 1, 1, crs) for crs in crss)


def test_map_bounds():
    bbox = BoundingBox(-180, -90, 180, 90, "epsg:4326")
    assert bbox.map_bounds() == ((-90, -180), (90, 180))

    bbox = BoundingBox(-180, -90, 180, 90)
    assert bbox.map_bounds() == ((-90, -180), (90, 180))

    bbox = BoundingBox(-10, -20, 24, 0, "epsg:4326")
    assert bbox.to_crs("epsg:3857").map_bounds() == ((-20, -10), (0, 24))


def test_bbox_to_crs():
    bbox = BoundingBox(-10, -20, 100, 0, "epsg:4326")
    assert bbox.to_crs("epsg:3857") == bbox.polygon.to_crs("epsg:3857").boundingbox


@pytest.mark.parametrize(
    "bb,expect",
    [
        ((-10, -20, 100, 0), (-10, -20, 100, 0)),
        ((-10.1, 2.3, 10.9, 2.4), (-11, 2, 11, 3)),
        ((-10.9, 2.6, 10.9, 2.9), (-11, 2, 11, 3)),
        ((0.9, 1.9, 1.01, 5.02), (0, 1, 2, 6)),
    ],
)
@pytest.mark.parametrize("crs", ["epsg:4326", None, "epsg:3857"])
def test_round(bb, expect, crs):
    expect = BoundingBox(*expect, crs)
    bbox = BoundingBox(*bb, crs)

    assert bbox.round().round() == bbox.round()
    assert bbox.round().polygon.contains(bbox.polygon)
    assert expect == bbox.round()
