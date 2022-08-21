# This file is part of the Open Data Cube, see https://opendatacube.org for more information
#
# Copyright (c) 2015-2020 ODC Contributors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from affine import Affine

from odc.geo import CRS, geom, ixy_, resyx_, wh_, xy_
from odc.geo.geobox import (
    AnchorEnum,
    GeoBox,
    bounding_box_in_pixel_domain,
    gbox_boundary,
    geobox_intersection_conservative,
    geobox_union_conservative,
    scaled_down_geobox,
)
from odc.geo.math import apply_affine, is_affine_st
from odc.geo.testutils import (
    epsg3577,
    epsg3857,
    epsg4326,
    esri54019,
    mkA,
    modis_crs,
    xy_from_gbox,
    xy_norm,
)

# pylint: disable=pointless-statement,too-many-statements,protected-access


def test_geobox_simple():
    t = GeoBox(
        (4000, 4000), Affine(0.00025, 0.0, 151.0, 0.0, -0.00025, -29.0), epsg4326
    )

    expect_lon = np.asarray(
        [
            151.000125,
            151.000375,
            151.000625,
            151.000875,
            151.001125,
            151.001375,
            151.001625,
            151.001875,
            151.002125,
            151.002375,
        ]
    )

    expect_lat = np.asarray(
        [
            -29.000125,
            -29.000375,
            -29.000625,
            -29.000875,
            -29.001125,
            -29.001375,
            -29.001625,
            -29.001875,
            -29.002125,
            -29.002375,
        ]
    )
    expect_resolution = pytest.approx((-0.00025, 0.00025))
    assert t.resolution.yx == expect_resolution
    assert t.rotate(33).resolution.yx == expect_resolution
    assert t.rotate(-11.37).resolution.yx == expect_resolution

    assert t.coordinates["latitude"].values.shape == (4000,)
    assert t.coordinates["longitude"].values.shape == (4000,)

    np.testing.assert_almost_equal(t.coords["latitude"].values[:10], expect_lat)
    np.testing.assert_almost_equal(t.coords["longitude"].values[:10], expect_lon)

    assert (t == "some random thing") is False

    # ensure GeoBox accepts string CRS
    assert isinstance(
        GeoBox(
            (4000, 4000), Affine(0.00025, 0.0, 151.0, 0.0, -0.00025, -29.0), "epsg:4326"
        ).crs,
        CRS,
    )

    # Check GeoBox class is hashable
    t_copy = GeoBox(t.shape, t.transform, t.crs)
    t_other = GeoBox(wh_(t.width + 1, t.height), t.transform, t.crs)
    assert t_copy is not t
    assert t == t_copy
    assert len({t, t, t_copy}) == 1
    assert len({t, t_copy, t_other}) == 2


def test_xy_from_geobox():
    gbox = GeoBox(wh_(3, 7), Affine.translation(10, 1000), epsg3857)
    xx, yy = xy_from_gbox(gbox)

    assert xx.shape == gbox.shape
    assert yy.shape == gbox.shape
    assert (xx[:, 0] == 10.5).all()
    assert (xx[:, 1] == 11.5).all()
    assert (yy[0, :] == 1000.5).all()
    assert (yy[6, :] == 1006.5).all()

    xx_, yy_, A = xy_norm(xx, yy)
    assert xx_.shape == xx.shape
    assert yy_.shape == yy.shape
    np.testing.assert_almost_equal((xx_.min(), xx_.max()), (0, 1))
    np.testing.assert_almost_equal((yy_.min(), yy_.max()), (0, 1))
    assert (xx_[0] - xx_[1]).sum() != 0
    assert (xx_[:, 0] - xx_[:, 1]).sum() != 0

    XX, YY = apply_affine(A, xx_, yy_)
    np.testing.assert_array_almost_equal(xx, XX)
    np.testing.assert_array_almost_equal(yy, YY)


def test_geobox():
    points_list = [
        [
            (148.2697, -35.20111),
            (149.31254, -35.20111),
            (149.31254, -36.331431),
            (148.2697, -36.331431),
        ],
        [
            (148.2697, 35.20111),
            (149.31254, 35.20111),
            (149.31254, 36.331431),
            (148.2697, 36.331431),
        ],
        [
            (-148.2697, 35.20111),
            (-149.31254, 35.20111),
            (-149.31254, 36.331431),
            (-148.2697, 36.331431),
        ],
        [
            (-148.2697, -35.20111),
            (-149.31254, -35.20111),
            (-149.31254, -36.331431),
            (-148.2697, -36.331431),
            (148.2697, -35.20111),
        ],
    ]
    for points in points_list:
        polygon = geom.polygon(points, crs=epsg3577)
        resolution = resyx_(-25, 25)
        geobox = GeoBox.from_geopolygon(polygon, resolution)

        # check single value resolution equivalence
        assert GeoBox.from_geopolygon(polygon, 25) == geobox
        assert GeoBox.from_geopolygon(polygon, 25.0) == geobox

        assert GeoBox.from_geopolygon(polygon, resolution, crs=geobox.crs) == geobox

        # check that extra padding added by alignment is smaller than pixel size
        assert abs(resolution.x) > abs(
            geobox.extent.boundingbox.left - polygon.boundingbox.left
        )
        assert abs(resolution.x) > abs(
            geobox.extent.boundingbox.right - polygon.boundingbox.right
        )
        assert abs(resolution.y) > abs(
            geobox.extent.boundingbox.top - polygon.boundingbox.top
        )
        assert abs(resolution.y) > abs(
            geobox.extent.boundingbox.bottom - polygon.boundingbox.bottom
        )

    A = mkA(0, scale=(10, -10), translation=(-48800, -2983006))

    w, h = 512, 256
    gbox = GeoBox(wh_(w, h), A, epsg3577)
    assert gbox.boundingbox.crs is epsg3577
    assert gbox.footprint("epsg:4326") == gbox.geographic_extent

    assert gbox.shape == (h, w)
    assert gbox.transform == A
    assert gbox.extent.crs == gbox.crs
    assert gbox.geographic_extent.crs == epsg4326
    assert gbox.extent.boundingbox.height == h * 10.0
    assert gbox.extent.boundingbox.width == w * 10.0
    assert gbox.alignment.yx == (4, 0)  # 4 because -2983006 % 10 is 4
    assert isinstance(str(gbox), str)
    assert "EPSG:3577" in repr(gbox)
    assert gbox.axis_aligned is True

    assert GeoBox((1, 1), mkA(0), epsg4326).geographic_extent.crs == epsg4326
    assert GeoBox((1, 1), mkA(0), None).dimensions == ("y", "x")

    g2 = gbox[:-10, :-20]
    assert g2.shape == (gbox.height - 10, gbox.width - 20)

    # step of 1 is ok
    g2 = gbox[::1, ::1]
    assert g2.shape == gbox.shape

    assert gbox[0].shape == (1, gbox.width)
    assert gbox[:3].shape == (3, gbox.width)
    assert gbox[0, 0] == gbox[:1, :1]
    assert gbox[0, 0:1] == gbox[:1, :1]
    assert gbox[0:1, 1] == gbox[:1, 1:2]

    with pytest.raises(NotImplementedError):
        gbox[::2, :]

    # too many slices
    with pytest.raises(ValueError):
        gbox[:1, :1, :]

    assert gbox.buffered(0, 10).shape == (gbox.height + 2 * 1, gbox.width)
    assert gbox.buffered(10).shape == (gbox.height + 2 * 1, gbox.width + 2 * 1)
    assert gbox.buffered(20, 30).shape == (gbox.height + 2 * 3, gbox.width + 2 * 2)

    assert (gbox | gbox) == gbox
    assert (gbox & gbox) == gbox
    assert gbox.is_empty() is False
    assert bool(gbox) is True

    assert (gbox[:3, :4] & gbox[3:, 4:]).is_empty()
    assert (gbox[:3, :4] & gbox[:3, 4:]).is_empty()
    assert (gbox[:3, :4] & gbox[30:, 40:]).is_empty()

    assert gbox[:3, :7].center_pixel == gbox[1, 3]
    assert gbox[:2, :4].center_pixel == gbox[1, 2]

    with pytest.raises(ValueError):
        geobox_intersection_conservative([])

    with pytest.raises(ValueError):
        geobox_union_conservative([])

    # can not combine across CRSs
    with pytest.raises(ValueError):
        bounding_box_in_pixel_domain(
            GeoBox((1, 1), mkA(0), epsg4326), GeoBox(ixy_(2, 3), mkA(0), epsg3577)
        )


def test_gbox_overlap_roi():
    gbox = GeoBox(wh_(100, 30), Affine.translation(0, 0), epsg4326)
    assert gbox.overlap_roi(gbox[3:7, 13:29]) == np.s_[3:7, 13:29]
    assert gbox[10:20, 5:10].overlap_roi(gbox) == np.s_[0:10, 0:5]

    a = gbox[10:20, 20:40]
    b = gbox[11:30, 10:25]
    assert a.overlap_roi(b) == np.s_[1:10, 0:5]


def test_gbox_boundary():
    geobox = GeoBox(wh_(6, 2), Affine.translation(0, 0), epsg4326)

    bb = gbox_boundary(geobox, 3)

    assert bb.shape == (4 + (3 - 2) * 4, 2)
    assert set(bb.T[0]) == {0.0, 3.0, 6.0}
    assert set(bb.T[1]) == {0.0, 1.0, 2.0}

    assert geobox.map_bounds() == geobox.boundingbox.map_bounds()


def test_geobox_scale_down():

    crs = CRS("EPSG:3857")

    A = mkA(0, (111.2, 111.2), translation=(125671, 251465))
    for s in [2, 3, 4, 8, 13, 16]:
        gbox = GeoBox(wh_(233 * s, 755 * s), A, crs)
        gbox_ = scaled_down_geobox(gbox, s)

        assert gbox_.width == 233
        assert gbox_.height == 755
        assert gbox_.crs is crs
        assert gbox_.extent.contains(gbox.extent)
        assert gbox.extent.difference(gbox.extent).area == 0.0

    gbox = GeoBox((1, 1), A, crs)
    for s in [2, 3, 5]:
        gbox_ = scaled_down_geobox(gbox, 3)

        assert gbox_.shape == (1, 1)
        assert gbox_.crs is crs
        assert gbox_.extent.contains(gbox.extent)


def test_non_st():
    A = mkA(rot=10, translation=(-10, 20), shear=0.3)
    assert is_affine_st(A) is False

    gbox = GeoBox((1, 1), A, "epsg:3857")

    assert gbox._confirm_axis_aligned() is False
    assert gbox.axis_aligned is False

    with pytest.raises(ValueError):
        assert gbox.coordinates


def test_from_polygon():
    box = geom.box(1, 13, 17, 37, "epsg:4326")
    gbox = GeoBox.from_geopolygon(box, 0.1)
    assert gbox.crs == box.crs
    assert gbox.extent.boundingbox == box.boundingbox
    assert gbox.resolution.y < 0

    # check that align= still works by name and by order
    assert gbox == GeoBox.from_geopolygon(gbox.extent, gbox.resolution, None, xy_(0, 0))
    assert gbox == GeoBox.from_geopolygon(gbox.extent, gbox.resolution, align=xy_(0, 0))

    box = geom.box(-1, 10, 2, 12, "epsg:4326")
    gbox = GeoBox.from_geopolygon(box, tight=True, shape=300)
    assert gbox.crs == box.crs
    assert gbox.shape.wh == (300, 200)
    assert gbox.resolution.x == -gbox.resolution.y
    assert gbox.aspect == 300 / 200

    gbox = GeoBox.from_geopolygon(box, tight=True, shape=(100, 29))
    assert gbox.crs == box.crs
    assert gbox.shape.wh == (29, 100)


def test_from_bbox():
    bbox = (1, 13, 17, 37)
    shape = (23, 47)
    gbox = GeoBox.from_bbox(bbox, tight=True, shape=shape)
    assert gbox.shape == shape
    assert gbox.crs == "epsg:4326"
    assert gbox.extent.boundingbox == bbox
    assert gbox.resolution.y < 0

    assert (
        GeoBox.from_bbox(bbox, gbox.crs, tight=True, resolution=gbox.resolution) == gbox
    )

    assert GeoBox.from_bbox(bbox, shape=shape, tight=False) != gbox
    assert GeoBox.from_bbox(bbox, resolution=gbox.resolution, tight=False) != gbox

    assert GeoBox.from_bbox(geom.BoundingBox(*bbox), shape=shape).crs == "epsg:4326"
    assert (
        GeoBox.from_bbox(geom.BoundingBox(*bbox, "epsg:3857"), shape=shape).crs
        == "epsg:3857"
    )

    gbox = GeoBox.from_bbox((0, 0, 2, 3), tight=True, shape=300, crs="epsg:3857")
    assert gbox.shape.wh == (200, 300)
    assert gbox.resolution.x == -gbox.resolution.y

    gbox = GeoBox.from_bbox((-1, 0, 2, 2), tight=True, shape=300, crs="epsg:3857")
    assert gbox.shape.wh == (300, 200)
    assert gbox.resolution.x == -gbox.resolution.y

    assert GeoBox.from_bbox(
        bbox, shape=shape, anchor=AnchorEnum.FLOATING
    ) == GeoBox.from_bbox(bbox, shape=shape, tight=True)

    assert GeoBox.from_bbox(
        bbox, shape=shape, anchor=AnchorEnum.EDGE
    ) == GeoBox.from_bbox(bbox, shape=shape, anchor=xy_(0, 0))

    assert GeoBox.from_bbox(
        bbox, shape=shape, anchor=AnchorEnum.CENTER
    ) == GeoBox.from_bbox(bbox, shape=shape, anchor=xy_(0.5, 0.5))

    # one of resolution= or shape= must be supplied
    with pytest.raises(ValueError):
        _ = GeoBox.from_bbox(bbox)


def test_outline():
    gbox = GeoBox.from_bbox([0, 0, 20, 10], "epsg:3857", shape=wh_(200, 100))
    assert gbox.outline() == gbox.outline("native")
    assert gbox.outline().crs == gbox.crs
    assert gbox.outline("geo").crs == "epsg:4326"
    assert gbox.outline("pixel").boundingbox == (0, 0, 200, 100)
    assert gbox.outline(notch=0).type == "GeometryCollection"


def test_svg():
    # smoke test only
    gbox = GeoBox.from_bbox([0, 0, 20, 10], "epsg:3857", shape=wh_(200, 100))
    assert isinstance(gbox.svg(), str)
    assert len(gbox.svg()) > 0
    assert gbox.svg(10) != gbox.svg(1)

    assert gbox.grid_lines(mode="pixel").crs == "epsg:3857"

    assert isinstance(gbox._repr_svg_(), str)

    # empty should still work
    assert isinstance((gbox[:3] & gbox[3:])._repr_svg_(), str)


def test_html_repr():
    # smoke test only
    gbox = GeoBox.from_bbox([0, 0, 20, 10], "epsg:3857", shape=wh_(200, 100))
    assert isinstance(gbox._repr_html_(), str)
    assert ">EPSG<" in gbox._repr_html_()
    # empties should still work
    assert isinstance(gbox[:0]._repr_html_(), str)
    assert isinstance(gbox[:0, :0]._repr_html_(), str)

    # non-epsg authority
    gbox = GeoBox.from_bbox((0, 0, 100, 200), esri54019, resolution=1)
    assert ">ESRI<" in gbox._repr_html_()

    # no authority CRS
    assert ">EPSG<" not in gbox.to_crs(modis_crs)._repr_html_()

    # no crs case
    gbox = GeoBox(wh_(200, 100), Affine.translation(-10, 20), None)
    assert gbox.crs is None
    assert isinstance(gbox._repr_html_(), str)
    assert ">EPSG<" not in gbox._repr_html_()

    # empty should still work
    assert isinstance((gbox[:3] & gbox[3:])._repr_html_(), str)

    # empty should still work
    assert isinstance((gbox[:0, :0])._repr_html_(), str)


def test_lrtb():
    gbox = GeoBox.from_bbox([0, 0, 20, 10], "epsg:3857", shape=wh_(200, 100))

    assert gbox.right.left == gbox
    assert gbox.left.right == gbox
    assert gbox.top.bottom == gbox
    assert gbox.bottom.top == gbox

    assert gbox.left == GeoBox.from_bbox([-20, 0, 0, 10], gbox.crs, shape=gbox.shape)
    assert gbox.right == GeoBox.from_bbox([20, 0, 40, 10], gbox.crs, shape=gbox.shape)

    assert gbox.bottom == GeoBox.from_bbox([0, -10, 20, 0], gbox.crs, shape=gbox.shape)
    assert gbox.top == GeoBox.from_bbox([0, 10, 20, 20], gbox.crs, shape=gbox.shape)


def test_compat():
    gbox = GeoBox.from_bbox([0, 0, 20, 10], "epsg:3857", shape=wh_(200, 100))
    gbox_ = gbox.compat
    if gbox_ is None:  # no datacube in this environment
        return
    assert gbox.width == gbox_.width
    assert gbox.height == gbox_.height
    assert gbox.affine == gbox_.affine
    assert gbox.crs == str(gbox.crs)
