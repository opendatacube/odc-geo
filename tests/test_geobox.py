# This file is part of the Open Data Cube, see https://opendatacube.org for more information
#
# Copyright (c) 2015-2020 ODC Contributors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from affine import Affine

from odc.geo import CRS, geom, ixy_, resyx_, wh_
from odc.geo.geobox import (
    GeoBox,
    bounding_box_in_pixel_domain,
    gbox_boundary,
    geobox_intersection_conservative,
    geobox_union_conservative,
    scaled_down_geobox,
)
from odc.geo.math import apply_affine
from odc.geo.testutils import epsg3577, epsg3857, epsg4326, mkA, xy_from_gbox, xy_norm

# pylint: disable=pointless-statement,too-many-statements


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

    assert gbox.shape == (h, w)
    assert gbox.transform == A
    assert gbox.extent.crs == gbox.crs
    assert gbox.geographic_extent.crs == epsg4326
    assert gbox.extent.boundingbox.height == h * 10.0
    assert gbox.extent.boundingbox.width == w * 10.0
    assert gbox.alignment.yx == (4, 0)  # 4 because -2983006 % 10 is 4
    assert isinstance(str(gbox), str)
    assert "EPSG:3577" in repr(gbox)

    assert GeoBox((1, 1), mkA(0), epsg4326).geographic_extent.crs == epsg4326
    assert GeoBox((1, 1), mkA(0), None).dimensions == ("y", "x")

    g2 = gbox[:-10, :-20]
    assert g2.shape == (gbox.height - 10, gbox.width - 20)

    # step of 1 is ok
    g2 = gbox[::1, ::1]
    assert g2.shape == gbox.shape

    assert gbox[0].shape == (1, gbox.width)
    assert gbox[:3].shape == (3, gbox.width)

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
    assert (gbox[:3, :4] & gbox[30:, 40:]).is_empty()

    with pytest.raises(ValueError):
        geobox_intersection_conservative([])

    with pytest.raises(ValueError):
        geobox_union_conservative([])

    # can not combine across CRSs
    with pytest.raises(ValueError):
        bounding_box_in_pixel_domain(
            GeoBox((1, 1), mkA(0), epsg4326), GeoBox(ixy_(2, 3), mkA(0), epsg3577)
        )


def test_gbox_boundary():
    xx = np.zeros((2, 6))

    bb = gbox_boundary(xx, 3)

    assert bb.shape == (4 + (3 - 2) * 4, 2)
    assert set(bb.T[0]) == {0.0, 3.0, 6.0}
    assert set(bb.T[1]) == {0.0, 1.0, 2.0}


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
