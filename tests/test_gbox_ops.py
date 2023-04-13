# This file is part of the Open Data Cube, see https://opendatacube.org for more information
#
# Copyright (c) 2015-2020 ODC Contributors
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
from affine import Affine

from odc.geo import geobox as gbx
from odc.geo import wh_
from odc.geo.geobox import GeoBox
from odc.geo.testutils import epsg3857

# pylint: disable=pointless-statement,too-many-statements


def test_gbox_ops():
    s = GeoBox(wh_(1000, 100), Affine(10, 0, 12340, 0, -10, 316770), epsg3857)
    assert s.shape == (100, 1000)

    d = gbx.flipy(s)
    assert d.shape == s.shape
    assert d.crs is s.crs
    assert d.resolution.yx == (-s.resolution.y, s.resolution.x)
    assert d.extent.contains(s.extent)
    with pytest.raises(ValueError):
        # flipped grid
        (s | d)
    with pytest.raises(ValueError):
        # flipped grid
        (s & d)

    d = gbx.flipx(s)
    assert d.shape == s.shape
    assert d.crs is s.crs
    assert d.resolution.yx == (s.resolution.y, -s.resolution.x)
    assert d.extent.contains(s.extent)

    assert gbx.flipy(gbx.flipy(s)).affine == s.affine
    assert gbx.flipx(gbx.flipx(s)).affine == s.affine

    d = gbx.zoom_out(s, 2)
    assert d.shape == (50, 500)
    assert d.crs is s.crs
    assert d.extent.contains(s.extent)
    assert d.resolution.yx == (s.resolution.y * 2, s.resolution.x * 2)

    d = gbx.zoom_out(s, 2 * max(s.shape))
    assert d.shape == (1, 1)
    assert d.crs is s.crs
    assert d.extent.contains(s.extent)

    d = gbx.zoom_out(s, 1.33719)
    assert d.crs is s.crs
    assert d.extent.contains(s.extent)
    assert all(ds < ss for ds, ss in zip(d.shape, s.shape))
    with pytest.raises(ValueError):
        # lower resolution grid
        (s | d)
    with pytest.raises(ValueError):
        # lower resolution grid
        (s & d)

    d = gbx.zoom_to(s, s.shape)
    assert d == s

    d = gbx.zoom_to(s, (1, 3))
    assert d.shape == (1, 3)
    assert d.extent == s.extent

    d = gbx.zoom_to(s, (10000, 10000))
    assert d.shape == (10000, 10000)
    assert d.extent == s.extent

    d = gbx.pad(s, 1)
    assert d.crs is s.crs
    assert d.resolution == s.resolution
    assert d.extent.contains(s.extent)
    assert s.extent.contains(d.extent) is False
    assert d[1:-1, 1:-1].affine == s.affine
    assert d[1:-1, 1:-1].shape == s.shape
    assert d == (s | d)
    assert s == (s & d)

    d = gbx.pad_wh(s, 10)
    assert d == s

    d = gbx.pad_wh(s, 100, 8)
    assert d.width == s.width
    assert d.height % 8 == 0
    assert 0 < d.height - s.height < 8
    assert d.affine == s.affine
    assert d.crs is s.crs

    d = gbx.pad_wh(s, 13, 17)
    assert d.affine == s.affine
    assert d.crs is s.crs
    assert d.height % 17 == 0
    assert d.width % 13 == 0
    assert 0 < d.height - s.height < 17
    assert 0 < d.width - s.width < 13

    d = gbx.translate_pix(s, 1, 2)
    assert d.crs is s.crs
    assert d.resolution == s.resolution
    assert d.extent != s.extent
    assert s[2:3, 1:2].extent == d[:1, :1].extent

    d = gbx.translate_pix(s, -10, -2)
    assert d.crs is s.crs
    assert d.resolution == s.resolution
    assert d.extent != s.extent
    assert s[:1, :1].extent == d[2:3, 10:11].extent

    d = gbx.translate_pix(s, 0.1, 0)
    assert d.crs is s.crs
    assert d.shape == s.shape
    assert d.resolution == s.resolution
    assert d.extent != s.extent
    assert d.extent.contains(s[:, 1:].extent)

    d = gbx.translate_pix(s, 0, -0.5)
    assert d.crs is s.crs
    assert d.shape == s.shape
    assert d.resolution == s.resolution
    assert d.extent != s.extent
    assert s.extent.contains(d[1:, :].extent)

    d = gbx.affine_transform_pix(s, Affine(1, 0, 0, 0, 1, 0))
    assert d.crs is s.crs
    assert d.shape == s.shape
    assert d.resolution == s.resolution
    assert d.extent == s.extent

    d = gbx.rotate(s, 180)
    assert d.crs is s.crs
    assert d.shape == s.shape
    assert d.extent != s.extent
    np.testing.assert_almost_equal(d.extent.area, s.extent.area, 5)
    assert s[49:52, 499:502].extent.contains(
        d[50:51, 500:501].extent
    ), "Check that center pixel hasn't moved"
