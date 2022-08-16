# pylint: disable=wrong-import-position
from pathlib import Path

import pytest

from odc.geo import geom
from odc.geo.gcp import GCPGeoBox, GCPMapping
from odc.geo.geobox import GeoBox

rasterio = pytest.importorskip("rasterio")


@pytest.fixture()
def au_gcp_rio(data_dir: Path):
    with rasterio.open(data_dir / "au-gcp.tif") as src:
        yield src


@pytest.fixture()
def au_gcp_geobox(au_gcp_rio):
    yield GCPGeoBox.from_rio(au_gcp_rio)


def test_gcp_geobox_from_rio(au_gcp_rio):
    src = au_gcp_rio
    assert src.gcps != ([], None)
    gbox = GCPGeoBox.from_rio(src)
    assert gbox.crs == "epsg:4326"

    assert src.width == gbox.width
    assert src.height == gbox.height


def test_gcp_geobox_basics(au_gcp_geobox: GCPGeoBox):
    gbox = au_gcp_geobox

    assert gbox.linear is False
    assert gbox.axis_aligned is False

    assert gbox.pad(1)[1:-1, 1:-1] == gbox
    assert gbox.pad(1).width == gbox.width + 2
    assert gbox.pad(-1).height == gbox.height - 2

    assert (gbox.extent - gbox.pad(2).extent).is_empty
    assert gbox.geographic_extent.crs == "epsg:4326"

    assert gbox.boundary().shape[1] == 2

    assert gbox.approx.shape == gbox.shape
    assert isinstance(gbox.approx, GeoBox)

    assert gbox.resolution.xy == pytest.approx((0.308896, -0.269107), abs=1e-5)
    assert gbox.resolution == gbox._mapping.resolution

    g2 = gbox.pad_wh(10)
    assert g2.width % 10 == 0 and g2.width >= gbox.width
    assert g2.height % 10 == 0 and g2.height >= gbox.height
    assert g2[0, 0] == gbox[0, 0]
    assert str(gbox) != str(g2)
    assert g2.resolution == gbox.resolution

    assert gbox.center_pixel == gbox[gbox.height // 2, gbox.width // 2]

    assert str(gbox) == repr(gbox)
    assert gbox != "some other thing"

    assert gbox.wld2pix(*gbox.pix2wld(0, 0)) == pytest.approx((0, 0), abs=1)
    assert gbox.wld2pix(*gbox.pix2wld(133, 83)) == pytest.approx((133, 83), abs=1)

    assert gbox.zoom_to(50).shape.wh == (50, 47)

    assert gbox.zoom_out(0.5).zoom_out(2) == gbox

    assert len(set([gbox, gbox.center_pixel, gbox[:1, :2], gbox[:1, :2]])) == 3


def test_gcp_mapping():
    gbox0 = GeoBox.from_bbox([0, 0, 20, 10], crs="epsg:4326", resolution=1)
    assert gbox0.shape.wh == (20, 10)
    assert gbox0.crs == "epsg:4326"

    px, py = gbox0.boundary(4).T
    pix = geom.multipoint([(x, y) for x, y in zip(px.tolist(), py.tolist())], None)

    wx, wy = gbox0.pix2wld(px, py)
    wld = geom.multipoint([(x, y) for x, y in zip(wx.tolist(), wy.tolist())], gbox0.crs)

    mapping = GCPMapping(pix, wld)
    assert mapping.crs == gbox0.crs
    _pix, _wld = mapping.points()
    assert _wld.crs == gbox0.crs
    assert _pix.crs is None

    assert (pix - _pix).is_empty
    assert (wld - _wld).is_empty

    assert mapping.approx[:6] == pytest.approx(gbox0.affine[:6], abs=1e-6)

    for pt in pix.geoms:
        # there and back with minimal loss of precision
        _pt = pt.transform(mapping.p2w).transform(mapping.w2p)
        assert _pt.coords[0] == pytest.approx(pt.coords[0], abs=1e-6)

    for pt in wld.geoms:
        # there and back with minimal loss of precision
        _pt = pt.transform(mapping.w2p).transform(mapping.p2w)
        assert _pt.coords[0] == pytest.approx(pt.coords[0], abs=1e-6)

    _m2 = GCPMapping(mapping._pix, wld)
    assert _m2.crs == mapping.crs
    assert _m2._pix is mapping._pix
