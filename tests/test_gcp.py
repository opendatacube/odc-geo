# pylint: disable=wrong-import-position, protected-access, redefined-outer-name
from pathlib import Path

import numpy as np
import pytest

from odc.geo import geom
from odc.geo.gcp import GCPGeoBox, GCPMapping
from odc.geo.geobox import GeoBox
from odc.geo.xr import xr_zeros

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

    assert len(gbox.gcps()) == 24

    gbox_ = gbox.to_crs("epsg:3857")
    assert gbox_.shape == gbox.shape
    assert gbox_.crs == "epsg:3857"
    assert gbox_.wld2pix(*gbox_.pix2wld(0, 0)) == pytest.approx((0, 0), abs=1)
    assert gbox_.wld2pix(*gbox_.pix2wld(133, 83)) == pytest.approx((133, 83), abs=1)

    gbox_ = gbox.to_crs("utm")
    assert gbox_.shape == gbox.shape
    assert gbox_.crs is not None
    assert gbox_.crs.proj.utm_zone is not None
    assert gbox_.wld2pix(*gbox_.pix2wld(0, 0)) == pytest.approx((0, 0), abs=1)
    assert gbox_.wld2pix(*gbox_.pix2wld(133, 83)) == pytest.approx((133, 83), abs=1)

    assert gbox.pad(2).to_crs("epsg:6933").crs == "epsg:6933"

    p1, p2 = gbox.map_bounds()
    assert p1 == pytest.approx((-44.50301336231415, 109.39806656168265))
    assert p2 == pytest.approx((-9.47177497427409, 157.04711254391185))

    # map bounds without CRS, should still work, since data is in epsg:4326
    _mapping = GCPMapping(gbox._mapping._pix, gbox._mapping._wld, None)
    assert _mapping.crs is None
    gbox_ = GCPGeoBox(gbox.shape, _mapping)
    assert gbox_.crs is None
    p1, p2 = gbox_.map_bounds()
    assert p1 == pytest.approx((-44.50301336231415, 109.39806656168265))
    assert p2 == pytest.approx((-9.47177497427409, 157.04711254391185))


def test_gcp_geobox_xr(au_gcp_geobox: GCPGeoBox):
    gbox = au_gcp_geobox
    xx = xr_zeros(gbox)
    _gbox = xx.odc.geobox
    assert _gbox.shape == gbox.shape
    assert _gbox.crs == gbox.crs
    assert (_gbox.extent ^ gbox.extent).is_empty

    # check the case when x/y coords are not populated
    yy = xx.drop_vars(xx.odc.spatial_dims)
    assert len(yy.coords) == 1
    _gbox = yy.odc.geobox

    assert _gbox.shape == gbox.shape
    assert _gbox.crs == gbox.crs
    assert (_gbox.extent ^ gbox.extent).is_empty

    # corrupt some gcps
    yy.spatial_ref.attrs["gcps"]["features"][0].pop("properties")
    # should not throw, just return None
    assert yy.odc.uncached.geobox is None


def test_gcp_reproject(au_gcp_geobox: GCPGeoBox):
    # smoke-test only
    gbox = au_gcp_geobox
    xx = xr_zeros(gbox, time=["2020-02-20", "2021-01-21"], dtype="uint8", nodata=255)
    assert xx.ndim == 3
    assert xx.odc.nodata == 255
    assert (xx.odc.geobox.extent ^ gbox.extent).is_empty
    assert isinstance(xx.odc.geobox, GCPGeoBox)

    yy_gbox = xx.odc.geobox.approx.zoom_to(320)
    yy = xx.odc.reproject(yy_gbox)
    assert yy.odc.nodata == 255
    assert yy.odc.geobox == yy_gbox
    assert yy.ndim == 3
    assert yy.shape[0] == xx.shape[0]
    assert set(np.unique(yy.values).tolist()) == set([255, 0])


def test_gcp_mapping():
    gbox0 = GeoBox.from_bbox([0, 0, 20, 10], crs="epsg:4326", resolution=1)
    assert gbox0.shape.wh == (20, 10)
    assert gbox0.crs == "epsg:4326"

    px, py = gbox0.boundary(4).T
    pix = geom.multipoint(
        [(float(x), float(y)) for x, y in zip(px.tolist(), py.tolist())], None
    )
    wld = gbox0.project(pix)

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


@pytest.mark.parametrize("n", [3, 4, 8])
def test_gcp_few_points(n):
    gbox0 = GeoBox.from_bbox([0, 0, 20, 10], crs="epsg:4326", resolution=1)
    if n <= 4:
        corners = gbox0.boundary(2)[:n]
    else:
        corners = gbox0.boundary(n * 3 // 4)[:n:3]

    pix = geom.multipoint([(float(x), float(y)) for x, y in corners], None)
    wld = gbox0.project(pix)

    mapping = GCPMapping(pix, wld)
    gbox = GCPGeoBox(gbox0.shape, mapping)

    wld_ = gbox.project(pix)
    p1 = np.vstack([pt.coords for pt in wld.geoms])
    p2 = np.vstack([pt.coords for pt in wld_.geoms])
    np.testing.assert_array_almost_equal(p1, p2)
