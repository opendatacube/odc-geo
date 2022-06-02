# This file is part of the Open Data Cube, see https://opendatacube.org for more information
#
# Copyright (c) 2015-2020 ODC Contributors
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
import xarray as xr

from odc.geo import geom
from odc.geo._interop import is_dask_collection
from odc.geo.data import ocean_geom
from odc.geo.geobox import GeoBox
from odc.geo.testutils import epsg3577, mkA, purge_crs_info
from odc.geo.xr import (
    ODCExtensionDa,
    rasterize,
    register_geobox,
    wrap_xr,
    xr_coords,
    xr_zeros,
)

# pylint: disable=redefined-outer-name


@pytest.fixture
def geobox_epsg4326():
    _box = geom.box(-20, -10, 20, 10, "epsg:4326")
    yield GeoBox.from_geopolygon(_box, 5)


@pytest.fixture
def xx_epsg4326(geobox_epsg4326: GeoBox):
    yield xr_zeros(geobox_epsg4326, dtype="uint16")


def test_geobox_xr_coords():
    A = mkA(0, scale=(10, -10), translation=(-48800, -2983006))

    w, h = 512, 256
    _shape = h, w
    gbox = GeoBox(_shape, A, epsg3577)

    cc = xr_coords(gbox, crs_coord_name=None)
    assert list(cc) == ["y", "x"]
    assert cc["y"].shape == (gbox.shape[0],)
    assert cc["x"].shape == (gbox.shape[1],)
    assert "crs" in cc["y"].attrs
    assert "crs" in cc["x"].attrs

    cc = xr_coords(gbox)
    assert list(cc) == ["y", "x", "spatial_ref"]
    assert cc["spatial_ref"].shape == ()
    assert cc["spatial_ref"].attrs["spatial_ref"] == gbox.crs.wkt
    assert isinstance(cc["spatial_ref"].attrs["grid_mapping_name"], str)

    # crs_coord_name should be default "spatial_ref"
    assert list(xr_coords(gbox)) == list(xr_coords(gbox, crs_coord_name="spatial_ref"))

    cc = xr_coords(gbox, crs_coord_name="Albers")
    assert list(cc) == ["y", "x", "Albers"]

    # non-axis aligned case
    _gbox = gbox.rotate(33)
    assert _gbox.axis_aligned is False
    cc = xr_coords(_gbox)
    assert list(cc) == ["y", "x", "spatial_ref"]
    assert cc["spatial_ref"].shape == ()
    assert cc["spatial_ref"].attrs["spatial_ref"] == gbox.crs.wkt
    assert isinstance(cc["spatial_ref"].attrs["grid_mapping_name"], str)
    assert cc["x"].encoding["_transform"] == _gbox.affine[:6]
    assert cc["y"].encoding["_transform"] == _gbox.affine[:6]

    # geographic CRS
    A = mkA(0, scale=(0.1, -0.1), translation=(10, 30))
    gbox = GeoBox(_shape, A, "epsg:4326")

    cc = xr_coords(gbox)
    assert list(cc) == ["latitude", "longitude", "spatial_ref"]
    assert cc["spatial_ref"].shape == ()
    assert cc["spatial_ref"].attrs["spatial_ref"] == gbox.crs.wkt
    assert isinstance(cc["spatial_ref"].attrs["grid_mapping_name"], str)


def test_xr_zeros(geobox_epsg4326: GeoBox):
    # missing CRS for GeoBox
    gbox = geobox_epsg4326
    xx = xr_zeros(gbox, dtype="uint16")
    assert "spatial_ref" in xx.coords
    assert xx.encoding["grid_mapping"] == "spatial_ref"
    assert (xx.values == 0).all()
    assert xx.spatial_ref.attrs["spatial_ref"] == gbox.crs.wkt
    assert xx.spatial_ref.attrs["grid_mapping_name"] == "latitude_longitude"
    assert xx.odc.ydim == 0
    assert xx.odc.xdim == 1

    # check custom name for crs coordinate
    xx = xr_zeros(gbox, dtype="uint16", crs_coord_name="_crs")
    assert "_crs" in xx.coords
    assert xx.encoding["grid_mapping"] == "_crs"
    assert (xx.values == 0).all()

    # rotated gebox
    gbox = geobox_epsg4326.center_pixel.rotate(-45).pad_wh(10, 20)
    xx = xr_zeros(gbox, dtype="uint16")
    assert xx.odc.geobox == gbox
    assert xx[3:, 1:].odc.geobox.affine[:6] == pytest.approx(gbox[3:, 1:].affine[:6])
    assert xx[5:, 7:].odc.geobox.affine[:6] == pytest.approx(gbox[5:, 7:].affine[:6])

    # dask version
    gbox = geobox_epsg4326.zoom_to((100, 200))
    xx = xr_zeros(gbox, dtype="uint16", chunks=(10, 20))
    assert is_dask_collection(xx) is True
    assert xx.dtype == "uint16"
    assert xx.data.chunksize == (10, 20)
    assert xx.shape == gbox.shape

    # time axis
    gbox = geobox_epsg4326.zoom_to((100, 200))
    xx = xr_zeros(
        gbox, dtype="uint16", chunks=(1, 10, 20), time=["2020-01-01", "2020-01-02"]
    )
    assert is_dask_collection(xx) is True
    assert xx.dtype == "uint16"
    assert xx.data.chunksize == (1, 10, 20)
    assert xx.shape == (2, *gbox.shape)
    assert xx.odc.ydim == 1
    assert xx.odc.xdim == 2


def test_purge_crs_info(xx_epsg4326: xr.DataArray):
    xx = xx_epsg4326
    assert xx.odc.crs is not None
    assert purge_crs_info(xx).odc.crs is None

    xx = purge_crs_info(xx_epsg4326)
    assert xx.odc.crs is None
    assert xx.encoding == {}


def test_odc_extension(xx_epsg4326: xr.DataArray, geobox_epsg4326: GeoBox):
    gbox = geobox_epsg4326
    xx = xx_epsg4326

    assert "spatial_ref" in xx.coords
    assert xx.encoding["grid_mapping"] == "spatial_ref"
    assert xx.odc.geobox == gbox
    assert (xx.values == 0).all()
    assert xx.odc.crs == "epsg:4326"
    assert xx.odc.transform == gbox.transform
    assert xx.odc.spatial_dims == ("latitude", "longitude")
    assert xx.spatial_ref.attrs["spatial_ref"] == gbox.crs.wkt
    assert xx.spatial_ref.attrs["grid_mapping_name"] == "latitude_longitude"
    assert xx.odc.uncached.transform == xx.odc.transform
    assert xx.odc.output_geobox("epsg:3857").crs == "epsg:3857"
    assert xx.odc.map_bounds() == gbox.map_bounds()

    # this drops encoding/attributes, but crs/geobox should remain the same
    _xx = xx * 10.0
    assert _xx.encoding == {}
    assert _xx.odc.crs == xx.odc.crs
    assert _xx.odc.geobox == xx.odc.geobox

    # test non-standard coordinate names
    _xx = xx.rename(longitude="XX", latitude="YY")
    assert _xx.dims == ("YY", "XX")
    assert _xx.odc.spatial_dims == ("YY", "XX")
    assert _xx.odc.crs == gbox.crs

    # 1-d xarrays should report None for everything
    assert _xx.XX.odc.spatial_dims is None
    assert _xx.XX.odc.crs is None
    assert _xx.XX.odc.transform is None
    assert _xx.XX.odc.geobox is None

    # when geobox is none output_geobox should fail
    with pytest.raises(ValueError):
        _ = _xx.XX.odc.output_geobox("epsg:4326")

    # no geobox - no map bounds
    with pytest.raises(ValueError):
        _ = _xx.XX.odc.map_bounds()

    # no spatial_dims - no ydim/xdim
    with pytest.raises(ValueError):
        _ = _xx.XX.odc.ydim
    with pytest.raises(ValueError):
        _ = _xx.XX.odc.xdim


def test_odc_extension_ds(xx_epsg4326: xr.DataArray, geobox_epsg4326: GeoBox):
    gbox = geobox_epsg4326
    xx = xx_epsg4326.to_dataset(name="band")

    assert "spatial_ref" in xx.coords
    assert xx.band.encoding["grid_mapping"] == "spatial_ref"
    assert xx.odc.geobox == gbox
    assert xx.odc.crs == "epsg:4326"
    assert xx.odc.transform == gbox.transform
    assert xx.odc.spatial_dims == ("latitude", "longitude")
    assert xx.spatial_ref.attrs["spatial_ref"] == gbox.crs.wkt
    assert xx.spatial_ref.attrs["grid_mapping_name"] == "latitude_longitude"
    assert xx.odc.uncached.transform == xx.odc.transform

    # this drops encoding/attributes, but crs/geobox should remain the same
    _xx = xx * 10.0
    assert _xx.band.encoding == {}
    assert _xx.odc.crs == xx.odc.crs
    assert _xx.odc.geobox == xx.odc.geobox

    # test non-standard coordinate names
    _xx = xx.rename(longitude="XX", latitude="YY")
    assert _xx.band.dims == ("YY", "XX")
    assert _xx.odc.spatial_dims == ("YY", "XX")
    assert _xx.odc.crs == gbox.crs

    # 1-d xarrays should report None for everything
    assert _xx.XX.odc.spatial_dims is None
    assert _xx.XX.odc.crs is None
    assert _xx.XX.odc.transform is None
    assert _xx.XX.odc.geobox is None


def test_assign_crs(xx_epsg4326: xr.DataArray):
    xx = purge_crs_info(xx_epsg4326)
    assert xx.odc.crs is None
    yy = xx.odc.assign_crs("epsg:4326")
    assert xx.odc.uncached.crs is None
    assert yy.odc.crs == "epsg:4326"

    # non-cf complaint CRS
    yy = xx.odc.assign_crs("epsg:3857")
    assert yy.spatial_ref.attrs.get("grid_mapping_name") is None
    assert yy.odc.crs == "epsg:3857"


def test_assign_crs_ds(xx_epsg4326: xr.DataArray):
    xx = purge_crs_info(xx_epsg4326).to_dataset(name="band")
    assert xx.odc.crs is None
    yy = xx.odc.assign_crs("epsg:4326")
    assert xx.odc.uncached.crs is None
    assert yy.odc.crs == "epsg:4326"
    assert yy.band.odc.crs == "epsg:4326"
    assert yy.band.encoding["grid_mapping"] == "spatial_ref"

    # non-cf complaint CRS
    yy = xx.odc.assign_crs("epsg:3857")
    assert yy.spatial_ref.attrs.get("grid_mapping_name") is None
    assert yy.odc.crs == "epsg:3857"


def test_corrupt_inputs(xx_epsg4326: xr.DataArray):
    xx = xx_epsg4326.copy()
    assert xx.odc.crs == "epsg:4326"

    # incorrect grid_mapping "pointer"
    xx.encoding["grid_mapping"] = "bad-ref"
    with pytest.warns(UserWarning):
        # it will still get it from attributes
        assert xx.odc.uncached.crs == "epsg:4326"

    # missing spatial_ref, but have crs_wkt
    xx = xx_epsg4326.copy()
    xx.spatial_ref.attrs.pop("spatial_ref")
    assert xx.odc.crs == "epsg:4326"

    # missing wkt text
    xx = xx_epsg4326.copy()
    xx.spatial_ref.attrs.pop("spatial_ref")
    xx.spatial_ref.attrs.pop("crs_wkt")
    assert xx.odc.crs is None

    # bad CRS string
    xx = xx_epsg4326.copy()
    xx.spatial_ref.attrs["spatial_ref"] = "this is not a CRS!!!"
    xx.spatial_ref.attrs["crs_wkt"] = "this is not a CRS!!!"
    assert xx.odc.crs is None

    # duplicated crs coords and no grid_mapping pointer
    xx = xx_epsg4326.copy()
    xx = xx.assign_coords(_crs2=xx.spatial_ref)
    xx.encoding.pop("grid_mapping")
    with pytest.warns(UserWarning):
        assert xx.odc.crs == "epsg:4326"

    # attribute based search with bad CRS string
    xx = purge_crs_info(xx_epsg4326)
    assert xx.odc.crs is None
    xx = xx.assign_attrs(crs="!some invalid CRS string!!")
    with pytest.warns(UserWarning):
        assert xx.odc.crs is None

    # attribute based search with bad CRS type
    xx = purge_crs_info(xx_epsg4326)
    assert xx.odc.crs is None
    xx = xx.assign_attrs(crs=[])
    with pytest.warns(UserWarning):
        assert xx.odc.crs is None

    # attribute based search with several different CRS candidates
    xx = purge_crs_info(xx_epsg4326)
    assert xx.odc.crs is None
    xx = xx.assign_attrs(crs="epsg:4326")
    xx.latitude.attrs["crs"] = epsg3577
    with pytest.warns(UserWarning):
        _crs = xx.odc.crs
        assert _crs is not None
        assert _crs in (epsg3577, "epsg:4326")


def test_geobox_hook(xx_epsg4326: xr.DataArray):
    register_geobox()
    xx = xx_epsg4326
    assert xx.odc.crs == "epsg:4326"
    assert xx.geobox == xx.odc.geobox
    assert xx.to_dataset(name="xx").geobox == xx.odc.geobox

    assert purge_crs_info(xx)[:1, :1].geobox is None
    assert purge_crs_info(xx)[:1, :1].to_dataset(name="xx").geobox is None


def test_wrap_xr():
    gbox = GeoBox.from_bbox([0, -10, 100, 20], "epsg:4326", tight=True, shape=(13, 29))
    data = np.zeros(gbox.shape, dtype="uint16")

    xx = wrap_xr(data, gbox)
    assert xx.shape == gbox.shape
    assert xx.odc.geobox == gbox
    assert xx.dims == gbox.dims
    assert xx.attrs == {}

    xx = wrap_xr(data, gbox, nodata=None)
    assert xx.attrs == {}

    xx = wrap_xr(data, gbox, nodata=10, some_flag=3)
    assert xx.attrs == dict(nodata=10, some_flag=3)

    xx = wrap_xr(data, gbox, time="2022-02-02T22:22:22.222222")
    assert xx.time.dt.year.item() == 2022
    assert xx.time.dt.month.item() == 2

    xx = wrap_xr(data[np.newaxis, ...], gbox, time="2022-02-02T22:22:22.222222")
    assert xx.shape == (1, *gbox.shape)
    assert xx.time.dt.year.values[0] == 2022
    assert xx.time.dt.month.values[0] == 2

    xx = wrap_xr(data[np.newaxis, ...], gbox, time=["2022-02-02T22:22:22.222222"])
    assert xx.shape == (1, *gbox.shape)
    assert xx.time.dt.year.values[0] == 2022
    assert xx.time.dt.month.values[0] == 2


def test_xr_reproject(xx_epsg4326: xr.DataArray):
    assert isinstance(xx_epsg4326.odc, ODCExtensionDa)
    # smoke-test only
    dst_gbox = xx_epsg4326.odc.geobox.zoom_out(1.3)
    xx = xx_epsg4326.odc.reproject(dst_gbox)
    assert xx.odc.geobox == dst_gbox

    # check crs input
    xx = xx_epsg4326.odc.reproject("epsg:3857")
    assert xx.odc.geobox.crs == "epsg:3857"
    assert xx.odc.geobox.shape == xx_epsg4326.odc.output_geobox("epsg:3857").shape

    # non-georegistered case
    with pytest.raises(ValueError):
        _ = xx_epsg4326[:0, :0].odc.reproject(dst_gbox)


def test_xr_rasterize():
    gg = ocean_geom()
    xx = rasterize(gg, 1)
    assert xx.odc.geobox.crs == gg.crs
    yy = rasterize(gg, xx.odc.geobox)
    assert (xx == yy).all()

    yy = rasterize(gg, xx.odc.geobox[3:-4, 4:-8])
    assert (xx[3:-4, 4:-8] == yy).all()

    # chunk of ocean just off Africa in 3857
    xx = rasterize(gg, GeoBox.from_bbox([-20, -10, 20, 10], "epsg:3857", resolution=1))
    assert isinstance(xx.odc, ODCExtensionDa)
    assert xx.odc.geobox is not None
    assert xx.odc.geobox.crs == "epsg:3857"
    assert xx.all().item() is True

    # same but inverse
    xx = rasterize(
        gg,
        GeoBox.from_bbox([-20, -10, 20, 10], "epsg:3857", resolution=1),
        value_inside=False,
    )
    assert xx.odc.geobox.crs == "epsg:3857"
    assert xx.any().item() is False


def test_is_dask_collection():
    import dask

    import odc.geo._interop

    assert "is_dask_collection" in dir(odc.geo._interop)
    assert is_dask_collection is dask.is_dask_collection
