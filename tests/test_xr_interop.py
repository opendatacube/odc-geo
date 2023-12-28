# This file is part of the Open Data Cube, see https://opendatacube.org for more information
#
# Copyright (c) 2015-2020 ODC Contributors
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
import xarray as xr

from odc.geo import geom
from odc.geo._interop import is_dask_collection
from odc.geo._xr_interop import _extract_geo_transform
from odc.geo.data import ocean_geom
from odc.geo.geobox import GeoBox
from odc.geo.testutils import approx_equal_geobox, epsg3577, mkA, purge_crs_info
from odc.geo.types import ROI, resxy_
from odc.geo.xr import (
    ODCExtensionDa,
    ODCExtensionDs,
    rasterize,
    register_geobox,
    wrap_xr,
    xr_coords,
    xr_reproject,
    xr_zeros,
)

# pylint: disable=redefined-outer-name,import-outside-toplevel,protected-access


@pytest.fixture
def geobox_epsg4326():
    _box = geom.box(-20, -10, 20, 10, "epsg:4326")
    yield GeoBox.from_geopolygon(_box, 5)


@pytest.fixture
def xx_chunks():
    yield None


@pytest.fixture
def xx_time():
    yield None


@pytest.fixture
def xx_epsg4326(geobox_epsg4326: GeoBox, xx_time, xx_chunks):
    if xx_time is not None and xx_chunks is not None:
        if len(xx_chunks) < 3:
            xx_chunks = (-1, *xx_chunks)
    yield xr_zeros(geobox_epsg4326, dtype="uint16", chunks=xx_chunks, time=xx_time)


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
    assert isinstance(cc["spatial_ref"].attrs["GeoTransform"], str)

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
    assert isinstance(cc["spatial_ref"].attrs["GeoTransform"], str)
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
    assert gbox.crs is not None
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

    assert gbox.crs is not None
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
    assert xx.odc.output_geobox("utm").crs.epsg is not None

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

    assert gbox.crs is not None
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

    xx = wrap_xr(data[..., np.newaxis], gbox)
    assert xx.shape == (*gbox.shape, 1)
    assert xx.band.data.tolist() == ["b0"]


@pytest.mark.parametrize("xx_time", [None, ["2020-01-30"]])
@pytest.mark.parametrize("xx_chunks", [None, (-1, -1), (4, 4)])
def test_xr_reproject(xx_epsg4326: xr.DataArray):
    assert isinstance(xx_epsg4326.odc, ODCExtensionDa)
    xx0 = xx_epsg4326
    xx0.attrs["crs"] = "epsg:4326"
    # smoke-test only
    assert isinstance(xx_epsg4326.odc.geobox, GeoBox)
    assert xx_epsg4326.odc.nodata is None
    dst_gbox = xx_epsg4326.odc.geobox.zoom_out(1.3)
    xx = xx_epsg4326.odc.reproject(dst_gbox)
    assert xx.odc.geobox == dst_gbox
    assert xx.encoding["grid_mapping"] == "spatial_ref"
    assert "crs" not in xx.attrs

    yy = xr.Dataset({"a": xx0, "b": xx0 + 1, "c": xr.DataArray([2, 3, 4])})
    assert isinstance(yy.odc, ODCExtensionDs)
    assert yy.odc.geobox == xx0.odc.geobox
    yy_ = yy.odc.reproject(dst_gbox)
    assert isinstance(yy_.odc, ODCExtensionDs)
    assert yy_.odc.geobox == dst_gbox
    assert yy_.a.odc.geobox == dst_gbox
    assert yy_.b.odc.geobox == dst_gbox
    assert (yy_.c == yy.c).all()

    yy_ = yy.odc.reproject("utm")
    assert yy_.odc.geobox.crs.proj.utm_zone is not None

    yy_ = xr_reproject(yy, "utm-n")
    assert isinstance(yy_, xr.Dataset)
    assert yy_.odc.geobox.crs.proj.utm_zone is not None
    assert yy_.odc.geobox.crs.proj.utm_zone.endswith("N")

    xx = xr_reproject(xx0, "utm-s")
    assert isinstance(xx, xr.DataArray)
    assert xx.odc.geobox.crs.proj.utm_zone is not None
    assert xx.odc.geobox.crs.proj.utm_zone.endswith("S")

    # check crs input
    xx = xx_epsg4326.odc.reproject("epsg:3857")
    assert xx.odc.geobox.crs == "epsg:3857"
    assert xx.odc.geobox.shape == xx_epsg4326.odc.output_geobox("epsg:3857").shape
    assert xx.odc.nodata is None

    # check dst_nodata override
    xx = xx_epsg4326.odc.reproject("epsg:3857", dst_nodata=255)
    assert xx.odc.nodata == 255

    # multi-time should work just the same
    if "time" in xx_epsg4326.dims:
        assert (xx_epsg4326.time == xx.time).all()
    else:
        xx2 = xx_epsg4326.expand_dims(time=2).odc.reproject("epsg:3857", dst_nodata=255)
        assert xx2.shape[0] == 2
        np.testing.assert_array_equal(xx2[0], xx)
        np.testing.assert_array_equal(xx2[1], xx)

    xx_i8 = xx_epsg4326.astype("int8")
    assert xx_i8.odc.reproject("epsg:3857").dtype == "int8"

    # non-georegistered case
    with pytest.raises(ValueError):
        _ = purge_crs_info(xx_epsg4326).odc.reproject(dst_gbox)

    with pytest.raises(ValueError):
        _ = xr.Dataset().odc.reproject("utm")


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


TEST_GEOBOXES_SMALL_AXIS_ALIGNED = [
    GeoBox.from_bbox((-10, -2, 5, 4), "epsg:4326", tight=True, resolution=0.2),
    GeoBox.from_bbox((-10, -2, 5, 4), "epsg:3857", tight=True, resolution=1),
    GeoBox.from_bbox((-10, -2, 5, 4), "epsg:3857", tight=True, resolution=resxy_(1, 2)),
]


@pytest.mark.parametrize("geobox", TEST_GEOBOXES_SMALL_AXIS_ALIGNED)
@pytest.mark.parametrize(
    "bad_geo_transform",
    [
        "some random string",
        "1 2 3 4 5 not-a-float",
    ],
)
def test_extract_transform(geobox, bad_geo_transform: str):
    xx = xr_zeros(geobox, dtype="int16")
    assert xx.odc.geobox == geobox
    assert _extract_geo_transform(xx.spatial_ref) == geobox.affine

    bad_attr_coord = xx.spatial_ref.copy()
    bad_attr_coord.attrs["GeoTransform"] = bad_geo_transform
    assert _extract_geo_transform(bad_attr_coord) is None


@pytest.mark.parametrize("geobox", TEST_GEOBOXES_SMALL_AXIS_ALIGNED)
@pytest.mark.parametrize(
    "roi",
    [
        np.s_[:1, :1],
        np.s_[-1:, -1:],
        np.s_[1:2, 3:4],
        np.s_[1:2, :5],
        np.s_[1:, 3:4],
    ],
)
def test_geobox_1px(geobox: GeoBox, roi: ROI):
    assert geobox.axis_aligned
    assert min(geobox.shape) > 1
    assert max(geobox.shape) < 1000

    xx = xr_zeros(geobox, "uint8")
    assert isinstance(xx.odc, ODCExtensionDa)
    assert xx.odc.geobox == geobox

    assert approx_equal_geobox(xx[roi].odc.geobox, geobox[roi])

    # Verify that missing GeoTransform is handled without exceptions
    yy = xx.copy()
    yy.spatial_ref.attrs.pop("GeoTransform")
    assert min(yy[roi].shape[-2:]) == 1
    assert yy[roi].odc.geobox is None


@pytest.mark.parametrize("geobox", TEST_GEOBOXES_SMALL_AXIS_ALIGNED)
@pytest.mark.parametrize(
    "roi",
    [
        np.s_[1:1, :1],
        np.s_[-1:, -1:1],
        np.s_[1:1, 1:1],
    ],
)
def test_geobox_0px(geobox: GeoBox, roi: ROI):
    assert geobox.axis_aligned
    assert min(geobox.shape) > 1
    assert max(geobox.shape) < 1000

    xx = xr_zeros(geobox, "uint8")
    assert isinstance(xx.odc, ODCExtensionDa)
    assert xx.odc.geobox == geobox

    assert xx[roi].odc.geobox is None


@pytest.mark.parametrize(
    "poly, expected_fail",
    [
        (
            geom.polygon(
                ((-8, 8), (6, 1), (-4, -6)),
                crs="EPSG:4326",
            ),
            False,  # Fully inside, matching CRS
        ),
        (
            geom.polygon(
                ((-24, 8), (-10, 1), (-20, -6)),
                crs="EPSG:4326",
            ),
            False,  # Overlapping, matching CRS
        ),
        (
            geom.polygon(
                ((-40, 8), (-26, 1), (-36, -6)),
                crs="EPSG:4326",
            ),
            True,  # Fully outside, matching CRS
        ),
        (
            geom.polygon(
                ((-890555, 893463), (667916, 111325), (-445277, -669141)),
                crs="EPSG:3857",
            ),
            False,  # Fully inside, different CRS
        ),
        (
            geom.polygon(
                ((-2671667, 893463), (-1113194, 111325), (-2226389, -669141)),
                crs="EPSG:3857",
            ),
            False,  # Overlapping, different CRS
        ),
        (
            geom.polygon(
                ((-4452779, 893463), (-2894306, 111325), (-4007501, -669141)),
                crs="EPSG:3857",
            ),
            True,  # Fully outside, different CRS
        ),
    ],
)
def test_crop(xx_epsg4326, poly, expected_fail):
    xx = xx_epsg4326

    # If fail is expected, pass test
    if expected_fail:
        with pytest.raises(ValueError):
            xx_cropped = xx.odc.crop(poly=poly)
        return

    # Crop with default settings
    xx_cropped = xx.odc.crop(poly=poly)

    # Verify that cropped data is smaller
    assert xx_cropped.size <= xx.size

    # Verify that resolution and alignment have not changed
    np.testing.assert_array_almost_equal(
        xx.odc.geobox.resolution.xy, xx_cropped.odc.geobox.resolution.xy
    )
    np.testing.assert_array_almost_equal(
        xx.odc.geobox.alignment.xy, xx_cropped.odc.geobox.alignment.xy
    )

    # Verify that data contains NaN from default masking step
    assert xx_cropped.isnull().any()

    # Verify that no NaNs exist if masking not applied
    xx_nomask = xx.odc.crop(poly=poly, apply_mask=False)
    assert xx_nomask.notnull().all()

    # Verify that cropping also works on datasets
    xx_ds = xx.to_dataset(name="test")
    xx_ds_cropped = xx_ds.odc.crop(poly=poly)
    assert xx_ds_cropped.test.size <= xx_ds.test.size
    np.testing.assert_array_almost_equal(
        xx_ds.odc.geobox.resolution.xy, xx_ds_cropped.odc.geobox.resolution.xy
    )
    np.testing.assert_array_almost_equal(
        xx_ds.odc.geobox.alignment.xy, xx_ds_cropped.odc.geobox.alignment.xy
    )


@pytest.mark.parametrize(
    "poly",
    [
        geom.polygon(((-8, 8), (6, 1), (-4, -6)), crs="EPSG:4326"),
        geom.polygon(
            ((-890555, 893463), (667916, 111325), (-445277, -669141)), crs="EPSG:3857"
        ),
    ],
)
def test_mask(xx_epsg4326, poly):
    # Create test data and replace values with random integers so we can
    # reliably test that pixel values are the same before and after masking
    xx = xx_epsg4326
    xx.values[:] = np.random.randint(0, 10, size=xx.shape)

    # Apply mask
    xx_masked = xx.odc.mask(poly)

    # Verify that geobox is the same
    assert xx_masked.odc.geobox == xx.odc.geobox

    # Verify that data contains NaN from default masking step
    assert xx_masked.isnull().any()

    # Verify that non-masked values are the same as in non-masked dataset
    masked_pixels = np.isfinite(xx_masked.data)
    np.testing.assert_array_almost_equal(
        xx_masked.data[masked_pixels], xx.data[masked_pixels]
    )

    # Verify that `all_touched=False` produces fewer unmasked pixels
    xx_notouched = xx.odc.mask(poly, all_touched=False)
    assert xx_notouched.notnull().sum() < xx_masked.notnull().sum()

    # Verify that masking also works on datasets
    xx_ds = xx.to_dataset(name="test")
    xx_ds_masked = xx_ds.odc.mask(poly=poly)
    assert xx_ds_masked.odc.geobox == xx_ds.odc.geobox
    assert xx_ds_masked.test.isnull().any()
