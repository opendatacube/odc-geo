# This file is part of the Open Data Cube, see https://opendatacube.org for more information
#
# Copyright (c) 2015-2020 ODC Contributors
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import MagicMock

import pytest
import xarray as xr

import odc.geo
from odc.geo._xr_interop import register_geobox, xr_coords
from odc.geo.geobox import GeoBox
from odc.geo.testutils import epsg3577, mkA, purge_crs_info, xr_zeros


@pytest.fixture
def geobox_epsg4326():
    _box = odc.geo.box(-20, -10, 20, 10, "epsg:4326")
    yield GeoBox.from_geopolygon(_box, 5)


@pytest.fixture
def xx_epsg4326(geobox_epsg4326: GeoBox):
    yield xr_zeros(geobox_epsg4326, dtype="uint16")


def test_geobox_xr_coords():
    A = mkA(0, scale=(10, -10), translation=(-48800, -2983006))

    w, h = 512, 256
    gbox = GeoBox(w, h, A, epsg3577)

    cc = xr_coords(gbox, with_crs=False)
    assert list(cc) == ["y", "x"]
    assert cc["y"].shape == (gbox.shape[0],)
    assert cc["x"].shape == (gbox.shape[1],)
    assert "crs" in cc["y"].attrs
    assert "crs" in cc["x"].attrs

    cc = xr_coords(gbox, with_crs=True)
    assert list(cc) == ["y", "x", "spatial_ref"]
    assert cc["spatial_ref"].shape == ()
    assert cc["spatial_ref"].attrs["spatial_ref"] == gbox.crs.wkt
    assert isinstance(cc["spatial_ref"].attrs["grid_mapping_name"], str)

    # with_crs should be default True
    assert list(xr_coords(gbox)) == list(xr_coords(gbox, with_crs=True))

    cc = xr_coords(gbox, with_crs="Albers")
    assert list(cc) == ["y", "x", "Albers"]

    # geographic CRS
    A = mkA(0, scale=(0.1, -0.1), translation=(10, 30))
    gbox = GeoBox(w, h, A, "epsg:4326")

    cc = xr_coords(gbox, with_crs=True)
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

    # check custom name for crs coordinate
    xx = xr_zeros(gbox, dtype="uint16", with_crs="_crs")
    assert "_crs" in xx.coords
    assert xx.encoding["grid_mapping"] == "_crs"
    assert (xx.values == 0).all()


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
    assert _xx.XX.odc.spatial_dims == None
    assert _xx.XX.odc.crs == None
    assert _xx.XX.odc.transform == None
    assert _xx.XX.odc.geobox == None


def test_assign_crs(xx_epsg4326: xr.DataArray):
    xx = purge_crs_info(xx_epsg4326)
    assert xx.odc.crs is None
    yy = xx.odc.assign_crs("epsg:4326")
    assert xx.odc.uncached.crs is None
    assert yy.odc.crs == "epsg:4326"

    # non-cf complaint CRS
    yy = xx.odc.assign_crs("epsg:3857")
    assert yy.spatial_ref.attrs["grid_mapping_name"] == "??"


def test_corrupt_inputs(xx_epsg4326: xr.DataArray):
    xx = xx_epsg4326.copy()
    assert xx.odc.crs == "epsg:4326"

    # incorrect grid_mapping "pointer"
    xx.encoding["grid_mapping"] = "bad-ref"
    with pytest.warns(UserWarning):
        # it will still get it from attributes
        assert xx.odc.uncached.crs == "epsg:4326"

    # missing wkt text
    xx = xx_epsg4326.copy()
    xx.spatial_ref.attrs.pop("spatial_ref")
    assert xx.odc.crs is None

    # bad CRS string
    xx = xx_epsg4326.copy()
    xx.spatial_ref.attrs["spatial_ref"] = "this is not a CRS!!!"
    assert xx.odc.crs is None

    # duplicated crs coords and no grid_mapping pointer
    xx = xx_epsg4326.copy()
    xx = xx.assign_coords(_crs2=xx.spatial_ref)
    xx.encoding.pop("grid_mapping")
    with pytest.warns(UserWarning):
        xx.odc.crs == "epsg:4326"


def test_geobox_hook(xx_epsg4326: xr.DataArray):
    register_geobox()
    xx = xx_epsg4326
    assert xx.odc.crs == "epsg:4326"
    assert xx.geobox == xx.odc.geobox
    assert xx.to_dataset(name="xx").geobox == xx.odc.geobox

    assert purge_crs_info(xx)[:1, :1].geobox is None
    assert purge_crs_info(xx)[:1, :1].to_dataset(name="xx").geobox is None
