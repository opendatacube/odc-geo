# pylint: disable=wrong-import-position,redefined-outer-name
from pathlib import Path
from unittest.mock import MagicMock
from warnings import catch_warnings, filterwarnings

import pytest

rasterio = pytest.importorskip("rasterio")
gpd = pytest.importorskip("geopandas")
gpd_datasets = pytest.importorskip("geopandas.datasets")

from odc.geo._interop import have
from odc.geo.converters import extract_gcps, from_geopandas, map_crs, rio_geobox
from odc.geo.gcp import GCPGeoBox
from odc.geo.geobox import GeoBox, GeoBoxBase


@pytest.fixture
def ne_lowres_path():
    with catch_warnings():
        filterwarnings("ignore")
        path = gpd_datasets.get_path("naturalearth_lowres")
    yield path


def test_from_geopandas(ne_lowres_path):
    df = gpd.read_file(ne_lowres_path)
    gg = from_geopandas(df)
    assert isinstance(gg, list)
    assert len(gg) == len(df)
    assert gg[0].crs == "epsg:4326"

    (au,) = from_geopandas(df[df.iso_a3 == "AUS"].to_crs(epsg=3577))
    assert au.crs.epsg == 3577

    (au,) = from_geopandas(df[df.iso_a3 == "AUS"].to_crs(epsg=3857).geometry)
    assert au.crs.epsg == 3857

    assert from_geopandas(df.continent) == []


def test_have():
    assert isinstance(have.geopandas, bool)
    assert isinstance(have.rasterio, bool)
    assert isinstance(have.xarray, bool)
    assert isinstance(have.dask, bool)
    assert isinstance(have.folium, bool)
    assert isinstance(have.ipyleaflet, bool)
    assert isinstance(have.datacube, bool)
    assert isinstance(have.tifffile, bool)
    assert have.rasterio is True
    assert have.xarray is True
    assert have.check_or_error("xarray", "rasterio") is None

    with pytest.raises(RuntimeError):
        have.check_or_error("xarray", "noSuchLibaBCD")


def test_extract_gcps(data_dir: Path):
    with rasterio.open(data_dir / "au-gcp.tif") as src:
        assert src.gcps != ([], None)
        pix1, wld_native = extract_gcps(src)
        pix2, wld_3857 = extract_gcps(src, "epsg:3857")

    assert pix1 == pix2
    assert all(pt.crs == "epsg:4326" for pt in wld_native)
    assert all(pt.crs == "epsg:3857" for pt in wld_3857)

    with rasterio.open(data_dir / "au-3577.tif") as src:
        assert src.gcps == ([], None)
        with pytest.raises(ValueError):
            _ = extract_gcps(src)


@pytest.mark.parametrize("fname", ["au-gcp.tif", "au-3577.tif", "au-3577-rotated.tif"])
def test_rio_geobox(data_dir: Path, fname: str):
    with rasterio.open(data_dir / fname) as rdr:
        pts, _ = rdr.gcps
        gbox = rio_geobox(rdr)
        assert isinstance(gbox, GeoBoxBase)
        assert gbox.width == rdr.width
        assert gbox.height == rdr.height

        if len(pts) > 0:
            assert isinstance(gbox, GCPGeoBox)
        else:
            assert isinstance(gbox, GeoBox)


def test_map_crs():
    proj_3031 = "+proj=stere +lat_0=-90 +lat_ts=-71 +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs +type=crs"
    assert map_crs(MagicMock(crs="EPSG4326")).epsg == 4326
    assert map_crs(MagicMock(crs=dict(name="EPSG3857"))).epsg == 3857
    assert map_crs(MagicMock(crs=dict(name="custom", proj4def=proj_3031))).epsg == 3031
