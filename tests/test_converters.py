# pylint: disable=wrong-import-position
from pathlib import Path

import pytest

rasterio = pytest.importorskip("rasterio")
gpd = pytest.importorskip("geopandas")
gpd_datasets = pytest.importorskip("geopandas.datasets")

from odc.geo._interop import have
from odc.geo.converters import extract_gcps, from_geopandas


def test_from_geopandas():
    df = gpd.read_file(gpd_datasets.get_path("naturalearth_lowres"))
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
    assert have.rasterio is True
    assert have.xarray is True


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
