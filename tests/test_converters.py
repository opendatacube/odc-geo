# pylint: disable=wrong-import-position,redefined-outer-name
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock
from warnings import catch_warnings, filterwarnings

import pytest

rasterio = pytest.importorskip("rasterio")
gpd = pytest.importorskip("geopandas")
gpd_datasets = pytest.importorskip("geopandas.datasets")

from odc.geo._interop import have
from odc.geo.converters import (
    GEOTIFF_TAGS,
    extract_gcps,
    from_geopandas,
    geotiff_metadata,
    map_crs,
    rio_geobox,
)
from odc.geo.gcp import GCPGeoBox
from odc.geo.geobox import GeoBox, GeoBoxBase

_gbox = GeoBox.from_bbox((-10, -20, 15, 30), 4326, resolution=1)


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


@pytest.mark.parametrize(
    "gbox",
    [
        _gbox,
        _gbox.to_crs(3857),
        _gbox.to_crs("ESRI:53010"),
        _gbox.rotate(10),
        _gbox.center_pixel.pad(3),
    ],
)
@pytest.mark.parametrize("nodata", [None, float("nan"), 0, -999])
@pytest.mark.parametrize("gdal_metadata", [None, "<GDALMetadata></GDALMetadata>"])
def test_geotiff_metadata(gbox: GeoBox, nodata, gdal_metadata: Optional[str]):
    assert gbox.crs is not None

    geo_tags, md = geotiff_metadata(gbox, nodata=nodata, gdal_metadata=gdal_metadata)
    assert isinstance(md, dict)
    assert isinstance(geo_tags, list)
    assert len(geo_tags) >= 2
    tag_codes = set([code for code, *_ in geo_tags])

    if nodata is not None:
        assert 42113 in tag_codes
    if gdal_metadata is not None:
        assert 42112 in tag_codes

    for code, dtype, count, val in geo_tags:
        assert code in GEOTIFF_TAGS
        assert isinstance(dtype, int)
        assert isinstance(count, int)
        if count > 0:
            assert isinstance(val, (tuple, str))
            if isinstance(val, str):
                assert len(val) + 1 == count
            else:
                assert len(val) == count

    if gbox.axis_aligned:
        assert "ModelPixelScale" in md
    else:
        assert "ModelTransformation" in md

    if gbox.crs.epsg is not None:
        if gbox.crs.projected:
            assert md["GTModelTypeGeoKey"] == 1
            assert md["ProjectedCSTypeGeoKey"] == gbox.crs.epsg
        else:
            assert md["GTModelTypeGeoKey"] == 2
            assert md["GeographicTypeGeoKey"] == gbox.crs.epsg
    else:
        assert md["GTModelTypeGeoKey"] == 32767
