import geopandas
import geopandas.datasets

from odc.geo._interop import have
from odc.geo.converters import from_geopandas


def test_from_geopandas():
    df = geopandas.read_file(geopandas.datasets.get_path("naturalearth_lowres"))
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
