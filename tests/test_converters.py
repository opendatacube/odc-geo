import geopandas
import geopandas.datasets

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
