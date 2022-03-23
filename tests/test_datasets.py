from odc.geo.data import data_path, ocean_geojson, ocean_geom


def test_ocean_gjson():
    g1 = ocean_geojson()
    g2 = ocean_geojson()

    assert g1 is g2
    assert len(g1["features"]) == 2


def test_ocean_geom():
    g = ocean_geom()
    assert g.crs == "epsg:4326"


def test_data_path():
    assert data_path().exists()
    assert data_path("--no-such-thing--").exists() is False
