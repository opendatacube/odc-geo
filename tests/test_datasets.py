from odc.geo.data import country_geom, data_path, gbox_css, ocean_geojson, ocean_geom


def test_ocean_gjson():
    g1 = ocean_geojson()
    g2 = ocean_geojson()

    assert g1 is g2
    assert len(g1["features"]) == 2


def test_ocean_geom():
    g = ocean_geom()
    assert g.crs == "epsg:4326"

    g = ocean_geom("epsg:3857", (-180, -80, 180, 80))
    assert g.crs == "epsg:3857"


def test_country_geom():
    g = country_geom("AUS")
    assert g.crs == "epsg:4326"

    g = country_geom("AUS", "epsg:3577")
    assert g.crs == "epsg:3577"


def test_gbox_css():
    assert isinstance(gbox_css(), str)


def test_data_path():
    assert data_path().exists()
    assert data_path("--no-such-thing--").exists() is False
