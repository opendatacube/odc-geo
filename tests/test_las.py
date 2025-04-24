from types import SimpleNamespace

import pytest

from odc.geo.io import load_las

pytest.importorskip("laspy")
pytest.importorskip("lazrs")

test_crss = {
    "autzen": """
COMPOUNDCRS["NAD83 / Oregon GIC Lambert (ft) + NAVD88 height (ftUS)",
    PROJCRS["NAD83 / Oregon GIC Lambert (ft)",
        BASEGEOGCRS["NAD83",
            DATUM["North American Datum 1983",
                ELLIPSOID["GRS 1980",6378137,298.257222101,
                    LENGTHUNIT["metre",1]]],
            PRIMEM["Greenwich",0,
                ANGLEUNIT["degree",0.0174532925199433]],
            ID["EPSG",4269]],
        CONVERSION["unnamed",
            METHOD["Lambert Conic Conformal (2SP)",
                ID["EPSG",9802]],
            PARAMETER["Latitude of false origin",41.75,
                ANGLEUNIT["degree",0.0174532925199433],
                ID["EPSG",8821]],
            PARAMETER["Longitude of false origin",-120.5,
                ANGLEUNIT["degree",0.0174532925199433],
                ID["EPSG",8822]],
            PARAMETER["Latitude of 1st standard parallel",43,
                ANGLEUNIT["degree",0.0174532925199433],
                ID["EPSG",8823]],
            PARAMETER["Latitude of 2nd standard parallel",45.5,
                ANGLEUNIT["degree",0.0174532925199433],
                ID["EPSG",8824]],
            PARAMETER["Easting at false origin",1312335.958,
                LENGTHUNIT["foot",0.3048],
                ID["EPSG",8826]],
            PARAMETER["Northing at false origin",0,
                LENGTHUNIT["foot",0.3048],
                ID["EPSG",8827]]],
        CS[Cartesian,2],
            AXIS["easting",east,
                ORDER[1],
                LENGTHUNIT["foot",0.3048]],
            AXIS["northing",north,
                ORDER[2],
                LENGTHUNIT["foot",0.3048]],
        ID["EPSG",2992]],
    VERTCRS["NAVD88 height (ftUS)",
        VDATUM["North American Vertical Datum 1988"],
        CS[vertical,1],
            AXIS["gravity-related height",up,
                LENGTHUNIT["US survey foot",0.304800609601219]],
        ID["EPSG",6360]]]
"""
}

AUTZEN_DATA_VARS = (
    "user_data",
    "withheld",
    "scan_direction_flag",
    "red",
    "green",
    "blue",
    "classification",
    "intensity",
    "return_number",
    "scanner_channel",
    "scan_angle",
    "edge_of_flight_line",
    "synthetic",
    "number_of_returns",
    "key_point",
    "overlap",
    "point_source_id",
)


@pytest.fixture(params=["autzen-tiny.copc.laz", "no-crs.las"])
def las_test_data(data_dir, request):
    src = request.param
    crs_key = src.split("-")[0]
    return SimpleNamespace(
        src=str(data_dir / src),
        expected_bands=AUTZEN_DATA_VARS,
        expected_crs=test_crss.get(crs_key, None),
    )


def test_las_load(las_test_data):
    src = las_test_data.src
    expected_crs = test_crss.get(las_test_data.expected_crs, las_test_data.expected_crs)
    xx = load_las(src)

    assert "x" in xx.coords
    assert "y" in xx.coords
    assert "z" in xx.coords
    assert "time" in xx.coords

    assert set(xx.data_vars) == set(las_test_data.expected_bands)

    assert xx.odc.crs == expected_crs

    yy = load_las(src, channels=["red", "green", "blue"])
    assert set(yy.data_vars) == {"red", "green", "blue"}

    assert yy.odc.crs == xx.odc.crs
    assert set(yy.coords) == set(xx.coords)
    assert len(yy.time) == len(xx.time)

    if ".copc." in src:
        xx0 = load_las(src, driver="copc", level=0)
        assert xx0.odc.crs == xx.odc.crs
        assert len(xx0.time) <= len(xx.time)
    else:
        assert load_las(src, driver="laspy").odc.crs == xx.odc.crs


@pytest.mark.parametrize("force_crs", ["EPSG:4326", "EPSG:3857"])
def test_force_crs(las_test_data, force_crs):
    src = las_test_data.src

    xx = load_las(src)
    assert xx.odc.crs != force_crs

    yy = load_las(src, force_crs=force_crs)
    assert yy.odc.crs == force_crs
