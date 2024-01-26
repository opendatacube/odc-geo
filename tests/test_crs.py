# This file is part of the Open Data Cube, see https://opendatacube.org for more information
#
# Copyright (c) 2015-2020 ODC Contributors
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import rasterio.crs
from pytest import approx

from odc.geo import geom
from odc.geo.crs import (
    _CRS,
    CRS,
    CRSError,
    CRSMismatchError,
    crs_units_per_degree,
    norm_crs,
    norm_crs_or_error,
)
from odc.geo.geom import common_crs
from odc.geo.testutils import epsg3577, epsg3857, epsg4326
from odc.geo.types import Unset, xy_

# pylint: disable=missing-class-docstring,use-implicit-booleaness-not-comparison
# pylint: disable=comparison-with-itself


def test_common_crs():
    assert common_crs([]) is None
    assert (
        common_crs([geom.point(0, 0, epsg4326), geom.line([(0, 0), (1, 1)], epsg4326)])
        is epsg4326
    )

    with pytest.raises(CRSMismatchError):
        common_crs([geom.point(0, 0, epsg4326), geom.line([(0, 0), (1, 1)], epsg3857)])


class TestCRSEqualityComparisons:
    def test_comparison_edge_cases(self):
        a = epsg4326
        none_crs = None
        assert a == a
        assert a == str(a)
        assert (a == none_crs) is False
        assert (a == []) is False
        assert (a == TestCRSEqualityComparisons) is False

    def test_australian_albers_comparison(self):
        a = CRS(
            """PROJCS["GDA94_Australian_Albers",GEOGCS["GCS_GDA_1994",
                            DATUM["Geocentric_Datum_of_Australia_1994",SPHEROID["GRS_1980",6378137,298.257222101]],
                            PRIMEM["Greenwich",0],UNIT["Degree",0.017453292519943295]],
                            PROJECTION["Albers_Conic_Equal_Area"],
                            PARAMETER["standard_parallel_1",-18],
                            PARAMETER["standard_parallel_2",-36],
                            PARAMETER["latitude_of_center",0],
                            PARAMETER["longitude_of_center",132],
                            PARAMETER["false_easting",0],
                            PARAMETER["false_northing",0],
                            UNIT["Meter",1]]"""
        )
        b = epsg3577

        assert a == b

        assert a != epsg4326


def test_no_epsg():
    c = CRS("+proj=longlat +no_defs +ellps=GRS80")
    b = CRS(
        """GEOGCS["GRS 1980(IUGG, 1980)",DATUM["unknown",SPHEROID["GRS80",6378137,298.257222101]],
                        PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]"""
    )

    assert c.epsg is None
    assert b.epsg is None
    assert c.authority == ("", "")


def test_crs():
    custom_crs = CRS(
        """PROJCS["unnamed",
                           GEOGCS["Unknown datum based upon the custom spheroid",
                           DATUM["Not specified (based on custom spheroid)", SPHEROID["Custom spheroid",6371007.181,0]],
                           PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]],
                           PROJECTION["Sinusoidal"],
                           PARAMETER["longitude_of_center",0],
                           PARAMETER["false_easting",0],
                           PARAMETER["false_northing",0],
                           UNIT["Meter",1]]"""
    )

    crs = epsg3577
    assert crs.geographic is False
    assert crs.projected is True
    assert crs.dimensions == ("y", "x")
    assert crs.epsg == 3577
    assert crs.units == ("metre", "metre")
    assert isinstance(repr(crs), str)

    crs = epsg4326
    assert crs.geographic is True
    assert crs.projected is False
    assert crs.dimensions == ("latitude", "longitude")
    assert crs.epsg == 4326
    assert epsg4326.semi_major_axis > 6_350_000
    assert epsg4326.semi_minor_axis > 6_350_000
    assert epsg4326.inverse_flattening == pytest.approx(298.257223563)

    crs2 = CRS(crs)
    assert crs2 == crs
    assert crs.proj is crs2.proj

    assert epsg4326.valid_region == geom.box(-180, -90, 180, 90, epsg4326)
    assert epsg3857.valid_region.crs == epsg4326
    xmin, _, xmax, _ = epsg3857.valid_region.boundingbox
    assert (xmin, xmax) == (-180, 180)
    assert custom_crs.valid_region is None

    assert epsg3577 == epsg3577
    assert epsg3577 == "EPSG:3577"
    assert (epsg3577 != epsg3577) is False
    assert (epsg3577 == epsg4326) is False
    assert (epsg3577 == "EPSG:4326") is False
    assert epsg3577 != epsg4326
    assert epsg3577 != "EPSG:4326"

    bad_crs = [
        "cupcakes",
        (
            'PROJCS["unnamed",'
            'GEOGCS["WGS 84", DATUM["WGS_1984", SPHEROID["WGS 84",6378137,298.257223563, AUTHORITY["EPSG","7030"]],'
            'AUTHORITY["EPSG","6326"]], PRIMEM["Greenwich",0, AUTHORITY["EPSG","8901"]],'
            'UNIT["degree",0.0174532925199433, AUTHORITY["EPSG","9122"]], AUTHORITY["EPSG","4326"]]]'
        ),
    ]

    for bad in bad_crs:
        with pytest.raises(CRSError):
            CRS(bad)

    with pytest.warns(DeprecationWarning):
        assert str(epsg3857) == epsg3857.crs_str


def test_crs_compat():
    crs = CRS("epsg:3577")
    assert crs.epsg == 3577
    assert crs.authority == ("EPSG", 3577)
    assert str(crs) == "EPSG:3577"
    crs2 = CRS(crs)
    assert crs.epsg == crs2.epsg

    assert crs == CRS(crs.proj)
    assert crs == CRS(crs)
    assert crs == CRS(f"epsg:{crs.to_epsg()}")
    assert crs == CRS(crs.to_wkt())

    assert str(CRS(crs.to_wkt())) == "EPSG:3577"

    crs_rio = rasterio.crs.CRS(init="epsg:3577")
    assert CRS(crs_rio).epsg == 3577

    assert (CRS(crs_rio) == crs_rio) is True
    assert CRS(_CRS(crs_rio)) == CRS(crs_rio)
    assert CRS(str(crs_rio)) == crs_rio
    assert CRS(crs_rio) == 3577

    with pytest.raises(CRSError):
        CRS(("random", "tuple"))

    crs = CRS("epsg:3857")
    with pytest.warns(UserWarning):
        crs_dict = crs.proj.to_dict()

    assert CRS(crs_dict) == crs

    wkt_crs = SimpleNamespace(to_wkt=Mock(return_value=crs.to_wkt()))
    assert getattr(wkt_crs, "to_epsg", None) is None
    assert CRS(wkt_crs) == crs

    assert CRS("ESRI:54019").authority == ("ESRI", 54019)


@pytest.mark.parametrize("epsg", [4326, 3577, 3587])
def test_crs_int(epsg):
    assert CRS(epsg) == CRS(f"EPSG:{epsg}")
    assert CRS(epsg).epsg == epsg

    crs1 = CRS(epsg)
    crs2 = CRS(str(crs1))
    crs3 = CRS(f"epsg:{epsg}")
    assert crs1.proj is crs2.proj
    assert crs1.proj is crs3.proj


def test_crs_hash():
    crs = CRS("epsg:3577")
    crs2 = CRS(crs)

    assert crs is not crs2
    assert len({crs, crs2}) == 1


def test_crs_units_per_degree():
    assert crs_units_per_degree("EPSG:3857", (0, 0)) == crs_units_per_degree(
        "EPSG:3857", 0, 0
    )
    assert crs_units_per_degree("EPSG:4326", (120, -10)) == approx(1.0, 1e-6)

    assert crs_units_per_degree("EPSG:3857", 0, 0) == approx(111319.49, 0.5)
    assert crs_units_per_degree("EPSG:3857", 20, 0) == approx(111319.49, 0.5)
    assert crs_units_per_degree("EPSG:3857", 30, 0) == approx(111319.49, 0.5)
    assert crs_units_per_degree("EPSG:3857", 180, 0) == approx(111319.49, 0.5)
    assert crs_units_per_degree("EPSG:3857", -180, 0) == approx(111319.49, 0.5)


def test_rio_crs__no_epsg():
    rio_crs = rasterio.crs.CRS.from_wkt(
        'PROJCS["unnamed",GEOGCS["Unknown datum based upon the custom spheroid",'
        'DATUM["Not specified (based on custom spheroid)",'
        'SPHEROID["Custom spheroid",6371007.181,0]],'
        'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]],'
        'PROJECTION["Sinusoidal"],PARAMETER["longitude_of_center",0],'
        'PARAMETER["false_easting",0],PARAMETER["false_northing",0],'
        'UNIT["Meter",1],AXIS["Easting",EAST],AXIS["Northing",NORTH]]'
    )
    assert CRS(rio_crs).epsg is None


def test_norm_crs():
    assert norm_crs(None) is None
    assert norm_crs(Unset()) is None

    for v in [None, Unset()]:
        with pytest.raises(ValueError):
            _ = norm_crs_or_error(v)


@pytest.mark.parametrize(
    "x, expected_epsg",
    [
        (-180, 32601),
        ((-180, -30), 32701),
        (xy_(0.33, 10), 32631),
        (xy_(180, 10), 32660),
        (geom.point(6.89, 50.3, "epsg:4326"), 32632),
        (geom.point(6.89, -50.3, "epsg:4326"), 32732),
        (geom.point(6.89, 50.3, None), 32632),
        (geom.point(6.89, 50.3, "epsg:4326").to_crs("epsg:3857"), 32632),
        (geom.point(6.89, 50.3, "epsg:4326").to_crs("epsg:3857").buffer(1000), 32632),
        (geom.BoundingBox(0, 10, 7, 20), 32631),
        (geom.BoundingBox(0, 10, 7, 20, "epsg:4326"), 32631),
        (geom.BoundingBox(5, 10, 8, 20), 32632),
    ],
)
def test_crs_utm(x, expected_epsg):
    if isinstance(x, tuple):
        crs = CRS.utm(*x)
    else:
        crs = CRS.utm(x)

    assert crs.epsg is not None
    assert crs.epsg == expected_epsg
    if isinstance(x, tuple):
        return

    assert norm_crs_or_error("utm", x).epsg == expected_epsg
    assert norm_crs_or_error("UTM", x).epsg == expected_epsg
    assert norm_crs("UTM", x).epsg == expected_epsg

    if crs.proj.utm_zone.endswith("S"):
        assert norm_crs("utm-s", x) == norm_crs("utm", x)
        assert norm_crs("utm-n", x).epsg == expected_epsg - 100
    elif crs.proj.utm_zone.endswith("N"):
        assert norm_crs("utM-n", x) == norm_crs("utm", x)
        assert norm_crs("uTm-s", x).epsg == expected_epsg + 100

    with pytest.raises(ValueError):
        _ = CRS.utm(x, datum_name="no such datum")


def test_crs_units_issue_120():
    compound_crs_wkt = """COMPOUNDCRS["GDA94 / MGA zone 50 + Instantaneous Water Level Height",
    PROJCRS["GDA94 / MGA zone 50",
        BASEGEOGCRS["GDA94",
            DATUM["Geocentric Datum of Australia 1994",
                ELLIPSOID["GRS 1980",6378137,298.257222101,
                    LENGTHUNIT["metre",1]]],
            PRIMEM["Greenwich",0,
                ANGLEUNIT["degree",0.0174532925199433]],
            ID["EPSG",4283]],
        CONVERSION["UTM zone 50S",
            METHOD["Transverse Mercator",
                ID["EPSG",9807]],
            PARAMETER["Latitude of natural origin",0,
                ANGLEUNIT["degree",0.0174532925199433],
                ID["EPSG",8801]],
            PARAMETER["Longitude of natural origin",117,
                ANGLEUNIT["degree",0.0174532925199433],
                ID["EPSG",8802]],
            PARAMETER["Scale factor at natural origin",0.9996,
                SCALEUNIT["unity",1],
                ID["EPSG",8805]],
            PARAMETER["False easting",500000,
                LENGTHUNIT["metre",1],
                ID["EPSG",8806]],
            PARAMETER["False northing",10000000,
                LENGTHUNIT["metre",1],
                ID["EPSG",8807]]],
        CS[Cartesian,2],
            AXIS["easting",east,
                ORDER[1],
                LENGTHUNIT["metre",1]],
            AXIS["northing",north,
                ORDER[2],
                LENGTHUNIT["metre",1]],
        ID["EPSG",28350]],
    VERTCRS["Instantaneous Water Level Height",
        VDATUM["Instantaneous Water Level"],
        CS[vertical,1],
            AXIS["up",up,
                LENGTHUNIT["metre",1]],
        ID["EPSG",5829]]]"""
    assert CRS(compound_crs_wkt).units == ("metre", "metre")
