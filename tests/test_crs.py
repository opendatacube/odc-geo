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
from odc.geo.crs import CRS, CRSError, CRSMismatchError, crs_units_per_degree
from odc.geo.geom import common_crs
from odc.geo.testutils import epsg3577, epsg3857, epsg4326

# pylint: disable=missing-class-docstring,use-implicit-booleaness-not-comparison
# pylint: disable=comparison-with-itself,no-self-use


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

    with pytest.raises(CRSError):
        CRS(("random", "tuple"))

    crs = CRS("epsg:3857")
    with pytest.warns(UserWarning):
        crs_dict = crs.proj.to_dict()

    assert CRS(crs_dict) == crs

    wkt_crs = SimpleNamespace(to_wkt=Mock(return_value=crs.to_wkt()))
    assert getattr(wkt_crs, "to_epsg", None) is None
    assert CRS(wkt_crs) == crs


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
