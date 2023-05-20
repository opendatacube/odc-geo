# This file is part of the Open Data Cube, see https://opendatacube.org for more information
#
# Copyright (c) 2015-2020 ODC Contributors
# SPDX-License-Identifier: Apache-2.0
import math
import pickle

import numpy as np
import pytest
from affine import Affine
from pytest import approx

from odc.geo import CRS, CRSMismatchError, geom, wh_
from odc.geo.geobox import GeoBox, _round_to_res
from odc.geo.geom import (
    chop_along_antimeridian,
    clip_lon180,
    densify,
    force_2d,
    multigeom,
    projected_lon,
    triangulate,
)
from odc.geo.testutils import (
    SAMPLE_WKT_WITHOUT_AUTHORITY,
    epsg3577,
    epsg3857,
    epsg4326,
    from_fixed_point,
    gen_test_image_xy,
    to_fixed_point,
    xy_from_gbox,
)

# pylint: disable=protected-access, pointless-statement
# pylint: disable=too-many-statements,too-many-locals,too-many-lines,unnecessary-lambda-assignment


def test_pickleable():
    poly = geom.polygon([(10, 20), (20, 20), (20, 10), (10, 20)], crs=epsg4326)
    pickled = pickle.dumps(poly, pickle.HIGHEST_PROTOCOL)
    unpickled = pickle.loads(pickled)
    assert poly == unpickled


def test_props():
    crs = epsg4326

    box1 = geom.box(10, 10, 30, 30, crs=crs)
    assert box1
    assert box1.is_valid
    assert not box1.is_empty
    assert box1.area == 400.0
    assert box1.boundary.length == 80.0
    assert box1.centroid == geom.point(20, 20, crs)
    assert isinstance(box1.wkt, str)

    triangle = geom.polygon([(10, 20), (20, 20), (20, 10), (10, 20)], crs=crs)
    assert triangle.boundingbox == geom.BoundingBox(10, 10, 20, 20, crs)
    assert triangle.envelope.contains(triangle)

    assert box1.length == 80.0

    box1copy = geom.box(10, 10, 30, 30, crs=crs)
    assert box1 == box1copy
    assert box1.convex_hull == box1copy  # NOTE: this might fail because of point order

    box2 = geom.box(20, 10, 40, 30, crs=crs)
    assert box1 != box2

    bbox = geom.BoundingBox(1, 0, 10, 13)
    assert bbox.width == 9
    assert bbox.height == 13
    assert bbox.points == [(1, 0), (1, 13), (10, 0), (10, 13)]

    assert bbox.transform(Affine.identity()) == bbox
    assert bbox.transform(Affine.translation(1, 2)) == geom.BoundingBox(2, 2, 11, 15)

    pt = geom.point(3, 4, crs)
    assert pt.json["coordinates"] == (3.0, 4.0)
    assert "Point" in str(pt)
    assert bool(pt) is True
    assert pt.__nonzero__() is True

    # check "CRS as string is converted to class automatically"
    assert isinstance(geom.point(3, 4, "epsg:3857").crs, geom.CRS)

    # constructor with bad input should raise ValueError
    with pytest.raises(ValueError):
        geom.Geometry(object())


def test_tests():
    box1 = geom.box(10, 10, 30, 30, crs=epsg4326)
    box2 = geom.box(20, 10, 40, 30, crs=epsg4326)
    box3 = geom.box(30, 10, 50, 30, crs=epsg4326)
    box4 = geom.box(40, 10, 60, 30, crs=epsg4326)
    minibox = geom.box(15, 15, 25, 25, crs=epsg4326)

    assert not box1.touches(box2)
    assert box1.touches(box3)
    assert not box1.touches(box4)

    assert box1.intersects(box2)
    assert box1.intersects(box3)
    assert not box1.intersects(box4)

    assert not box1.crosses(box2)
    assert not box1.crosses(box3)
    assert not box1.crosses(box4)

    assert not box1.disjoint(box2)
    assert not box1.disjoint(box3)
    assert box1.disjoint(box4)

    assert box1.contains(minibox)
    assert not box1.contains(box2)
    assert not box1.contains(box3)
    assert not box1.contains(box4)

    assert minibox.within(box1)
    assert not box1.within(box2)
    assert not box1.within(box3)
    assert not box1.within(box4)


def test_ops():
    box1 = geom.box(10, 10, 30, 30, crs=epsg4326)
    box2 = geom.box(20, 10, 40, 30, crs=epsg4326)
    box3 = geom.box(20, 10, 40, 30, crs=epsg4326)
    box4 = geom.box(40, 10, 60, 30, crs=epsg4326)
    no_box = None

    assert box1 != box2
    assert box2 == box3
    assert box3 != no_box

    union1 = box1.union(box2)
    assert union1.area == 600.0

    assert box1.crs == epsg4326
    assert box1.assign_crs(None).crs is None
    assert box1.assign_crs(None).geom is box1.geom

    with pytest.raises(geom.CRSMismatchError):
        box1.union(box2.to_crs(epsg3857))

    inter1 = box1.intersection(box2)
    assert bool(inter1)
    assert inter1.area == 200.0

    inter2 = box1.intersection(box4)
    assert not bool(inter2)
    assert inter2.is_empty
    # assert not inter2.is_valid  TODO: what's going on here?

    diff1 = box1.difference(box2)
    assert diff1.area == 200.0

    symdiff1 = box1.symmetric_difference(box2)
    assert symdiff1.area == 400.0

    # test segmented
    line = geom.line([(0, 0), (0, 5), (10, 5)], epsg4326)
    line2 = line.segmented(2)
    assert line.crs is line2.crs
    assert line.length == line2.length
    assert len(line.coords) < len(line2.coords)
    poly = geom.polygon([(0, 0), (0, 5), (10, 5)], epsg4326)
    poly2 = poly.segmented(2)
    assert poly.crs is poly2.crs
    assert poly.length == poly2.length
    assert poly.area == poly2.area
    assert len(poly.geom.exterior.coords) < len(poly2.geom.exterior.coords)

    poly2 = poly.exterior.segmented(2)
    assert poly.crs is poly2.crs
    assert poly.length == poly2.length
    assert len(poly.geom.exterior.coords) < len(poly2.geom.coords)

    # test point.segmented is just a clone
    pt = geom.point(10, 20, "EPSG:4326")
    assert pt.segmented(1) is not pt
    assert pt.segmented(1) == pt

    # test interpolate
    pt = line.interpolate(1)
    assert pt.crs is line.crs
    assert pt.coords[0] == (0, 1)
    assert isinstance(pt.coords, list)

    with pytest.raises(TypeError):
        pt.interpolate(3)

    # test simplify
    poly = geom.polygon([(0, 0), (0, 5), (10, 5)], epsg4326)
    assert poly.simplify(100) == poly

    # test iteration
    poly_2_parts = geom.Geometry(
        {
            "type": "MultiPolygon",
            "coordinates": [
                [
                    [
                        [102.0, 2.0],
                        [103.0, 2.0],
                        [103.0, 3.0],
                        [102.0, 3.0],
                        [102.0, 2.0],
                    ]
                ],
                [
                    [
                        [100.0, 0.0],
                        [101.0, 0.0],
                        [101.0, 1.0],
                        [100.0, 1.0],
                        [100.0, 0.0],
                    ],
                    [
                        [100.2, 0.2],
                        [100.8, 0.2],
                        [100.8, 0.8],
                        [100.2, 0.8],
                        [100.2, 0.2],
                    ],
                ],
            ],
        },
        "EPSG:4326",
    )
    pp = list(poly_2_parts.geoms)
    assert len(pp) == 2
    assert all(p.crs == poly_2_parts.crs for p in pp)

    # test transform
    assert geom.point(0, 0, epsg4326).transform(
        lambda x, y: (x + 1, y + 2)
    ) == geom.point(1, 2, epsg4326)

    # test sides
    box = geom.box(1, 2, 11, 22, epsg4326)
    lines = list(geom.sides(box))
    assert all(line.crs is epsg4326 for line in lines)
    assert len(lines) == 4
    assert lines[0] == geom.line([(1, 2), (1, 22)], epsg4326)
    assert lines[1] == geom.line([(1, 22), (11, 22)], epsg4326)
    assert lines[2] == geom.line([(11, 22), (11, 2)], epsg4326)
    assert lines[3] == geom.line([(11, 2), (1, 2)], epsg4326)


def test_geom_split():
    box = geom.box(0, 0, 10, 30, epsg4326)
    line = geom.line([(5, 0), (5, 30)], epsg4326)
    bb = list(box.split(line))
    assert len(bb) == 2
    assert box.contains(bb[0] | bb[1])
    assert (box ^ (bb[0] | bb[1])).is_empty

    with pytest.raises(CRSMismatchError):
        list(box.split(geom.line([(5, 0), (5, 30)], epsg3857)))


def test_multigeom():
    p1, p2 = (0, 0), (1, 2)
    p3, p4 = (3, 4), (5, 6)
    b1 = geom.box(*p1, *p2, epsg4326)
    b2 = geom.box(*p3, *p4, epsg4326)
    bb = multigeom([b1, b2])
    assert bb.geom_type == "MultiPolygon"
    assert bb.crs is b1.crs
    assert len(list(bb.geoms)) == 2

    assert geom.multipolygon([[b1.boundary.coords], [b2.boundary.coords]], b1.crs) == bb

    g1 = geom.line([p1, p2], None)
    g2 = geom.line([p3, p4], None)
    gg = multigeom(iter([g1, g2, g1]))
    assert gg.geom_type == "MultiLineString"
    assert gg.crs is g1.crs
    assert len(list(gg.geoms)) == 3
    assert geom.multiline([[p1, p2], [p3, p4], [p1, p2]], None) == gg

    g1 = geom.point(*p1, epsg3857)
    g2 = geom.point(*p2, epsg3857)
    g3 = geom.point(*p3, epsg3857)
    gg = multigeom(iter([g1, g2, g3]))
    assert gg.geom_type == "MultiPoint"
    assert gg.crs is g1.crs
    assert len(list(gg.geoms)) == 3
    assert list(gg.geoms)[0] == g1
    assert list(gg.geoms)[1] == g2
    assert list(gg.geoms)[2] == g3

    assert geom.multipoint([p1, p2, p3], epsg3857) == gg

    # check mixed types
    assert (
        multigeom([geom.line([p1, p2], None), geom.point(*p1, None)]).geom_type
        == "GeometryCollection"
    )

    # can't mix CRSs
    with pytest.raises(CRSMismatchError):
        multigeom([geom.line([p1, p2], epsg4326), geom.line([p3, p4], epsg3857)])

    assert multigeom([gg]).geom_type == "GeometryCollection"


def test_shapely_wrappers():
    poly = geom.polygon([(0, 0), (0, 5), (10, 5)], epsg4326)

    assert isinstance(poly.svg(), str)
    assert isinstance(poly._repr_svg_(), str)

    with_hole = poly.buffer(1) - poly
    assert len(poly.interiors) == 0
    assert len(with_hole.interiors) == 1
    assert isinstance(with_hole.interiors, list)
    assert isinstance(with_hole.interiors[0], geom.Geometry)
    assert with_hole.interiors[0].crs == with_hole.crs
    assert poly.exterior.crs == poly.crs

    x, y = poly.exterior.xy
    assert len(x) == len(y)
    assert x.typecode == y.typecode
    assert x.typecode == "d"

    assert ((poly | poly) - poly).is_empty
    assert ((poly & poly) - poly).is_empty
    assert (poly ^ poly).is_empty
    assert (poly - poly).is_empty


def test_to_crs():
    poly = geom.polygon([(0, 0), (0, 5), (10, 5)], epsg4326)
    src_num_points = len(poly.exterior.xy[0])

    assert poly.crs is epsg4326
    assert poly.to_crs(epsg3857).crs is epsg3857
    assert poly.to_crs("EPSG:3857").crs == "EPSG:3857"
    assert poly.to_crs("EPSG:3857", 0.1).crs == epsg3857
    assert poly.to_crs(3857, "auto").crs == epsg3857

    assert poly.exterior.to_crs(epsg3857) == poly.to_crs(epsg3857).exterior

    # test that by default segmentation happens
    assert len(poly.to_crs(epsg3857, resolution=1).exterior.xy[0]) > src_num_points

    # test no segmentation by default or when inf/nan is supplied
    assert len(poly.to_crs(epsg3857).exterior.xy[0]) == src_num_points
    assert len(poly.to_crs(epsg3857, float("+inf")).exterior.xy[0]) == src_num_points
    assert len(poly.to_crs(epsg3857, float("nan")).exterior.xy[0]) == src_num_points

    # test the segmentation works on multi-polygons
    mpoly = geom.box(0, 0, 1, 3, "EPSG:4326") | geom.box(2, 4, 3, 6, "EPSG:4326")

    assert mpoly.geom_type == "MultiPolygon"
    assert mpoly.to_crs(epsg3857).geom_type == "MultiPolygon"
    assert mpoly.to_crs(epsg3857, 1).geom_type == "MultiPolygon"

    poly = geom.polygon([(0, 0), (0, 5), (10, 5)], None)
    assert poly.crs is None
    with pytest.raises(ValueError):
        poly.to_crs(epsg3857)


def test_to_crs_with_check():
    crs = "+proj=ortho +lat0=40"
    gg = geom.line([[-180, 0], [180, 0]], 4326).segmented(5)
    assert gg.to_crs(crs).is_valid is False
    assert gg.to_crs(crs, check_and_fix=True).is_valid is True

    gg = gg.buffer(1.1) & geom.box(-180, -90, 180, 90, 4326)

    assert gg.to_crs(crs).is_valid is False
    assert gg.to_crs(crs, check_and_fix=True).is_valid is True


def test_to_crs_utm():
    poly = geom.box(0.1, 43, 1.3, 44, epsg4326)
    assert poly.to_crs("utm").crs.epsg == 32631
    assert poly.to_crs("utm") == poly.to_crs(32631)
    assert poly.to_crs("utm-s").crs.epsg == 32731

    poly = geom.box(0.1, -44, 1.3, -43, epsg4326)
    assert poly.to_crs("utm-n").crs.epsg == 32631
    assert poly.to_crs("utm") == poly.to_crs(32731)


def test_densify():
    s_x10 = [(0, 0), (10, 0)]
    assert densify(s_x10, 20) == s_x10
    assert densify(s_x10, 200) == s_x10
    assert densify(s_x10, 5) == [(0, 0), (5, 0), (10, 0)]
    assert densify(s_x10, 4) == [(0, 0), (4, 0), (8, 0), (10, 0)]


def test_unary_union():
    box1 = geom.box(10, 10, 30, 30, crs=epsg4326)
    box2 = geom.box(20, 10, 40, 30, crs=epsg4326)
    box3 = geom.box(30, 10, 50, 30, crs=epsg4326)
    box4 = geom.box(40, 10, 60, 30, crs=epsg4326)

    union0 = geom.unary_union([box1])
    assert union0 == box1

    union1 = geom.unary_union([box1, box4])
    assert union1.geom_type == "MultiPolygon"
    assert union1.area == 2.0 * box1.area

    union2 = geom.unary_union([box1, box2])
    assert union2.geom_type == "Polygon"
    assert union2.area == 1.5 * box1.area

    union3 = geom.unary_union([box1, box2, box3, box4])
    assert union3.geom_type == "Polygon"
    assert union3.area == 2.5 * box1.area

    union4 = geom.unary_union([union1, box2, box3])
    assert union4.geom_type == "Polygon"
    assert union4.area == 2.5 * box1.area

    assert geom.unary_union([]) is None

    with pytest.raises(ValueError):
        geom.unary_union([box1, box1.to_crs(epsg3577)])


def test_unary_intersection():
    box1 = geom.box(10, 10, 30, 30, crs=epsg4326)
    box2 = geom.box(15, 10, 35, 30, crs=epsg4326)
    box3 = geom.box(20, 10, 40, 30, crs=epsg4326)
    box4 = geom.box(25, 10, 45, 30, crs=epsg4326)
    box5 = geom.box(30, 10, 50, 30, crs=epsg4326)
    box6 = geom.box(35, 10, 55, 30, crs=epsg4326)

    assert geom.intersects(box1, box2)
    inter1 = geom.unary_intersection([box1])
    assert bool(inter1)
    assert inter1 == box1

    inter2 = geom.unary_intersection([box1, box2])
    assert bool(inter2)
    assert inter2.area == 300.0

    inter3 = geom.unary_intersection([box1, box2, box3])
    assert bool(inter3)
    assert inter3.area == 200.0

    inter4 = geom.unary_intersection([box1, box2, box3, box4])
    assert bool(inter4)
    assert inter4.area == 100.0

    inter5 = geom.unary_intersection([box1, box2, box3, box4, box5])
    assert bool(inter5)
    assert inter5.geom_type == "LineString"
    assert inter5.length == 20.0

    inter6 = geom.unary_intersection([box1, box2, box3, box4, box5, box6])
    assert not bool(inter6)
    assert inter6.is_empty


def test_gen_test_image_xy():
    gbox = GeoBox(wh_(3, 7), Affine.translation(10, 1000), epsg3857)

    xy, denorm = gen_test_image_xy(gbox, "float64")
    assert xy.dtype == "float64"
    assert xy.shape == (2,) + gbox.shape

    x, y = denorm(xy)
    x_, y_ = xy_from_gbox(gbox)

    np.testing.assert_almost_equal(x, x_)
    np.testing.assert_almost_equal(y, y_)

    xy, denorm = gen_test_image_xy(gbox, "uint16")
    assert xy.dtype == "uint16"
    assert xy.shape == (2,) + gbox.shape

    x, y = denorm(xy[0], xy[1])
    assert x.shape == xy.shape[1:]
    assert y.shape == xy.shape[1:]
    assert x.dtype == "float64"

    x_, y_ = xy_from_gbox(gbox)

    np.testing.assert_almost_equal(x, x_, 4)
    np.testing.assert_almost_equal(y, y_, 4)

    for dt in ("int8", np.int16, np.dtype(np.uint64)):
        xy, _ = gen_test_image_xy(gbox, dt)
        assert xy.dtype == dt

    # check no-data
    xy, denorm = gen_test_image_xy(gbox, "float32")
    assert xy.dtype == "float32"
    assert xy.shape == (2,) + gbox.shape
    xy[0, 0, :] = np.nan
    xy[1, 1, :] = np.nan
    _xy = denorm(xy, nodata=np.nan)
    assert np.isnan(_xy[:, :2]).all()
    np.testing.assert_almost_equal(_xy[0][2:], x_[2:], 6)
    np.testing.assert_almost_equal(_xy[1][2:], y_[2:], 6)

    xy, denorm = gen_test_image_xy(gbox, "int16")
    assert xy.dtype == "int16"
    assert xy.shape == (2,) + gbox.shape
    xy[0, 0, :] = -999
    xy[1, 1, :] = -999
    _xy = denorm(xy, nodata=-999)
    assert np.isnan(_xy[:, :2]).all()
    np.testing.assert_almost_equal(_xy[0][2:], x_[2:], 4)
    np.testing.assert_almost_equal(_xy[1][2:], y_[2:], 4)

    # call without arguments should return linear mapping
    A = denorm()
    assert isinstance(A, Affine)


def test_fixed_point():
    aa = np.asarray([0, 0.5, 1])
    uu = to_fixed_point(aa, "uint8")
    assert uu.dtype == "uint8"
    assert aa.shape == uu.shape
    assert tuple(uu.ravel()) == (0, 128, 255)

    aa_ = from_fixed_point(uu)
    assert aa_.dtype == "float64"
    dd = np.abs(aa - aa_)
    assert (dd < 1 / 255.0).all()

    uu = to_fixed_point(aa, "uint16")
    assert uu.dtype == "uint16"
    assert tuple(uu.ravel()) == (0, 0x8000, 0xFFFF)

    uu = to_fixed_point(aa, "int16")
    assert uu.dtype == "int16"
    assert tuple(uu.ravel()) == (0, 0x4000, 0x7FFF)

    aa_ = from_fixed_point(uu)
    dd = np.abs(aa - aa_)
    assert (dd < 1.0 / 0x7FFF).all()


def test_projected_lon():
    assert projected_lon(epsg3857, 180).crs is epsg3857
    assert projected_lon("EPSG:3577", 100).crs == epsg3577


def test_chop():
    poly = geom.box(618300, -1876800, 849000, -1642500, "EPSG:32660")

    chopped = chop_along_antimeridian(poly)
    assert chopped.crs is poly.crs
    assert chopped.geom_type == "MultiPolygon"
    assert len(list(chopped.geoms)) == 2

    poly = geom.box(0, 0, 10, 20, "EPSG:4326")._to_crs(epsg3857)
    assert poly.crs is epsg3857
    assert chop_along_antimeridian(poly) is poly

    with pytest.raises(ValueError):
        chop_along_antimeridian(geom.box(0, 1, 2, 3, None))


def test_clip_lon180():
    err = 1e-9

    def b(rside):
        return geom.box(170, 0, rside, 10, epsg4326)

    def b_neg(lside):
        return geom.box(lside, 0, -170, 10, epsg4326)

    assert clip_lon180(b(180 - err)) == b(180)
    assert clip_lon180(b(-180 + err)) == b(180)

    assert clip_lon180(b_neg(180 - err)) == b_neg(-180)
    assert clip_lon180(b_neg(-180 + err)) == b_neg(-180)

    bb = multigeom([b(180 - err), b_neg(180 - err)])
    bb_ = list(clip_lon180(bb).geoms)
    assert bb_[0] == b(180)
    assert bb_[1] == b_neg(-180)


def test_wrap_dateline():
    albers_crs = epsg3577
    geog_crs = epsg4326

    wrap = geom.polygon(
        [
            (3658653.1976781483, -4995675.379595791),
            (4025493.916030875, -3947239.249752495),
            (4912789.243100313, -4297237.125269571),
            (4465089.861944263, -5313778.16975072),
            (3658653.1976781483, -4995675.379595791),
        ],
        crs=albers_crs,
    )
    wrapped = wrap.to_crs(geog_crs)
    assert wrapped.geom_type == "Polygon"
    assert wrapped.intersects(geom.line([(0, -90), (0, 90)], crs=geog_crs))
    wrapped = wrap.to_crs(geog_crs, wrapdateline=True)
    assert wrapped.geom_type == "MultiPolygon"
    assert not wrapped.intersects(geom.line([(0, -90), (0, 90)], crs=geog_crs))


@pytest.mark.parametrize(
    "pts",
    [
        [
            (12231455.716333, -5559752.598333),
            (12231455.716333, -4447802.078667),
            (13343406.236, -4447802.078667),
            (13343406.236, -5559752.598333),
            (12231455.716333, -5559752.598333),
        ],
        [
            (13343406.236, -5559752.598333),
            (13343406.236, -4447802.078667),
            (14455356.755667, -4447802.078667),
            (14455356.755667, -5559752.598333),
            (13343406.236, -5559752.598333),
        ],
        [
            (14455356.755667, -5559752.598333),
            (14455356.755667, -4447802.078667),
            (15567307.275333, -4447802.078667),
            (15567307.275333, -5559752.598333),
            (14455356.755667, -5559752.598333),
        ],
    ],
)
def test_wrap_dateline_sinusoidal(pts):
    sinus_crs = geom.CRS(
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

    wrap = geom.polygon(pts, crs=sinus_crs)
    wrapped = wrap.to_crs(epsg4326)
    assert wrapped.geom_type == "Polygon"
    wrapped = wrap.to_crs(epsg4326, wrapdateline=True)
    assert wrapped.geom_type == "MultiPolygon"
    assert not wrapped.intersects(geom.line([(0, -90), (0, 90)], crs=epsg4326))


def test_wrap_dateline_utm():
    poly = geom.box(618300, -1876800, 849000, -1642500, "EPSG:32660")

    wrapped = poly.to_crs(epsg4326)
    assert wrapped.geom_type == "Polygon"
    assert wrapped.intersects(geom.line([(0, -90), (0, 90)], crs=epsg4326))
    wrapped = poly.to_crs(epsg4326, wrapdateline=True)
    assert wrapped.geom_type == "MultiPolygon"
    assert not wrapped.intersects(geom.line([(0, -90), (0, 90)], crs=epsg4326))


def test_3d_geometry_converted_to_2d_geometry():
    coordinates = [
        (115.8929714190001, -28.577007674999948, 0.0),
        (115.90275429200005, -28.57698532699993, 0.0),
        (115.90412631000004, -28.577577566999935, 0.0),
        (115.90157040700001, -28.58521105999995, 0.0),
        (115.89382838900008, -28.585473711999953, 0.0),
        (115.8929714190001, -28.577007674999948, 0.0),
    ]
    geom_3d = {"coordinates": [coordinates], "type": "Polygon"}
    geom_2d = {"coordinates": [[(x, y) for x, y, z in coordinates]], "type": "Polygon"}

    g_2d = geom.Geometry(geom_2d)
    g_3d = geom.Geometry(geom_3d)

    assert {2} == {len(pt) for pt in g_3d.boundary.coords}  # All coordinates are 2D

    assert g_2d == g_3d  # 3D geometry has been converted to a 2D by dropping the Z axis


def test_3d_point_converted_to_2d_point():
    point = (-35.5029340, 145.9312455, 0.0)

    point_3d = {"coordinates": point, "type": "Point"}
    point_2d = {"coordinates": (point[0], point[1]), "type": "Point"}

    p_2d = geom.Geometry(point_2d)
    p_3d = geom.Geometry(point_3d)

    assert len(p_3d.coords[0]) == 2

    assert p_2d == p_3d


def test_mid_longitude():
    assert geom.mid_longitude(geom.point(10, 3, "epsg:4326")) == approx(10)
    assert geom.mid_longitude(geom.point(10, 3, "epsg:4326").buffer(3)) == approx(10)
    assert geom.mid_longitude(
        geom.point(10, 3, "epsg:4326").buffer(3).to_crs("epsg:3857")
    ) == approx(10)


def test_point_transformer():
    tr = epsg3857.transformer_to_crs(epsg4326)
    tr_back = epsg4326.transformer_to_crs(epsg3857)

    pts = [(0, 0), (0, 1), (1, 2), (10, 11)]
    x, y = np.vstack(pts).astype("float64").T

    pts_expect = [geom.point(*pt, epsg3857).to_crs(epsg4326).points[0] for pt in pts]

    x_expect = [pt[0] for pt in pts_expect]
    y_expect = [pt[1] for pt in pts_expect]

    x_, y_ = tr(x, y)
    assert x_.shape == x.shape
    np.testing.assert_array_almost_equal(x_, x_expect)
    np.testing.assert_array_almost_equal(y_, y_expect)

    x, y = (a.reshape(2, 2) for a in (x, y))
    x_, y_ = tr(x, y)
    assert x_.shape == x.shape

    xb, yb = tr_back(x_, y_)
    np.testing.assert_array_almost_equal(x, xb)
    np.testing.assert_array_almost_equal(y, yb)

    # check nans
    x_, y_ = tr(np.asarray([np.nan, 0, np.nan]), np.asarray([0, np.nan, np.nan]))

    assert np.isnan(x_).all()
    assert np.isnan(y_).all()


def test_base_internals():
    no_epsg_crs = CRS(SAMPLE_WKT_WITHOUT_AUTHORITY)
    assert no_epsg_crs.epsg is None

    gjson_bad = {"type": "a", "coordinates": [1, [2, 3, 4]]}
    assert force_2d(gjson_bad) == {"type": "a", "coordinates": [1, [2, 3]]}

    with pytest.raises(ValueError):
        force_2d({"type": "a", "coordinates": [set("not a valid element")]})

    assert _round_to_res(0.2, 1.0) == 1
    assert _round_to_res(0.0, 1.0) == 0
    assert _round_to_res(0.05, 1.0) == 0


def test_geom_clone():
    b = geom.box(0, 0, 10, 20, epsg4326)
    assert b == b.clone()
    assert b.geom is not b.clone().geom

    assert b == geom.Geometry(b)
    assert b.geom is not geom.Geometry(b).geom


@pytest.mark.parametrize("crs", [None, epsg4326])
@pytest.mark.parametrize("nside", [3, 7, 10])
def test_bbox_boundary(crs, nside):
    n_pts_expect = (nside - 1) * 4
    bbox = geom.BoundingBox(0, 20, 11, 29, crs)
    poly = bbox.polygon
    bb = bbox.boundary()
    assert bb.crs == bbox.crs
    assert bb.is_ring
    assert bb.geom_type == "LineString"
    assert len(bb.coords[:-1]) == 4
    assert (poly.exterior ^ bbox.boundary()).is_empty
    assert (bbox.boundary() - poly).is_empty

    bb = bbox.boundary(nside)
    assert n_pts_expect == len(bb.coords[:-1])
    assert (bbox.boundary() - poly).is_empty


def test_lonlat_bounds():
    # example from landsat scene: spans lon=180
    poly = geom.box(618300, -1876800, 849000, -1642500, "EPSG:32660")

    bb = geom.lonlat_bounds(poly)
    assert bb.left < 180 < bb.right
    assert geom.lonlat_bounds(poly) == geom.lonlat_bounds(poly, resolution=1e8)

    bb = geom.lonlat_bounds(poly, mode="quick")
    assert bb.right - bb.left > 180

    poly = geom.box(1, -10, 2, 20, "EPSG:4326")
    assert geom.lonlat_bounds(poly) == poly.boundingbox

    with pytest.raises(ValueError):
        geom.lonlat_bounds(geom.box(0, 0, 1, 1, None))

    multi = {
        "type": "MultiPolygon",
        "coordinates": [
            [[[174, 52], [174, 53], [175, 53], [174, 52]]],
            [[[168, 54], [167, 55], [167, 54], [168, 54]]],
        ],
    }

    multi_geom = geom.Geometry(multi, "epsg:4326")
    multi_geom_projected = multi_geom.to_crs("epsg:32659")
    expect = approx(geom.lonlat_bounds(multi_geom))

    assert geom.lonlat_bounds(multi_geom_projected) == expect
    assert geom.lonlat_bounds(multi_geom_projected, resolution="auto") == expect


def test_geojson():
    b = geom.box(0, 0, 10, 20, epsg4326)
    gjson = b.geojson()
    assert set(list(gjson)) == set(["type", "geometry", "properties"])
    assert gjson["type"] == "Feature"
    assert gjson["properties"] == {}
    _b = geom.Geometry(gjson["geometry"], crs=epsg4326)
    assert (b - _b).area < 1e-6

    # crs=None case should work too
    gjson = b.assign_crs(None).geojson()
    assert set(list(gjson)) == set(["type", "geometry", "properties"])
    assert gjson["type"] == "Feature"
    assert gjson["properties"] == {}
    _b = geom.Geometry(gjson["geometry"], crs=epsg4326)
    assert (b - _b).area < 1e-6

    gjson = b.to_crs(epsg3857).geojson(region_code="33")
    assert set(list(gjson)) == set(["type", "geometry", "properties"])
    assert gjson["type"] == "Feature"
    assert gjson["properties"] == {"region_code": "33"}

    _b = geom.Geometry(gjson, crs=epsg4326)
    assert (b - _b).area < 1e-6

    _b = geom.Geometry(gjson)
    assert _b.crs == epsg4326
    assert (b - _b).area < 1e-6

    _b = geom.Geometry(dict(type="FeatureCollection", features=[gjson]))
    assert _b.crs == epsg4326
    assert (b - _b).area < 1e-6

    _b = geom.Geometry(gjson["geometry"], crs=epsg4326)
    assert (b - _b).area < 1e-6

    with pytest.raises(ValueError):
        _ = geom.Geometry({})

    # check FeatureCollection output
    gg = b.buffer(0.1).exterior | b.centroid
    assert gg.geom_type == "GeometryCollection"
    gjson = gg.geojson(k=3)
    assert gjson["type"] == "FeatureCollection"
    assert set(gjson) == set(["type", "features"])
    assert gjson["features"][0]["properties"]["k"] == 3
    assert gjson["features"][1]["properties"]["k"] == 3
    gg_ = geom.Geometry(gjson)
    assert gg_.geom_type == "GeometryCollection"
    assert gg_.crs == "epsg:4326"
    assert len(list(gg_.geoms)) == len(list(gg.geoms))

    gg = geom.Geometry(
        {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [0, 1]},
                },
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [10, 20]},
                },
            ],
        }
    )
    assert gg.geom_type == "MultiPoint"
    assert len(list(gg.geoms)) == 2


@pytest.mark.xfail(
    True, reason="Bounds computation for large geometries in safe mode is broken"
)
def test_lonalt_bounds_more_than_180():
    poly = geom.box(-150, -30, 150, 30, epsg4326).to_crs(epsg3857, math.inf)

    assert geom.lonlat_bounds(poly, "quick") == approx((-150, -30, 150, 30))
    assert geom.lonlat_bounds(poly, "safe") == approx((-150, -30, 150, 30))


def test_mul_affine():
    g = geom.point(1, 2, epsg4326)
    x10 = Affine.scale(10)

    with pytest.warns(DeprecationWarning):
        # it warns because x10.__mul__(g) is called first and it attempts to
        # iterate over g, which is deprecated.
        #
        # Only then g.__rmul__(x10) is called
        assert Affine.translation(10, 100) * g == geom.point(11, 102, epsg4326)

        assert x10 * g == geom.point(10, 20, epsg4326)
        assert x10 * g == g.transform(x10)

    mg = geom.multigeom([g, g, g])
    assert mg.is_multi
    assert mg.transform(x10).is_multi
    assert mg.transform(x10).crs == mg.crs
    assert mg.transform(x10, crs=None).crs is None

    with pytest.warns(DeprecationWarning):
        assert x10 * mg == mg.transform(x10)


def test_deprecation_warnings():
    g = geom.point(1, 2, epsg4326)
    mg = g | g.buffer(10).boundary
    assert mg.is_multi

    with pytest.warns(DeprecationWarning):
        for _ in mg:
            pass


def test_filter():
    _keep = lambda x, y: True
    _drop = lambda x, y: False

    pt = geom.point(1, 2, epsg4326)
    poly = geom.box(1, 2, 3, 4, epsg4326)
    ring = poly.exterior
    multi_pt = geom.multipoint([(1, 1), (2, 2)], epsg4326)
    poly_with_holes = poly - poly.centroid.buffer(0.1)
    assert len(poly_with_holes.interiors) > 0

    for g in [
        pt,
        poly,
        poly_with_holes,
        ring,
        multi_pt,
        pt | ring,
        ring.buffer(1) | ring,
    ]:
        assert g.filter(_drop).is_empty
        assert g.filter(_keep) == g
        assert g.dropna() == g

    multi_pt_nan = multi_pt | geom.point(float("nan"), 7, epsg4326)
    assert len(list(multi_pt_nan.geoms)) == len(list(multi_pt.geoms)) + 1
    assert multi_pt_nan.dropna() == multi_pt


@pytest.mark.parametrize("crs", [None, "epsg:4326"])
@pytest.mark.parametrize("n", [10, 100, 23])
@pytest.mark.parametrize("with_edges", [False, True])
def test_qr2sample(crs, n, with_edges):
    bbox = geom.BoundingBox(10, 33, 21, 55, crs)

    def run_checks(g: geom.Geometry):
        assert g.crs == bbox.crs
        assert (g - bbox.polygon).is_empty
        assert g.is_multi
        assert g.geom_type == "MultiPoint"
        if with_edges is False:
            assert len(list(g.geoms)) == n
        else:
            assert len(list(g.geoms)) > n
            assert (bbox.polygon - triangulate(g)).is_empty

    xx1 = bbox.qr2sample(n, with_edges=with_edges)
    xx2 = bbox.qr2sample(n, offset=100, with_edges=with_edges)
    assert not (xx1 ^ xx2).is_empty
    run_checks(xx1)
    run_checks(xx2)

    xx1_ = bbox.qr2sample(n, with_edges=with_edges)
    assert (xx1 ^ xx1_).is_empty


@pytest.mark.parametrize("crs", [None, "epsg:4326"])
def test_triangulate(crs):
    bbox = geom.BoundingBox(0, 0, 1, 1, crs)

    gg = triangulate(bbox.boundary())
    geoms = list(gg.geoms)
    assert gg.crs == bbox.crs
    assert gg.is_multi
    assert gg.is_valid
    assert len(geoms) == 2
    assert geoms[0].geom_type == "Polygon"
    assert (gg - bbox.polygon).is_empty

    gg = triangulate(bbox.boundary(4), edges=True)
    geoms = list(gg.geoms)
    assert gg.crs == bbox.crs
    assert gg.is_multi
    assert gg.is_valid
    assert geoms[0].geom_type == "LineString"
    assert (gg - bbox.polygon).is_empty
