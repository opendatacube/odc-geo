# This file is part of the Open Data Cube, see https://opendatacube.org for more information
#
# Copyright (c) 2015-2020 ODC Contributors
# SPDX-License-Identifier: Apache-2.0
import math
import pickle
from unittest.mock import MagicMock

import numpy as np
import pytest
from affine import Affine
from pytest import approx

from odc import geo as geometry
from odc.geo import (
    CRS,
    BoundingBox,
    CRSMismatchError,
    GeoBox,
    affine_from_pts,
    apply_affine,
    bbox_union,
    chop_along_antimeridian,
    clip_lon180,
    common_crs,
    compute_axis_overlap,
    compute_reproject_roi,
    crs_units_per_degree,
    decompose_rws,
    get_scale_at_point,
    is_affine_st,
    multigeom,
    native_pix_transform,
    projected_lon,
    roi_is_empty,
    roi_normalise,
    roi_shape,
    scaled_down_geobox,
    split_translation,
    w_,
)
from odc.geo._base import (
    _align_pix,
    _guess_crs_str,
    _mk_crs_coord,
    _norm_crs,
    _norm_crs_or_error,
    _round_to_res,
    bounding_box_in_pixel_domain,
    force_2d,
    geobox_intersection_conservative,
    geobox_union_conservative,
)
from odc.geo.testutils.geom import (
    SAMPLE_WKT_WITHOUT_AUTHORITY,
    AlbersGS,
    epsg3577,
    epsg3857,
    epsg4326,
    from_fixed_point,
    gen_test_image_xy,
    mkA,
    to_fixed_point,
    xy_from_gbox,
    xy_norm,
)
from odc.geo.tools import gbox_boundary


def test_pickleable():
    poly = geometry.polygon([(10, 20), (20, 20), (20, 10), (10, 20)], crs=epsg4326)
    pickled = pickle.dumps(poly, pickle.HIGHEST_PROTOCOL)
    unpickled = pickle.loads(pickled)
    assert poly == unpickled


def test_geobox_simple():
    t = geometry.GeoBox(
        4000, 4000, Affine(0.00025, 0.0, 151.0, 0.0, -0.00025, -29.0), epsg4326
    )

    expect_lon = np.asarray(
        [
            151.000125,
            151.000375,
            151.000625,
            151.000875,
            151.001125,
            151.001375,
            151.001625,
            151.001875,
            151.002125,
            151.002375,
        ]
    )

    expect_lat = np.asarray(
        [
            -29.000125,
            -29.000375,
            -29.000625,
            -29.000875,
            -29.001125,
            -29.001375,
            -29.001625,
            -29.001875,
            -29.002125,
            -29.002375,
        ]
    )
    expect_resolution = np.asarray([-0.00025, 0.00025])

    assert t.coordinates["latitude"].values.shape == (4000,)
    assert t.coordinates["longitude"].values.shape == (4000,)

    np.testing.assert_almost_equal(t.resolution, expect_resolution)
    np.testing.assert_almost_equal(t.coords["latitude"].values[:10], expect_lat)
    np.testing.assert_almost_equal(t.coords["longitude"].values[:10], expect_lon)

    assert (t == "some random thing") is False

    # ensure GeoBox accepts string CRS
    assert isinstance(
        geometry.GeoBox(
            4000, 4000, Affine(0.00025, 0.0, 151.0, 0.0, -0.00025, -29.0), "epsg:4326"
        ).crs,
        CRS,
    )

    # Check GeoBox class is hashable
    t_copy = GeoBox(t.width, t.height, t.transform, t.crs)
    t_other = GeoBox(t.width + 1, t.height, t.transform, t.crs)
    assert t_copy is not t
    assert t == t_copy
    assert len({t, t, t_copy}) == 1
    assert len({t, t_copy, t_other}) == 2


def test_props():
    crs = epsg4326

    box1 = geometry.box(10, 10, 30, 30, crs=crs)
    assert box1
    assert box1.is_valid
    assert not box1.is_empty
    assert box1.area == 400.0
    assert box1.boundary.length == 80.0
    assert box1.centroid == geometry.point(20, 20, crs)

    triangle = geometry.polygon([(10, 20), (20, 20), (20, 10), (10, 20)], crs=crs)
    assert triangle.boundingbox == geometry.BoundingBox(10, 10, 20, 20)
    assert triangle.envelope.contains(triangle)

    assert box1.length == 80.0

    box1copy = geometry.box(10, 10, 30, 30, crs=crs)
    assert box1 == box1copy
    assert box1.convex_hull == box1copy  # NOTE: this might fail because of point order

    box2 = geometry.box(20, 10, 40, 30, crs=crs)
    assert box1 != box2

    bbox = geometry.BoundingBox(1, 0, 10, 13)
    assert bbox.width == 9
    assert bbox.height == 13
    assert bbox.points == [(1, 0), (1, 13), (10, 0), (10, 13)]

    assert bbox.transform(Affine.identity()) == bbox
    assert bbox.transform(Affine.translation(1, 2)) == geometry.BoundingBox(
        2, 2, 11, 15
    )

    pt = geometry.point(3, 4, crs)
    assert pt.json["coordinates"] == (3.0, 4.0)
    assert "Point" in str(pt)
    assert bool(pt) is True
    assert pt.__nonzero__() is True

    # check "CRS as string is converted to class automatically"
    assert isinstance(geometry.point(3, 4, "epsg:3857").crs, geometry.CRS)

    # constructor with bad input should raise ValueError
    with pytest.raises(ValueError):
        geometry.Geometry(object())


def test_tests():
    box1 = geometry.box(10, 10, 30, 30, crs=epsg4326)
    box2 = geometry.box(20, 10, 40, 30, crs=epsg4326)
    box3 = geometry.box(30, 10, 50, 30, crs=epsg4326)
    box4 = geometry.box(40, 10, 60, 30, crs=epsg4326)
    minibox = geometry.box(15, 15, 25, 25, crs=epsg4326)

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
    box1 = geometry.box(10, 10, 30, 30, crs=epsg4326)
    box2 = geometry.box(20, 10, 40, 30, crs=epsg4326)
    box3 = geometry.box(20, 10, 40, 30, crs=epsg4326)
    box4 = geometry.box(40, 10, 60, 30, crs=epsg4326)
    no_box = None

    assert box1 != box2
    assert box2 == box3
    assert box3 != no_box

    union1 = box1.union(box2)
    assert union1.area == 600.0

    with pytest.raises(geometry.CRSMismatchError):
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
    line = geometry.line([(0, 0), (0, 5), (10, 5)], epsg4326)
    line2 = line.segmented(2)
    assert line.crs is line2.crs
    assert line.length == line2.length
    assert len(line.coords) < len(line2.coords)
    poly = geometry.polygon([(0, 0), (0, 5), (10, 5)], epsg4326)
    poly2 = poly.segmented(2)
    assert poly.crs is poly2.crs
    assert poly.length == poly2.length
    assert poly.area == poly2.area
    assert len(poly.geom.exterior.coords) < len(poly2.geom.exterior.coords)

    poly2 = poly.exterior.segmented(2)
    assert poly.crs is poly2.crs
    assert poly.length == poly2.length
    assert len(poly.geom.exterior.coords) < len(poly2.geom.coords)

    # test interpolate
    pt = line.interpolate(1)
    assert pt.crs is line.crs
    assert pt.coords[0] == (0, 1)
    assert isinstance(pt.coords, list)

    with pytest.raises(TypeError):
        pt.interpolate(3)

    # test array interface
    assert line.__array_interface__ is not None
    assert np.array(line).shape == (3, 2)

    # test simplify
    poly = geometry.polygon([(0, 0), (0, 5), (10, 5)], epsg4326)
    assert poly.simplify(100) == poly

    # test iteration
    poly_2_parts = geometry.Geometry(
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
    pp = list(poly_2_parts)
    assert len(pp) == 2
    assert all(p.crs == poly_2_parts.crs for p in pp)

    # test transform
    assert geometry.point(0, 0, epsg4326).transform(
        lambda x, y: (x + 1, y + 2)
    ) == geometry.point(1, 2, epsg4326)

    # test sides
    box = geometry.box(1, 2, 11, 22, epsg4326)
    lines = list(geometry.sides(box))
    assert all(line.crs is epsg4326 for line in lines)
    assert len(lines) == 4
    assert lines[0] == geometry.line([(1, 2), (1, 22)], epsg4326)
    assert lines[1] == geometry.line([(1, 22), (11, 22)], epsg4326)
    assert lines[2] == geometry.line([(11, 22), (11, 2)], epsg4326)
    assert lines[3] == geometry.line([(11, 2), (1, 2)], epsg4326)


def test_geom_split():
    box = geometry.box(0, 0, 10, 30, epsg4326)
    line = geometry.line([(5, 0), (5, 30)], epsg4326)
    bb = list(box.split(line))
    assert len(bb) == 2
    assert box.contains(bb[0] | bb[1])
    assert (box ^ (bb[0] | bb[1])).is_empty

    with pytest.raises(CRSMismatchError):
        list(box.split(geometry.line([(5, 0), (5, 30)], epsg3857)))


def test_common_crs():
    assert common_crs([]) is None
    assert (
        common_crs(
            [geometry.point(0, 0, epsg4326), geometry.line([(0, 0), (1, 1)], epsg4326)]
        )
        is epsg4326
    )

    with pytest.raises(CRSMismatchError):
        common_crs(
            [geometry.point(0, 0, epsg4326), geometry.line([(0, 0), (1, 1)], epsg3857)]
        )


def test_multigeom():
    p1, p2 = (0, 0), (1, 2)
    p3, p4 = (3, 4), (5, 6)
    b1 = geometry.box(*p1, *p2, epsg4326)
    b2 = geometry.box(*p3, *p4, epsg4326)
    bb = multigeom([b1, b2])
    assert bb.type == "MultiPolygon"
    assert bb.crs is b1.crs
    assert len(list(bb)) == 2

    g1 = geometry.line([p1, p2], None)
    g2 = geometry.line([p3, p4], None)
    gg = multigeom(iter([g1, g2, g1]))
    assert gg.type == "MultiLineString"
    assert gg.crs is g1.crs
    assert len(list(gg)) == 3

    g1 = geometry.point(*p1, epsg3857)
    g2 = geometry.point(*p2, epsg3857)
    g3 = geometry.point(*p3, epsg3857)
    gg = multigeom(iter([g1, g2, g3]))
    assert gg.type == "MultiPoint"
    assert gg.crs is g1.crs
    assert len(list(gg)) == 3
    assert list(gg)[0] == g1
    assert list(gg)[1] == g2
    assert list(gg)[2] == g3

    # can't mix types
    with pytest.raises(ValueError):
        multigeom([geometry.line([p1, p2], None), geometry.point(*p1, None)])

    # can't mix CRSs
    with pytest.raises(CRSMismatchError):
        multigeom(
            [geometry.line([p1, p2], epsg4326), geometry.line([p3, p4], epsg3857)]
        )

    # only some types are supported on input
    with pytest.raises(ValueError):
        multigeom([gg])


def test_shapely_wrappers():
    poly = geometry.polygon([(0, 0), (0, 5), (10, 5)], epsg4326)

    assert isinstance(poly.svg(), str)
    assert isinstance(poly._repr_svg_(), str)

    with_hole = poly.buffer(1) - poly
    assert len(poly.interiors) == 0
    assert len(with_hole.interiors) == 1
    assert isinstance(with_hole.interiors, list)
    assert isinstance(with_hole.interiors[0], geometry.Geometry)
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
    poly = geometry.polygon([(0, 0), (0, 5), (10, 5)], epsg4326)
    num_points = 3
    assert poly.crs is epsg4326
    assert poly.to_crs(epsg3857).crs is epsg3857
    assert poly.to_crs("EPSG:3857").crs == "EPSG:3857"
    assert poly.to_crs("EPSG:3857", 0.1).crs == epsg3857

    assert poly.exterior.to_crs(epsg3857) == poly.to_crs(epsg3857).exterior

    # test that by default segmentation happens
    # +1 is because exterior loops back to start point
    assert len(poly.to_crs(epsg3857).exterior.xy[0]) > num_points + 1

    # test that +inf disables segmentation
    # +1 is because exterior loops back to start point
    assert len(poly.to_crs(epsg3857, float("+inf")).exterior.xy[0]) == num_points + 1

    # test the segmentation works on multi-polygons
    mpoly = geometry.box(0, 0, 1, 3, "EPSG:4326") | geometry.box(
        2, 4, 3, 6, "EPSG:4326"
    )

    assert mpoly.type == "MultiPolygon"
    assert mpoly.to_crs(epsg3857).type == "MultiPolygon"

    poly = geometry.polygon([(0, 0), (0, 5), (10, 5)], None)
    assert poly.crs is None
    with pytest.raises(ValueError):
        poly.to_crs(epsg3857)


def test_boundingbox():
    bb = BoundingBox(0, 3, 2, 4)
    assert bb.width == 2
    assert bb.height == 1
    assert bb.width == bb.span_x
    assert bb.height == bb.span_y

    bb = BoundingBox(0, 3, 2.1, 4)
    assert bb.width == 2
    assert bb.height == 1
    assert bb.span_x == 2.1
    assert bb.width != bb.span_x
    assert bb.height == bb.span_y

    assert BoundingBox.from_xy(bb.range_x, bb.range_y) == bb

    assert BoundingBox.from_xy((1, 2), (10, 20)) == (1, 10, 2, 20)
    assert BoundingBox.from_xy((2, 1), (20, 10)) == (1, 10, 2, 20)
    assert BoundingBox.from_points((1, 11), (2, 22)) == (1, 11, 2, 22)
    assert BoundingBox.from_points((1, 22), (2, 11)) == (1, 11, 2, 22)


def test_densify():
    from odc.geo._base import densify

    s_x10 = [(0, 0), (10, 0)]
    assert densify(s_x10, 20) == s_x10
    assert densify(s_x10, 200) == s_x10
    assert densify(s_x10, 5) == [(0, 0), (5, 0), (10, 0)]
    assert densify(s_x10, 4) == [(0, 0), (4, 0), (8, 0), (10, 0)]


def test_bbox_union():
    b1 = BoundingBox(0, 1, 10, 20)
    b2 = BoundingBox(5, 6, 11, 22)

    assert bbox_union([b1]) == b1
    assert bbox_union([b2]) == b2

    bb = bbox_union(iter([b1, b2]))
    assert bb == BoundingBox(0, 1, 11, 22)

    bb = bbox_union(iter([b2, b1] * 10))
    assert bb == BoundingBox(0, 1, 11, 22)


def test_unary_union():
    box1 = geometry.box(10, 10, 30, 30, crs=epsg4326)
    box2 = geometry.box(20, 10, 40, 30, crs=epsg4326)
    box3 = geometry.box(30, 10, 50, 30, crs=epsg4326)
    box4 = geometry.box(40, 10, 60, 30, crs=epsg4326)

    union0 = geometry.unary_union([box1])
    assert union0 == box1

    union1 = geometry.unary_union([box1, box4])
    assert union1.type == "MultiPolygon"
    assert union1.area == 2.0 * box1.area

    union2 = geometry.unary_union([box1, box2])
    assert union2.type == "Polygon"
    assert union2.area == 1.5 * box1.area

    union3 = geometry.unary_union([box1, box2, box3, box4])
    assert union3.type == "Polygon"
    assert union3.area == 2.5 * box1.area

    union4 = geometry.unary_union([union1, box2, box3])
    assert union4.type == "Polygon"
    assert union4.area == 2.5 * box1.area

    assert geometry.unary_union([]) is None

    with pytest.raises(ValueError):
        geometry.unary_union([box1, box1.to_crs(epsg3577)])


def test_unary_intersection():
    box1 = geometry.box(10, 10, 30, 30, crs=epsg4326)
    box2 = geometry.box(15, 10, 35, 30, crs=epsg4326)
    box3 = geometry.box(20, 10, 40, 30, crs=epsg4326)
    box4 = geometry.box(25, 10, 45, 30, crs=epsg4326)
    box5 = geometry.box(30, 10, 50, 30, crs=epsg4326)
    box6 = geometry.box(35, 10, 55, 30, crs=epsg4326)

    inter1 = geometry.unary_intersection([box1])
    assert bool(inter1)
    assert inter1 == box1

    inter2 = geometry.unary_intersection([box1, box2])
    assert bool(inter2)
    assert inter2.area == 300.0

    inter3 = geometry.unary_intersection([box1, box2, box3])
    assert bool(inter3)
    assert inter3.area == 200.0

    inter4 = geometry.unary_intersection([box1, box2, box3, box4])
    assert bool(inter4)
    assert inter4.area == 100.0

    inter5 = geometry.unary_intersection([box1, box2, box3, box4, box5])
    assert bool(inter5)
    assert inter5.type == "LineString"
    assert inter5.length == 20.0

    inter6 = geometry.unary_intersection([box1, box2, box3, box4, box5, box6])
    assert not bool(inter6)
    assert inter6.is_empty


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
        a = geometry.CRS(
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
    c = geometry.CRS("+proj=longlat +no_defs +ellps=GRS80")
    b = geometry.CRS(
        """GEOGCS["GRS 1980(IUGG, 1980)",DATUM["unknown",SPHEROID["GRS80",6378137,298.257222101]],
                        PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]"""
    )

    assert c.epsg is None
    assert b.epsg is None


def test_xy_from_geobox():
    gbox = GeoBox(3, 7, Affine.translation(10, 1000), epsg3857)
    xx, yy = xy_from_gbox(gbox)

    assert xx.shape == gbox.shape
    assert yy.shape == gbox.shape
    assert (xx[:, 0] == 10.5).all()
    assert (xx[:, 1] == 11.5).all()
    assert (yy[0, :] == 1000.5).all()
    assert (yy[6, :] == 1006.5).all()

    xx_, yy_, A = xy_norm(xx, yy)
    assert xx_.shape == xx.shape
    assert yy_.shape == yy.shape
    np.testing.assert_almost_equal((xx_.min(), xx_.max()), (0, 1))
    np.testing.assert_almost_equal((yy_.min(), yy_.max()), (0, 1))
    assert (xx_[0] - xx_[1]).sum() != 0
    assert (xx_[:, 0] - xx_[:, 1]).sum() != 0

    XX, YY = apply_affine(A, xx_, yy_)
    np.testing.assert_array_almost_equal(xx, XX)
    np.testing.assert_array_almost_equal(yy, YY)


def test_gen_test_image_xy():
    gbox = GeoBox(3, 7, Affine.translation(10, 1000), epsg3857)

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
    xy_ = denorm(xy, nodata=np.nan)
    assert np.isnan(xy_[:, :2]).all()
    np.testing.assert_almost_equal(xy_[0][2:], x_[2:], 6)
    np.testing.assert_almost_equal(xy_[1][2:], y_[2:], 6)

    xy, denorm = gen_test_image_xy(gbox, "int16")
    assert xy.dtype == "int16"
    assert xy.shape == (2,) + gbox.shape
    xy[0, 0, :] = -999
    xy[1, 1, :] = -999
    xy_ = denorm(xy, nodata=-999)
    assert np.isnan(xy_[:, :2]).all()
    np.testing.assert_almost_equal(xy_[0][2:], x_[2:], 4)
    np.testing.assert_almost_equal(xy_[1][2:], y_[2:], 4)

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


def test_geobox():
    points_list = [
        [
            (148.2697, -35.20111),
            (149.31254, -35.20111),
            (149.31254, -36.331431),
            (148.2697, -36.331431),
        ],
        [
            (148.2697, 35.20111),
            (149.31254, 35.20111),
            (149.31254, 36.331431),
            (148.2697, 36.331431),
        ],
        [
            (-148.2697, 35.20111),
            (-149.31254, 35.20111),
            (-149.31254, 36.331431),
            (-148.2697, 36.331431),
        ],
        [
            (-148.2697, -35.20111),
            (-149.31254, -35.20111),
            (-149.31254, -36.331431),
            (-148.2697, -36.331431),
            (148.2697, -35.20111),
        ],
    ]
    for points in points_list:
        polygon = geometry.polygon(points, crs=epsg3577)
        resolution = (-25, 25)
        geobox = geometry.GeoBox.from_geopolygon(polygon, resolution)

        assert abs(resolution[0]) > abs(
            geobox.extent.boundingbox.left - polygon.boundingbox.left
        )
        assert abs(resolution[0]) > abs(
            geobox.extent.boundingbox.right - polygon.boundingbox.right
        )
        assert abs(resolution[1]) > abs(
            geobox.extent.boundingbox.top - polygon.boundingbox.top
        )
        assert abs(resolution[1]) > abs(
            geobox.extent.boundingbox.bottom - polygon.boundingbox.bottom
        )

    A = mkA(0, scale=(10, -10), translation=(-48800, -2983006))

    w, h = 512, 256
    gbox = GeoBox(w, h, A, epsg3577)

    assert gbox.shape == (h, w)
    assert gbox.transform == A
    assert gbox.extent.crs == gbox.crs
    assert gbox.geographic_extent.crs == epsg4326
    assert gbox.extent.boundingbox.height == h * 10.0
    assert gbox.extent.boundingbox.width == w * 10.0
    assert isinstance(str(gbox), str)
    assert "EPSG:3577" in repr(gbox)

    assert GeoBox(1, 1, mkA(0), epsg4326).geographic_extent.crs == epsg4326
    assert GeoBox(1, 1, mkA(0), None).dimensions == ("y", "x")

    g2 = gbox[:-10, :-20]
    assert g2.shape == (gbox.height - 10, gbox.width - 20)

    # step of 1 is ok
    g2 = gbox[::1, ::1]
    assert g2.shape == gbox.shape

    assert gbox[0].shape == (1, gbox.width)
    assert gbox[:3].shape == (3, gbox.width)

    with pytest.raises(NotImplementedError):
        gbox[::2, :]

    # too many slices
    with pytest.raises(ValueError):
        gbox[:1, :1, :]

    assert gbox.buffered(10, 0).shape == (gbox.height + 2 * 1, gbox.width)
    assert gbox.buffered(30, 20).shape == (gbox.height + 2 * 3, gbox.width + 2 * 2)

    assert (gbox | gbox) == gbox
    assert (gbox & gbox) == gbox
    assert gbox.is_empty() is False
    assert bool(gbox) is True

    assert (gbox[:3, :4] & gbox[3:, 4:]).is_empty()
    assert (gbox[:3, :4] & gbox[30:, 40:]).is_empty()

    with pytest.raises(ValueError):
        geobox_intersection_conservative([])

    with pytest.raises(ValueError):
        geobox_union_conservative([])

    # can not combine across CRSs
    with pytest.raises(ValueError):
        bounding_box_in_pixel_domain(
            GeoBox(1, 1, mkA(0), epsg4326), GeoBox(2, 3, mkA(0), epsg3577)
        )


def test_geobox_xr_coords():
    A = mkA(0, scale=(10, -10), translation=(-48800, -2983006))

    w, h = 512, 256
    gbox = GeoBox(w, h, A, epsg3577)

    cc = gbox.xr_coords()
    assert list(cc) == ["y", "x"]
    assert cc["y"].shape == (gbox.shape[0],)
    assert cc["x"].shape == (gbox.shape[1],)
    assert "crs" in cc["y"].attrs
    assert "crs" in cc["x"].attrs

    cc = gbox.xr_coords(with_crs=True)
    assert list(cc) == ["y", "x", "spatial_ref"]
    assert cc["spatial_ref"].shape == ()
    assert cc["spatial_ref"].attrs["spatial_ref"] == gbox.crs.wkt
    assert isinstance(cc["spatial_ref"].attrs["grid_mapping_name"], str)

    cc = gbox.xr_coords(with_crs="Albers")
    assert list(cc) == ["y", "x", "Albers"]

    # geographic CRS
    A = mkA(0, scale=(0.1, -0.1), translation=(10, 30))
    gbox = GeoBox(w, h, A, "epsg:4326")
    cc = gbox.xr_coords(with_crs=True)
    assert list(cc) == ["latitude", "longitude", "spatial_ref"]
    assert cc["spatial_ref"].shape == ()
    assert cc["spatial_ref"].attrs["spatial_ref"] == gbox.crs.wkt
    assert isinstance(cc["spatial_ref"].attrs["grid_mapping_name"], str)

    # missing CRS for GeoBox
    gbox = GeoBox(w, h, A, None)
    cc = gbox.xr_coords(with_crs=True)
    assert list(cc) == ["y", "x"]

    # check CRS without name
    crs = MagicMock()
    crs.projected = True
    crs.wkt = epsg3577.wkt
    crs.epsg = epsg3577.epsg
    crs._crs = MagicMock()
    crs._crs.to_cf.return_value = {}
    assert _mk_crs_coord(crs).attrs["grid_mapping_name"] == "??"


def test_projected_lon():
    assert projected_lon(epsg3857, 180).crs is epsg3857
    assert projected_lon("EPSG:3577", 100).crs == epsg3577


def test_chop():
    poly = geometry.box(618300, -1876800, 849000, -1642500, "EPSG:32660")

    chopped = chop_along_antimeridian(poly)
    assert chopped.crs is poly.crs
    assert chopped.type == "MultiPolygon"
    assert len([g for g in chopped]) == 2

    poly = geometry.box(0, 0, 10, 20, "EPSG:4326")._to_crs(epsg3857)
    assert poly.crs is epsg3857
    assert chop_along_antimeridian(poly) is poly

    with pytest.raises(ValueError):
        chop_along_antimeridian(geometry.box(0, 1, 2, 3, None))


def test_clip_lon180():
    err = 1e-9

    def b(rside):
        return geometry.box(170, 0, rside, 10, epsg4326)

    def b_neg(lside):
        return geometry.box(lside, 0, -170, 10, epsg4326)

    assert clip_lon180(b(180 - err)) == b(180)
    assert clip_lon180(b(-180 + err)) == b(180)

    assert clip_lon180(b_neg(180 - err)) == b_neg(-180)
    assert clip_lon180(b_neg(-180 + err)) == b_neg(-180)

    bb = multigeom([b(180 - err), b_neg(180 - err)])
    bb_ = [g for g in clip_lon180(bb)]
    assert bb_[0] == b(180)
    assert bb_[1] == b_neg(-180)


def test_wrap_dateline():
    albers_crs = epsg3577
    geog_crs = epsg4326

    wrap = geometry.polygon(
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
    assert wrapped.type == "Polygon"
    assert wrapped.intersects(geometry.line([(0, -90), (0, 90)], crs=geog_crs))
    wrapped = wrap.to_crs(geog_crs, wrapdateline=True)
    assert wrapped.type == "MultiPolygon"
    assert not wrapped.intersects(geometry.line([(0, -90), (0, 90)], crs=geog_crs))


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
    sinus_crs = geometry.CRS(
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

    wrap = geometry.polygon(pts, crs=sinus_crs)
    wrapped = wrap.to_crs(epsg4326)
    assert wrapped.type == "Polygon"
    wrapped = wrap.to_crs(epsg4326, wrapdateline=True)
    assert wrapped.type == "MultiPolygon"
    assert not wrapped.intersects(geometry.line([(0, -90), (0, 90)], crs=epsg4326))


def test_wrap_dateline_utm():
    poly = geometry.box(618300, -1876800, 849000, -1642500, "EPSG:32660")

    wrapped = poly.to_crs(epsg4326)
    assert wrapped.type == "Polygon"
    assert wrapped.intersects(geometry.line([(0, -90), (0, 90)], crs=epsg4326))
    wrapped = poly.to_crs(epsg4326, wrapdateline=True)
    assert wrapped.type == "MultiPolygon"
    assert not wrapped.intersects(geometry.line([(0, -90), (0, 90)], crs=epsg4326))


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

    g_2d = geometry.Geometry(geom_2d)
    g_3d = geometry.Geometry(geom_3d)

    assert {2} == {len(pt) for pt in g_3d.boundary.coords}  # All coordinates are 2D

    assert g_2d == g_3d  # 3D geometry has been converted to a 2D by dropping the Z axis


def test_3d_point_converted_to_2d_point():
    point = (-35.5029340, 145.9312455, 0.0)

    point_3d = {"coordinates": point, "type": "Point"}
    point_2d = {"coordinates": (point[0], point[1]), "type": "Point"}

    p_2d = geometry.Geometry(point_2d)
    p_3d = geometry.Geometry(point_3d)

    assert len(p_3d.coords[0]) == 2

    assert p_2d == p_3d


def test_crs():
    CRS = geometry.CRS
    custom_crs = geometry.CRS(
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

    crs2 = CRS(crs)
    assert crs2 == crs
    assert crs.proj is crs2.proj

    assert epsg4326.valid_region == geometry.box(-180, -90, 180, 90, epsg4326)
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
        with pytest.raises(geometry.CRSError):
            CRS(bad)

    with pytest.warns(DeprecationWarning):
        assert str(epsg3857) == epsg3857.crs_str


def test_polygon_path():
    from odc.geo.tools import polygon_path

    pp = polygon_path([0, 1])
    assert pp.shape == (2, 5)
    assert set(pp.ravel()) == {0, 1}

    pp2 = polygon_path([0, 1], [0, 1])
    assert (pp2 == pp).all()

    pp = polygon_path([0, 1], [2, 3])
    assert set(pp[0].ravel()) == {0, 1}
    assert set(pp[1].ravel()) == {2, 3}


def test_gbox_boundary():
    xx = np.zeros((2, 6))

    bb = gbox_boundary(xx, 3)

    assert bb.shape == (4 + (3 - 2) * 4, 2)
    assert set(bb.T[0]) == {0.0, 3.0, 6.0}
    assert set(bb.T[1]) == {0.0, 1.0, 2.0}


def test_geobox_scale_down():

    crs = CRS("EPSG:3857")

    A = mkA(0, (111.2, 111.2), translation=(125671, 251465))
    for s in [2, 3, 4, 8, 13, 16]:
        gbox = GeoBox(233 * s, 755 * s, A, crs)
        gbox_ = scaled_down_geobox(gbox, s)

        assert gbox_.width == 233
        assert gbox_.height == 755
        assert gbox_.crs is crs
        assert gbox_.extent.contains(gbox.extent)
        assert gbox.extent.difference(gbox.extent).area == 0.0

    gbox = GeoBox(1, 1, A, crs)
    for s in [2, 3, 5]:
        gbox_ = scaled_down_geobox(gbox, 3)

        assert gbox_.shape == (1, 1)
        assert gbox_.crs is crs
        assert gbox_.extent.contains(gbox.extent)


def test_roi_tools():
    from numpy import s_

    from odc.geo import (
        roi_boundary,
        roi_center,
        roi_from_points,
        roi_intersect,
        roi_is_empty,
        roi_is_full,
        roi_normalise,
        roi_pad,
        roi_shape,
        scaled_down_roi,
        scaled_down_shape,
        scaled_up_roi,
    )

    assert roi_shape(s_[2:4, 3:4]) == (2, 1)
    assert roi_shape(s_[:4, :7]) == (4, 7)

    assert roi_is_empty(s_[:4, :5]) is False
    assert roi_is_empty(s_[1:1, :10]) is True
    assert roi_is_empty(s_[7:3, :10]) is True

    assert roi_is_empty(s_[:3]) is False
    assert roi_is_empty(s_[4:4]) is True

    assert roi_is_full(s_[:3], 3) is True
    assert roi_is_full(s_[:3, 0:4], (3, 4)) is True
    assert roi_is_full(s_[:, 0:4], (33, 4)) is True
    assert roi_is_full(s_[1:3, 0:4], (3, 4)) is False
    assert roi_is_full(s_[1:3, 0:4], (2, 4)) is False
    assert roi_is_full(s_[0:4, 0:4], (3, 4)) is False

    roi = s_[0:8, 0:4]
    roi_ = scaled_down_roi(roi, 2)
    assert roi_shape(roi_) == (4, 2)
    assert scaled_down_roi(scaled_up_roi(roi, 3), 3) == roi

    assert scaled_down_shape(roi_shape(roi), 2) == roi_shape(scaled_down_roi(roi, 2))

    assert roi_shape(scaled_up_roi(roi, 10000, (40, 50))) == (40, 50)

    assert roi_normalise(s_[3:4], 40) == s_[3:4]
    assert roi_normalise(s_[:4], (40,)) == s_[0:4]
    assert roi_normalise(s_[:], (40,)) == s_[0:40]
    assert roi_normalise(s_[:-1], (3,)) == s_[0:2]
    assert roi_normalise(s_[-2:-1, :], (10, 20)) == s_[8:9, 0:20]
    assert roi_normalise(s_[-2:-1, :, 3:4], (10, 20, 100)) == s_[8:9, 0:20, 3:4]
    assert roi_center(s_[0:3]) == 1.5
    assert roi_center(s_[0:2, 0:6]) == (1, 3)

    roi = s_[0:2, 4:13]
    xy = roi_boundary(roi)

    assert xy.shape == (4, 2)
    assert roi_from_points(xy, (2, 13)) == roi

    assert roi_intersect(roi, roi) == roi
    assert roi_intersect(s_[0:3], s_[1:7]) == s_[1:3]
    assert roi_intersect(s_[0:3], (s_[1:7],)) == s_[1:3]
    assert roi_intersect((s_[0:3],), s_[1:7]) == (s_[1:3],)

    assert roi_intersect(s_[4:7, 5:6], s_[0:1, 7:8]) == s_[4:4, 6:6]

    assert roi_pad(s_[0:4], 1, 4) == s_[0:4]
    assert roi_pad(s_[0:4, 1:5], 1, (4, 6)) == s_[0:4, 0:6]
    assert roi_pad(s_[2:3, 1:5], 10, (7, 9)) == s_[0:7, 0:9]


def test_apply_affine():
    A = mkA(rot=10, scale=(3, 1.3), translation=(-100, +2.3))
    xx, yy = np.meshgrid(np.arange(13), np.arange(11))

    xx_, yy_ = apply_affine(A, xx, yy)

    assert xx_.shape == xx.shape
    assert yy_.shape == xx.shape

    xy_expect = [A * (x, y) for x, y in zip(xx.ravel(), yy.ravel())]
    xy_got = [(x, y) for x, y in zip(xx_.ravel(), yy_.ravel())]

    np.testing.assert_array_almost_equal(xy_expect, xy_got)


def test_point_transformer():
    from odc.geo import point

    tr = epsg3857.transformer_to_crs(epsg4326)
    tr_back = epsg4326.transformer_to_crs(epsg3857)

    pts = [(0, 0), (0, 1), (1, 2), (10, 11)]
    x, y = np.vstack(pts).astype("float64").T

    pts_expect = [point(*pt, epsg3857).to_crs(epsg4326).points[0] for pt in pts]

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


def test_split_translation():
    def verify(a, b):
        a = np.asarray(a)
        b = np.asarray(b)
        np.testing.assert_array_almost_equal(a, b)

    def tt(tx, ty, *expect):
        verify(split_translation((tx, ty)), expect)

    assert split_translation((1, 2)) == ((1, 2), (0, 0))
    assert split_translation((-1, -2)) == ((-1, -2), (0, 0))
    tt(1.3, 2.5, (1, 2), (0.3, 0.5))
    tt(1.1, 2.6, (1, 3), (0.1, -0.4))
    tt(-1.1, 2.8, (-1, 3), (-0.1, -0.2))
    tt(-1.9, 2.05, (-2, 2), (+0.1, 0.05))
    tt(-1.5, 2.45, (-1, 2), (-0.5, 0.45))


def get_diff(A, B):
    from math import sqrt

    return sqrt(sum((a - b) ** 2 for a, b in zip(A, B)))


def test_affine_checks():
    assert is_affine_st(mkA(scale=(1, 2), translation=(3, -10))) is True
    assert is_affine_st(mkA(scale=(1, -2), translation=(-3, -10))) is True
    assert is_affine_st(mkA(rot=0.1)) is False
    assert is_affine_st(mkA(shear=0.4)) is False


def test_affine_rsw():
    def run_test(a, scale, shear=0, translation=(0, 0), tol=1e-8):
        A = mkA(a, scale=scale, shear=shear, translation=translation)

        R, W, S = decompose_rws(A)

        assert get_diff(A, R * W * S) < tol
        assert get_diff(S, mkA(0, scale)) < tol
        assert get_diff(R, mkA(a, translation=translation)) < tol

    for a in (0, 12, 45, 33, 67, 89, 90, 120, 170):
        run_test(a, (1, 1))
        run_test(a, (0.5, 2))
        run_test(-a, (0.5, 2))

        run_test(a, (1, 2))
        run_test(-a, (1, 2))

        run_test(a, (2, -1))
        run_test(-a, (2, -1))

    run_test(0, (3, 4), 10)
    run_test(-33, (3, -1), 10, translation=(100, -333))


def test_fit():
    from random import uniform

    def run_test(A, n, tol=1e-5):
        X = [(uniform(0, 1), uniform(0, 1)) for _ in range(n)]
        Y = [A * x for x in X]
        A_ = affine_from_pts(X, Y)

        assert get_diff(A, A_) < tol

    A = mkA(13, scale=(3, 4), shear=3, translation=(100, -3000))

    run_test(A, 3)
    run_test(A, 10)

    run_test(mkA(), 3)
    run_test(mkA(), 10)


def test_scale_at_point():
    def mk_transform(sx, sy):
        A = mkA(37, scale=(sx, sy), translation=(2127, 93891))

        def transofrom(pts):
            return [A * x for x in pts]

        return transofrom

    tol = 1e-4
    pt = (0, 0)
    for sx, sy in [(3, 4), (0.4, 0.333)]:
        tr = mk_transform(sx, sy)
        sx_, sy_ = get_scale_at_point(pt, tr)
        assert abs(sx - sx_) < tol
        assert abs(sy - sy_) < tol

        sx_, sy_ = get_scale_at_point(pt, tr, 0.1)
        assert abs(sx - sx_) < tol
        assert abs(sy - sy_) < tol


def test_pix_transform():
    pt = tuple(
        int(x / 10) * 10
        for x in geometry.point(145, -35, epsg4326).to_crs(epsg3577).coords[0]
    )

    A = mkA(scale=(20, -20), translation=pt)

    src = geometry.GeoBox(1024, 512, A, epsg3577)
    dst = geometry.GeoBox.from_geopolygon(src.geographic_extent, (0.0001, -0.0001))

    tr = native_pix_transform(src, dst)

    pts_src = [(0, 0), (10, 20), (300, 200)]
    pts_dst = tr(pts_src)
    pts_src_ = tr.back(pts_dst)

    np.testing.assert_almost_equal(pts_src, pts_src_)
    assert tr.linear is None

    # check identity transform
    tr = native_pix_transform(src, src)

    pts_src = [(0, 0), (10, 20), (300, 200)]
    pts_dst = tr(pts_src)
    pts_src_ = tr.back(pts_dst)

    np.testing.assert_almost_equal(pts_src, pts_src_)
    np.testing.assert_almost_equal(pts_src, pts_dst)
    assert tr.linear is not None
    assert tr.back.linear is not None
    assert tr.back.back is tr

    # check scale only change
    tr = native_pix_transform(src, scaled_down_geobox(src, 2))
    pts_dst = tr(pts_src)
    pts_src_ = tr.back(pts_dst)

    assert tr.linear is not None
    assert tr.back.linear is not None
    assert tr.back.back is tr

    np.testing.assert_almost_equal(pts_dst, [(x / 2, y / 2) for (x, y) in pts_src])

    np.testing.assert_almost_equal(pts_src, pts_src_)


def test_compute_reproject_roi():
    src = AlbersGS.tile_geobox((15, -40))
    dst = geometry.GeoBox.from_geopolygon(
        src.extent.to_crs(epsg3857).buffer(10), resolution=src.resolution
    )

    rr = compute_reproject_roi(src, dst)

    assert rr.roi_src == np.s_[0 : src.height, 0 : src.width]
    assert 0 < rr.scale < 1
    assert rr.is_st is False
    assert rr.transform.linear is None
    assert rr.scale in rr.scale2

    # check pure translation case
    roi_ = np.s_[113:-100, 33:-10]
    rr = compute_reproject_roi(src, src[roi_])
    assert rr.roi_src == roi_normalise(roi_, src.shape)
    assert rr.scale == 1
    assert rr.is_st is True

    rr = compute_reproject_roi(src, src[roi_], padding=0, align=0)
    assert rr.roi_src == roi_normalise(roi_, src.shape)
    assert rr.scale == 1
    assert rr.scale2 == (1, 1)

    # check pure translation case
    roi_ = np.s_[113:-100, 33:-10]
    rr = compute_reproject_roi(src, src[roi_], align=256)

    assert rr.roi_src == np.s_[0 : src.height, 0 : src.width]
    assert rr.scale == 1

    roi_ = np.s_[113:-100, 33:-10]
    rr = compute_reproject_roi(src, src[roi_])

    assert rr.scale == 1
    assert roi_shape(rr.roi_src) == roi_shape(rr.roi_dst)
    assert roi_shape(rr.roi_dst) == src[roi_].shape


def test_compute_reproject_roi_issue647():
    """In some scenarios non-overlapping geoboxes will result in non-empty
    `roi_dst` even though `roi_src` is empty.

    Test this case separately.
    """

    src = GeoBox(
        10980, 10980, Affine(10, 0, 300000, 0, -10, 5900020), CRS("epsg:32756")
    )

    dst = GeoBox(976, 976, Affine(10, 0, 1730240, 0, -10, -4170240), CRS("EPSG:3577"))

    assert src.extent.overlaps(dst.extent.to_crs(src.crs)) is False

    rr = compute_reproject_roi(src, dst)

    assert roi_is_empty(rr.roi_src)
    assert roi_is_empty(rr.roi_dst)


def test_compute_reproject_roi_issue1047():
    """`compute_reproject_roi(geobox, geobox[roi])` sometimes returns
    `src_roi != roi`, when `geobox` has (1) tiny pixels and (2) oddly
    sized `alignment`.

    Test this issue is resolved.
    """
    geobox = GeoBox(
        3000,
        3000,
        Affine(
            0.00027778, 0.0, 148.72673054908861, 0.0, -0.00027778, -34.98825802556622
        ),
        "EPSG:4326",
    )
    src_roi = np.s_[2800:2810, 10:30]
    rr = compute_reproject_roi(geobox, geobox[src_roi])

    assert rr.is_st is True
    assert rr.roi_src == src_roi
    assert rr.roi_dst == np.s_[0:10, 0:20]


def test_window_from_slice():
    from numpy import s_

    assert w_[None] is None
    assert w_[s_[:3, 4:5]] == ((0, 3), (4, 5))
    assert w_[s_[0:3, :5]] == ((0, 3), (0, 5))
    assert w_[list(s_[0:3, :5])] == ((0, 3), (0, 5))

    for roi in [s_[:3], s_[:3, :4, :5], 0]:
        with pytest.raises(ValueError):
            w_[roi]


def test_axis_overlap():
    s_ = np.s_

    # Source overlaps destination fully
    #
    # S: |<--------------->|
    # D:      |<----->|
    assert compute_axis_overlap(100, 20, 1, 10) == s_[10:30, 0:20]
    assert compute_axis_overlap(100, 20, 2, 10) == s_[10:50, 0:20]
    assert compute_axis_overlap(100, 20, 0.25, 10) == s_[10:15, 0:20]
    assert compute_axis_overlap(100, 20, -1, 80) == s_[60:80, 0:20]
    assert compute_axis_overlap(100, 20, -0.5, 50) == s_[40:50, 0:20]
    assert compute_axis_overlap(100, 20, -2, 90) == s_[50:90, 0:20]

    # Destination overlaps source fully
    #
    # S:      |<-------->|
    # D: |<----------------->|
    assert compute_axis_overlap(10, 100, 1, -10) == s_[0:10, 10:20]
    assert compute_axis_overlap(10, 100, 2, -10) == s_[0:10, 5:10]
    assert compute_axis_overlap(10, 100, 0.5, -10) == s_[0:10, 20:40]
    assert compute_axis_overlap(10, 100, -1, 11) == s_[0:10, 1:11]

    # Partial overlaps
    #
    # S: |<----------->|
    # D:     |<----------->|
    assert compute_axis_overlap(10, 10, 1, 3) == s_[3:10, 0:7]
    assert compute_axis_overlap(10, 15, 1, 3) == s_[3:10, 0:7]

    # S:     |<----------->|
    # D: |<----------->|
    assert compute_axis_overlap(10, 10, 1, -5) == s_[0:5, 5:10]
    assert compute_axis_overlap(50, 10, 1, -5) == s_[0:5, 5:10]

    # No overlaps
    # S: |<--->|
    # D:         |<--->|
    assert compute_axis_overlap(10, 10, 1, 11) == s_[10:10, 0:0]
    assert compute_axis_overlap(10, 40, 1, 11) == s_[10:10, 0:0]

    # S:         |<--->|
    # D: |<--->|
    assert compute_axis_overlap(10, 10, 1, -11) == s_[0:0, 10:10]
    assert compute_axis_overlap(40, 10, 1, -11) == s_[0:0, 10:10]


def test_crs_compat():
    import rasterio.crs

    crs = CRS("epsg:3577")
    assert crs.epsg == 3577
    crs2 = CRS(crs)
    assert crs.epsg == crs2.epsg

    crs_rio = rasterio.crs.CRS(init="epsg:3577")
    assert CRS(crs_rio).epsg == 3577

    assert (CRS(crs_rio) == crs_rio) is True

    with pytest.raises(geometry.CRSError):
        CRS(("random", "tuple"))

    crs = CRS("epsg:3857")
    with pytest.warns(UserWarning):
        crs_dict = crs.proj.to_dict()

    assert CRS(crs_dict) == crs


def test_crs_hash():
    crs = CRS("epsg:3577")
    crs2 = CRS(crs)

    assert crs is not crs2
    assert len({crs, crs2}) == 1


def test_base_internals():
    assert _guess_crs_str(CRS("epsg:3577")) == "EPSG:3577"
    no_epsg_crs = CRS(SAMPLE_WKT_WITHOUT_AUTHORITY)
    assert _guess_crs_str(no_epsg_crs) == no_epsg_crs.to_wkt()

    gjson_bad = {"type": "a", "coordinates": [1, [2, 3, 4]]}
    assert force_2d(gjson_bad) == {"type": "a", "coordinates": [1, [2, 3]]}

    with pytest.raises(ValueError):
        force_2d({"type": "a", "coordinates": [set("not a valid element")]})

    assert _round_to_res(0.2, 1.0) == 1
    assert _round_to_res(0.0, 1.0) == 0
    assert _round_to_res(0.05, 1.0) == 0

    assert _norm_crs(None) is None

    with pytest.raises(ValueError):
        _norm_crs_or_error(None)


def test_geom_clone():
    b = geometry.box(0, 0, 10, 20, epsg4326)
    assert b == b.clone()
    assert b.geom is not b.clone().geom

    assert b == geometry.Geometry(b)
    assert b.geom is not geometry.Geometry(b).geom


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


@pytest.mark.parametrize(
    "left, right, off, res, expect",
    [
        (20, 30, 10, 0, (20, 1)),
        (20, 30.5, 10, 0, (20, 1)),
        (20, 31.5, 10, 0, (20, 2)),
        (20, 30, 10, 3, (13, 2)),
        (20, 30, 10, -3, (17, 2)),
        (20, 30, -10, 0, (30, 1)),
        (19.5, 30, -10, 0, (30, 1)),
        (18.5, 30, -10, 0, (30, 2)),
        (20, 30, -10, 3, (33, 2)),
        (20, 30, -10, -3, (37, 2)),
    ],
)
def test_align_pix(left, right, off, res, expect):
    assert _align_pix(left, right, off, res) == expect


def test_lonlat_bounds():
    # example from landsat scene: spans lon=180
    poly = geometry.box(618300, -1876800, 849000, -1642500, "EPSG:32660")

    bb = geometry.lonlat_bounds(poly)
    assert bb.left < 180 < bb.right
    assert geometry.lonlat_bounds(poly) == geometry.lonlat_bounds(poly, resolution=1e8)

    bb = geometry.lonlat_bounds(poly, mode="quick")
    assert bb.right - bb.left > 180

    poly = geometry.box(1, -10, 2, 20, "EPSG:4326")
    assert geometry.lonlat_bounds(poly) == poly.boundingbox

    with pytest.raises(ValueError):
        geometry.lonlat_bounds(geometry.box(0, 0, 1, 1, None))

    multi = {
        "type": "MultiPolygon",
        "coordinates": [
            [[[174, 52], [174, 53], [175, 53], [174, 52]]],
            [[[168, 54], [167, 55], [167, 54], [168, 54]]],
        ],
    }

    multi_geom = geometry.Geometry(multi, "epsg:4326")
    multi_geom_projected = multi_geom.to_crs("epsg:32659", math.inf)

    ll_bounds = geometry.lonlat_bounds(multi_geom)
    ll_bounds_projected = geometry.lonlat_bounds(multi_geom_projected)

    assert ll_bounds == approx(ll_bounds_projected)


@pytest.mark.xfail(
    True, reason="Bounds computation for large geometries in safe mode is broken"
)
def test_lonalt_bounds_more_than_180():
    poly = geometry.box(-150, -30, 150, 30, epsg4326).to_crs(epsg3857, math.inf)

    assert geometry.lonlat_bounds(poly, "quick") == approx((-150, -30, 150, 30))
    assert geometry.lonlat_bounds(poly, "safe") == approx((-150, -30, 150, 30))
