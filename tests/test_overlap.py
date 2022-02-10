# This file is part of the Open Data Cube, see https://opendatacube.org for more information
#
# Copyright (c) 2015-2020 ODC Contributors
# SPDX-License-Identifier: Apache-2.0
import math
from random import uniform

import numpy as np
from affine import Affine

from odc.geo import CRS, geom, resyx_, xy_
from odc.geo._overlap import (
    LinearPointTransform,
    affine_from_pts,
    compute_axis_overlap,
    compute_reproject_roi,
    decompose_rws,
    get_scale_at_point,
    native_pix_transform,
    stack_xy,
)
from odc.geo.geobox import GeoBox, scaled_down_geobox
from odc.geo.math import is_affine_st
from odc.geo.roi import roi_is_empty, roi_normalise, roi_shape
from odc.geo.testutils import AlbersGS, epsg3577, epsg3857, epsg4326, mkA


def diff_affine(A: Affine, B: Affine) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(A, B)))


def test_affine_checks():
    assert is_affine_st(mkA(scale=(1, 2), translation=(3, -10))) is True
    assert is_affine_st(mkA(scale=(1, -2), translation=(-3, -10))) is True
    assert is_affine_st(mkA(rot=0.1)) is False
    assert is_affine_st(mkA(shear=0.4)) is False


def test_affine_rsw():
    def run_test(a, scale, shear=0, translation=(0, 0), tol=1e-8):
        A = mkA(a, scale=scale, shear=shear, translation=translation)

        R, W, S = decompose_rws(A)

        assert diff_affine(A, R * W * S) < tol
        assert diff_affine(S, mkA(0, scale)) < tol
        assert diff_affine(R, mkA(a, translation=translation)) < tol

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
    def run_test(A, n, tol=1e-5):
        X = [xy_(uniform(0, 1), uniform(0, 1)) for _ in range(n)]
        Y = [xy_(A * pt.xy) for pt in X]
        A_ = affine_from_pts(X, Y)

        assert diff_affine(A, A_) < tol

    A = mkA(13, scale=(3, 4), shear=3, translation=(100, -3000))

    run_test(A, 3)
    run_test(A, 10)

    run_test(mkA(), 3)
    run_test(mkA(), 10)


def test_scale_at_point():
    def mk_transform(sx, sy):
        A = mkA(37, scale=(sx, sy), translation=(2127, 93891))
        return LinearPointTransform(A)

    tol = 1e-4
    pt = xy_(0, 0)
    for sx, sy in [(3, 4), (0.4, 0.333)]:
        tr = mk_transform(sx, sy)
        sx_, sy_ = get_scale_at_point(pt, tr).xy
        assert abs(sx - sx_) < tol
        assert abs(sy - sy_) < tol

        sx_, sy_ = get_scale_at_point(pt, tr, 0.1).xy
        assert abs(sx - sx_) < tol
        assert abs(sy - sy_) < tol


def test_pix_transform():
    pt = tuple(
        int(x / 10) * 10
        for x in geom.point(145, -35, epsg4326).to_crs(epsg3577).coords[0]
    )

    A = mkA(scale=(20, -20), translation=pt)

    src = GeoBox((512, 1024), A, epsg3577)
    dst = GeoBox.from_geopolygon(src.geographic_extent, resyx_(0.0001, -0.0001))

    tr = native_pix_transform(src, dst)

    pts_src = [xy_(0, 0), xy_(10, 20), xy_(300, 200)]
    pts_dst = tr(pts_src)
    pts_src_ = tr.back(pts_dst)

    np.testing.assert_almost_equal(stack_xy(pts_src), stack_xy(pts_src_))
    assert tr.linear is None

    # check identity transform
    tr = native_pix_transform(src, src)

    pts_src = [xy_(0, 0), xy_(10, 20), xy_(300, 200)]
    pts_dst = tr(pts_src)
    pts_src_ = tr.back(pts_dst)

    np.testing.assert_almost_equal(stack_xy(pts_src), stack_xy(pts_src_))
    np.testing.assert_almost_equal(stack_xy(pts_src), stack_xy(pts_dst))
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

    np.testing.assert_almost_equal(
        stack_xy(pts_dst), [(pt.x / 2, pt.y / 2) for pt in pts_src]
    )

    np.testing.assert_almost_equal(stack_xy(pts_src), stack_xy(pts_src_))


def test_compute_reproject_roi():
    src = AlbersGS.tile_geobox((15, -40))
    dst = GeoBox.from_geopolygon(
        src.extent.to_crs(epsg3857).buffer(10), resolution=src.resolution
    )

    rr = compute_reproject_roi(src, dst)

    assert rr.roi_src == np.s_[0 : src.height, 0 : src.width]
    assert 0 < rr.scale < 1
    assert rr.is_st is False
    assert rr.transform.linear is None
    assert rr.scale in rr.scale2.xy
    assert rr.transform.back is not None
    assert rr.transform.back.linear is None

    # check pure translation case
    roi_ = np.s_[113:-100, 33:-10]
    rr = compute_reproject_roi(src, src[roi_])
    assert rr.roi_src == roi_normalise(roi_, src.shape)
    assert rr.scale == 1
    assert rr.is_st is True

    rr = compute_reproject_roi(src, src[roi_], padding=0, align=0)
    assert rr.roi_src == roi_normalise(roi_, src.shape)
    assert rr.scale == 1
    assert rr.scale2.xy == (1, 1)

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
        (10980, 10980), Affine(10, 0, 300000, 0, -10, 5900020), CRS("epsg:32756")
    )

    dst = GeoBox((976, 976), Affine(10, 0, 1730240, 0, -10, -4170240), CRS("EPSG:3577"))

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
        (3000, 3000),
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
