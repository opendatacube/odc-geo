# This file is part of the Open Data Cube, see https://opendatacube.org for more information
#
# Copyright (c) 2015-2020 ODC Contributors
# SPDX-License-Identifier: Apache-2.0
import math
from random import uniform

import numpy as np
import pytest
from affine import Affine

from odc.geo import CRS, geom, res_, resyx_, wh_, xy_
from odc.geo.geobox import GeoBox, scaled_down_geobox
from odc.geo.gridspec import GridSpec
from odc.geo.math import affine_from_pts, decompose_rws, is_affine_st, stack_xy
from odc.geo.overlap import (
    LinearPointTransform,
    ReprojectInfo,
    _can_paste,
    compute_axis_overlap,
    compute_output_geobox,
    compute_reproject_roi,
    get_scale_at_point,
    native_pix_transform,
)
from odc.geo.roi import (
    roi_is_empty,
    roi_normalise,
    roi_shape,
    scaled_down_roi,
    scaled_up_roi,
)
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
    assert repr(tr).startswith("GbxPointTransform(")

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
    assert repr(tr).startswith("LinearPointTransform(")

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
    assert all(0 < scale < 1 for scale in rr.scale2.xy)
    assert rr.transform.linear is None
    assert rr.transform.back is not None
    assert rr.transform.back.linear is None

    # check pure translation case
    roi_ = np.s_[113:-100, 33:-10]
    rr = compute_reproject_roi(src, src[roi_])
    assert rr.roi_src == roi_normalise(roi_, src.shape)
    assert rr.scale == 1

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

    # check pasteable zoom_out
    dst = src.zoom_out(2)
    rr = compute_reproject_roi(src, dst)
    assert rr.paste_ok is True
    assert rr.read_shrink == 2
    assert roi_shape(rr.roi_src) == src.shape
    assert roi_shape(rr.roi_dst) == dst.shape

    # check non-pasteable zoom_out
    dst = src[1:, :].zoom_out(2)
    rr = compute_reproject_roi(src, dst)
    assert rr.paste_ok is False
    assert rr.read_shrink == 2


def test_compute_reproject_roi_paste():
    src = GeoBox(
        wh_(1000, 2000),
        mkA(scale=(10, -10), translation=(10 * 123, -10 * 230)),
        epsg3857,
    )

    def _check(src: GeoBox, dst: GeoBox, rr: ReprojectInfo):
        assert rr.read_shrink >= 1

        if roi_is_empty(rr.roi_src):
            assert roi_is_empty(rr.roi_dst)
            return

        if rr.paste_ok:
            if rr.read_shrink == 1:
                assert roi_shape(rr.roi_src) == roi_shape(rr.roi_dst)
                assert src[rr.roi_src].shape == dst[rr.roi_dst].shape
            else:
                # roi source must align to read scale
                # => round-triping roi to overview and back should not change roi
                assert (
                    scaled_up_roi(
                        scaled_down_roi(rr.roi_src, rr.read_shrink), rr.read_shrink
                    )
                    == rr.roi_src
                )

                src_ = src[rr.roi_src].zoom_out(rr.read_shrink)
                assert src_.shape == dst[rr.roi_dst].shape

        if rr.read_shrink == 1:
            assert rr.scale <= 1.1
        else:
            assert rr.scale >= rr.read_shrink

        if src.crs == dst.crs:
            _src = src[rr.roi_src].extent
            _dst = dst[rr.roi_dst].extent
        else:
            _src = src[rr.roi_src].geographic_extent
            _dst = dst[rr.roi_dst].geographic_extent

        assert _src.intersection(_dst).area > 0

    def _yes(src: GeoBox, dst: GeoBox, **kw):
        rr = compute_reproject_roi(src, dst, **kw)
        assert rr.paste_ok is True
        _check(src, dst, rr)

    def _no_(src: GeoBox, dst: GeoBox, **kw):
        rr = compute_reproject_roi(src, dst, **kw)
        assert rr.paste_ok is False
        _check(src, dst, rr)

    t_ = Affine.translation
    s_ = Affine.scale

    # plain pixel aligned translation
    _yes(src, src)
    _yes(src, src[10:, 29:])
    _yes(src[10:, 29:], src)

    # subpixel translation below threshhold
    _no_(src, src * t_(0.3, 0.3))
    _yes(src, src * t_(0.3, 0.3), ttol=0.5)
    _no_(src, src * t_(0.0, 0.1))
    _yes(src, src * t_(0.0, 0.1), ttol=0.15)
    _no_(src, src * t_(-0.1, 0.0))
    _yes(src, src * t_(-0.1, 0.0), ttol=0.15)

    # tiny scale deviations
    _no_(src, src[20:, :30] * s_(1.003, 1.003))
    _yes(src, src[20:, :30] * s_(1.003, 1.003), stol=0.01)

    # integer shrink
    _no_(src, src.zoom_out(2.3))
    _yes(src, src.zoom_out(2))
    _yes(src, src.zoom_out(3))
    _yes(src, src.zoom_out(2 + 1e-5))  # rounding issues should not matter
    _no_(src.zoom_out(3), src)
    _no_(src.zoom_out(2), src)

    # integer shrink but with sub-pixel translation after shrinking
    _yes(src, src[4:, 8:].zoom_out(4))
    _no_(src, src[2:, 8:].zoom_out(4))
    _no_(src, src[8:, 3:].zoom_out(4))
    _yes(src, src[8:, 3:].zoom_out(4), ttol=0.5)


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

    assert rr.paste_ok is True
    assert rr.roi_src == src_roi
    assert rr.roi_dst == np.s_[0:10, 0:20]


def test_compute_reproject_roi_overhang():
    """
    Images with global coverage in epsg:4326 often have slightly
    wrong georegistration that causes image boundaries to reach outside
    of the [-180, -90, 180, 90] bounding box.

    Reproject roi introduces clipping to deal with that issue.
    """
    tol = 1e-3
    src_geobox = GeoBox.from_bbox(
        (-180 - tol, -90 - tol, 180 + tol, 90 + tol),
        epsg4326,
        shape=wh_(2000, 1000),
        tight=True,
    )
    assert src_geobox.shape.wh == (2000, 1000)
    assert src_geobox.extent.boundingbox[0] < -180
    assert src_geobox.extent.boundingbox[1] < -90
    assert src_geobox.extent.boundingbox[2] > +180
    assert src_geobox.extent.boundingbox[3] > +90

    dst_geobox = GridSpec.web_tiles(0)[0, 0]

    rr = compute_reproject_roi(src_geobox, dst_geobox)
    assert rr.paste_ok is False
    assert dst_geobox[rr.roi_dst] == dst_geobox


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


def test_can_paste():
    assert _can_paste(mkA(translation=(10, -20))) == (True, None)
    assert _can_paste(mkA(scale=(10, 10))) == (True, None)
    assert _can_paste(mkA(scale=(-10, 10), translation=(0, -4 * 10))) == (True, None)

    assert _can_paste(mkA(shear=0.3)) == (False, "has rotation or shear")
    assert _can_paste(mkA(rot=30)) == (False, "has rotation or shear")

    assert _can_paste(mkA(scale=(-11.1, 11.1))) == (False, "non-integer scale")
    assert _can_paste(mkA(scale=(0.5, 0.5))) == (False, "non-integer scale")
    assert _can_paste(mkA(scale=(2, 3))) == (False, "sx!=sy, probably")

    assert _can_paste(mkA(scale=(-10, 10), translation=(0, -4))) == (
        False,
        "sub-pixel translation",
    )
    assert _can_paste(mkA(scale=(-10, 10), translation=(-4, 10))) == (
        False,
        "sub-pixel translation",
    )
    assert _can_paste(mkA(translation=(0, 0.4))) == (False, "sub-pixel translation")
    assert _can_paste(mkA(translation=(0.4, 0))) == (False, "sub-pixel translation")


def test_compute_output_geobox():
    # sentinel2 over Gibraltar strait
    src = GeoBox.from_bbox(
        [199980, 3890220, 309780, 4000020], "EPSG:32630", resolution=10
    )

    # just copy resolution since both in meters
    dst = compute_output_geobox(src, "epsg:6933")
    assert dst.crs.units == src.crs.units
    assert dst.crs == "epsg:6933"
    assert dst.resolution == src.resolution
    assert dst.geographic_extent.contains(src.geographic_extent)
    assert compute_output_geobox(src, "epsg:6933") == src.to_crs("epsg:6933")

    assert compute_output_geobox(
        src, "epsg:6933", resolution="auto"
    ) == compute_output_geobox(src, "epsg:6933", resolution="same")

    # force estimation of new resolution
    dst = compute_output_geobox(src, "epsg:6933", resolution="fit")
    assert dst.crs == "epsg:6933"
    assert dst.resolution != src.resolution
    assert dst.resolution.x == -dst.resolution.y
    assert dst.geographic_extent.contains(src.geographic_extent)

    # force specific resolution
    dst = compute_output_geobox(src, "epsg:6933", resolution=101)
    assert dst.crs == "epsg:6933"
    assert dst.resolution == res_(101)
    assert dst.geographic_extent.contains(src.geographic_extent)

    # check identity case
    assert src is compute_output_geobox(src, src.crs)
    assert src is compute_output_geobox(src, src.crs, resolution="same")
    assert src is compute_output_geobox(src, src.crs, resolution="auto")

    # check conversion to lon/lat
    dst = compute_output_geobox(src, "epsg:4326")
    assert dst.crs == "epsg:4326"
    assert dst.resolution != src.resolution
    assert dst.resolution.x == -dst.resolution.y
    assert dst.geographic_extent.contains(src.geographic_extent)
    npix_change = (src.shape[0] * src.shape[1]) / (dst.shape[0] * dst.shape[1])
    assert 0.8 < npix_change < 1.1

    # go back from 4326
    _src = dst
    dst = compute_output_geobox(_src, src.crs)
    npix_change = (_src.shape[0] * _src.shape[1]) / (dst.shape[0] * dst.shape[1])
    assert 0.8 < npix_change < 1.1
    assert dst.geographic_extent.contains(_src.geographic_extent)

    # test bad input
    with pytest.raises(ValueError):
        _ = compute_output_geobox(src, "epsg:6933", resolution="bad-one")
