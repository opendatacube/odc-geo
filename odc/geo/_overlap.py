# This file is part of the Open Data Cube, see https://opendatacube.org for more information
#
# Copyright (c) 2015-2020 ODC Contributors
# SPDX-License-Identifier: Apache-2.0
import math
from types import SimpleNamespace
from typing import Optional, Tuple, Union

import numpy as np
from affine import Affine
from numpy import linalg

from .roi import roi_boundary, roi_center, roi_from_points, roi_is_empty
from .geobox import GeoBox, gbox_boundary
from .math import is_affine_st, maybe_int, snap_scale


def decompose_rws(
    A: Union[Affine, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute decomposition Affine matrix sans translation into Rotation, Shear and Scale.

    Note: that there are ambiguities for negative scales.

    Example: R(90)*S(1,1) == R(-90)*S(-1,-1),
    (R*(-I))*((-I)*S) == R*S

    A = R W S

    Where:

    R [ca -sa]  W [1, w]  S [sx,  0]
      [sa  ca]    [0, 1]    [ 0, sy]

    :return: Rotation, Sheer, Scale
    """
    # pylint: disable=too-many-locals

    if isinstance(A, Affine):

        def to_affine(m, t=(0, 0)):
            a, b, d, e = m.ravel()
            c, f = t
            return Affine(a, b, c, d, e, f)

        (a, b, c, d, e, f, *_) = A
        R, W, S = decompose_rws(np.asarray([[a, b], [d, e]], dtype="float64"))

        return to_affine(R, (c, f)), to_affine(W), to_affine(S)

    assert A.shape == (2, 2)

    WS = linalg.cholesky(A.T @ A).T
    R = A @ linalg.inv(WS)

    if linalg.det(R) < 0:
        R[:, -1] *= -1
        WS[-1, :] *= -1

    ss = np.diag(WS)
    S = np.diag(ss)
    W = WS @ np.diag(1.0 / ss)

    return R, W, S


def affine_from_pts(X, Y):
    """
    Given points X,Y compute A, such that: Y = A*X.

    Needs at least 3 points.

    :rtype: Affine
    """

    assert len(X) == len(Y)
    assert len(X) >= 3

    n = len(X)

    XX = np.ones((n, 3), dtype="float64")
    YY = np.vstack(Y)
    for i, x in enumerate(X):
        XX[i, :2] = x

    mm, *_ = linalg.lstsq(XX, YY, rcond=-1)
    a, d, b, e, c, f = mm.ravel()

    return Affine(a, b, c, d, e, f)


def get_scale_from_linear_transform(A):
    """
    Given a linear transform compute scale change.

    1. Y = A*X + t
    2. Extract scale components of A

    Returns (sx, sy), where sx > 0, sy > 0
    """
    _, _, S = decompose_rws(A)
    return abs(S.a), abs(S.e)


def get_scale_at_point(pt, tr, r=None):
    """
    Given an arbitrary locally linear transform estimate scale change around a point.

    1. Approximate Y = tr(X) as Y = A*X+t in the neighbourhood of pt, for X,Y in R2
    2. Extract scale components of A


    pt - estimate transform around this point
    r  - radius around the point (default 1)

    tr - List((x,y)) -> List((x,y))
         takes list of 2-d points on input and outputs same length list of 2d on output

    Returns (sx, sy), where sx > 0, sy > 0
    """
    pts0 = [(0, 0), (-1, 0), (0, -1), (1, 0), (0, 1)]
    x0, y0 = pt
    if r is None:
        XX = [(float(x + x0), float(y + y0)) for x, y in pts0]
    else:
        XX = [(float(x * r + x0), float(y * r + y0)) for x, y in pts0]
    YY = tr(XX)
    A = affine_from_pts(XX, YY)
    return get_scale_from_linear_transform(A)


def _same_crs_pix_transform(src, dst):
    assert src.crs == dst.crs

    def transform(pts, A):
        return [A * pt[:2] for pt in pts]

    _fwd = (~dst.transform) * src.transform  # src -> dst
    _bwd = ~_fwd  # dst -> src

    def pt_tr(pts):
        return transform(pts, _fwd)

    pt_tr.back = lambda pts: transform(pts, _bwd)
    pt_tr.back.back = pt_tr
    pt_tr.linear = _fwd
    pt_tr.back.linear = _bwd

    return pt_tr


def compute_axis_overlap(Ns: int, Nd: int, s: float, t: float) -> Tuple[slice, slice]:
    """
    s, t define linear transform from destination coordinate space to source
    >>  x_s = s * x_d + t

    Ns -- number of pixels along some dimension of source image: (0, Ns)
    Nd -- same as Ns but for destination image

    :returns: (slice in the source image,
               slice in the destination image)
    """

    needs_flip = s < 0

    if needs_flip:
        # change s, t to map into flipped src, i.e. src[::-1]
        s, t = -s, Ns - t

    assert s > 0

    # x_d = (x_s - t)/s => 1/s * x_s + t*(-1/s)
    #
    # x_d = s_ * x_s + t_
    s_ = 1.0 / s
    t_ = -t * s_

    if t < 0:
        #  |<------- ... D
        #      |<--- ... S
        _in = (0, min(math.floor(t_), Nd))
    else:
        #        |<--... D
        # |<---------... S
        _in = (min(math.floor(t), Ns), 0)

    a = math.ceil(Nd * s + t)
    if a <= Ns:
        # ...----->|    D
        # ...-------->| S
        _out = (max(a, 0), Nd)
    else:
        # ...-------->|  D
        # ...----->|     S
        _out = (Ns, max(0, math.ceil(Ns * s_ + t_)))

    src, dst = (slice(_in[i], _out[i]) for i in range(2))

    if needs_flip:
        # remap src from flipped space to normal
        src = slice(Ns - src.stop, Ns - src.start)  # type: ignore

    return (src, dst)


def box_overlap(src_shape, dst_shape, ST, tol):
    """
    Given two image planes whose coordinate systems are related via scale and
    translation only, find overlapping regions within both.

    :param src_shape: Shape of source image plane
    :param dst_shape: Shape of destination image plane
    :param        ST: Affine transform with only scale/translation,
                      direction is: Xsrc = ST*Xdst
    :param       tol: Sub-pixel translation tolerance that's scaled by resolution.
    """

    (sx, _, tx, _, sy, ty, *_) = ST

    sy = snap_scale(sy)
    sx = snap_scale(sx)

    ty = maybe_int(ty, tol)
    tx = maybe_int(tx, tol)

    s0, d0 = compute_axis_overlap(src_shape[0], dst_shape[0], sy, ty)
    s1, d1 = compute_axis_overlap(src_shape[1], dst_shape[1], sx, tx)
    return (s0, s1), (d0, d1)


def native_pix_transform(src: GeoBox, dst: GeoBox):
    """

    direction: from src to dst
    .back: goes the other way
    .linear: None|Affine linear transform src->dst if transform is linear (i.e. same CRS)
    """
    # Special case CRS_in == CRS_out
    if src.crs == dst.crs:
        return _same_crs_pix_transform(src, dst)

    _in = SimpleNamespace(crs=src.crs, A=src.transform)
    _out = SimpleNamespace(crs=dst.crs, A=dst.transform)

    _fwd = _in.crs.transformer_to_crs(_out.crs)
    _bwd = _out.crs.transformer_to_crs(_in.crs)

    _fwd = (_in.A, _fwd, ~_out.A)
    _bwd = (_out.A, _bwd, ~_in.A)

    def transform(pts, params):
        A, f, B = params
        return [B * pt[:2] for pt in [f(*(A * pt[:2])) for pt in pts]]

    def tr(pts):
        return transform(pts, _fwd)

    # TODO: re-work with dataclasses

    tr.back = lambda pts: transform(pts, _bwd)  # type: ignore
    tr.back.back = tr  # type: ignore
    tr.linear = None  # type: ignore
    tr.back.linear = None  # type: ignore

    return tr


def compute_reproject_roi(
    src: GeoBox,
    dst: GeoBox,
    tol: float = 0.05,
    padding: Optional[int] = None,
    align: Optional[int] = None,
):
    """Given two GeoBoxes find the region within the source GeoBox that overlaps
    with the destination GeoBox, and also compute the scale factor (>1 means
    shrink). Scale is chosen such that if you apply it to the source image
    before reprojecting, then reproject will have roughly no scale component.

    So we breaking up reprojection into two stages:

    1. Scale in the native pixel CRS
    2. Reprojection (possibly non-linear with CRS change)

    - src[roi] -> scale      -> reproject -> dst  (using native pixels)
    - src(scale)[roi(scale)] -> reproject -> dst  (using overview image)

    Here roi is "minimal", padding is configurable though, so you only read what you need.
    Also scale can be used to pick the right kind of overview level to read.

    Applying reprojection in two steps allows us to use pre-computed overviews,
    particularly useful when shrink factor is large. But even for data sources
    without overviews there are advantages for shrinking source image before
    applying reprojection: mainly quality of the output (reduces aliasing for
    large shrink factors), improved efficiency of the computation is likely as
    well.

    Also compute and return ROI of the dst geobox that is affected by src.

    If padding is None "appropriate" padding will be used depending on the
    transform between src<>dst:

    - No padding beyond sub-pixel alignment if Scale+Translation
    - 1 pixel source padding in all other cases

    :param tol: Sub-pixel translation tolerance as a percentage of resolution.

    :returns: SimpleNamespace with following fields:
     .roi_src    : (slice, slice)
     .roi_dst    : (slice, slice)
     .scale      : float
     .scale2     : (sx: float, sy: float)
     .is_st      : True|False
     .transform  : src coord -> dst coord

    For scale direction is: "scale > 1 --> shrink src to fit dst"

    """
    pts_per_side = 5

    def compute_roi(src, dst, tr, pts_per_side, padding, align):
        XY = np.vstack(tr.back(gbox_boundary(dst, pts_per_side)))
        roi_src = roi_from_points(XY, src.shape, padding, align=align)

        if roi_is_empty(roi_src):
            return (roi_src, np.s_[0:0, 0:0])

        # project src roi back into dst and compute roi from that
        xy = np.vstack(tr(roi_boundary(roi_src, pts_per_side)))
        roi_dst = roi_from_points(
            xy, dst.shape, padding=0
        )  # no need to add padding twice
        return (roi_src, roi_dst)

    tr = native_pix_transform(src, dst)

    if tr.linear is not None:
        tight_ok = align in (None, 0) and padding in (0, None)
        is_st = is_affine_st(tr.linear)

        if tight_ok and is_st:
            roi_src, roi_dst = box_overlap(src.shape, dst.shape, tr.back.linear, tol)
        else:
            padding = 1 if padding is None else padding
            roi_src, roi_dst = compute_roi(src, dst, tr, 2, padding, align)

        scale2 = get_scale_from_linear_transform(tr.linear)
    else:
        is_st = False
        padding = 1 if padding is None else padding

        roi_src, roi_dst = compute_roi(src, dst, tr, pts_per_side, padding, align)
        center_pt = roi_center(roi_src)[::-1]
        scale2 = get_scale_at_point(center_pt, tr)

    # change scale direction to be a shrink by factor
    scale2 = tuple(1 / s for s in scale2)
    scale = min(scale2)

    return SimpleNamespace(
        roi_src=roi_src,
        roi_dst=roi_dst,
        scale=scale,
        scale2=scale2,
        is_st=is_st,
        transform=tr,
    )
