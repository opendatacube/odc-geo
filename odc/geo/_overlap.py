# This file is part of the Open Data Cube, see https://opendatacube.org for more information
#
# Copyright (c) 2015-2020 ODC Contributors
# SPDX-License-Identifier: Apache-2.0
import math
from dataclasses import dataclass
from typing import List, Optional, Protocol, Sequence, Tuple, TypeVar, Union

import numpy as np
from affine import Affine
from numpy import linalg

from .geobox import GeoBox, gbox_boundary
from .math import is_affine_st, maybe_int, snap_scale
from .roi import (
    NormalizedROI,
    NormalizedSlice,
    roi_boundary,
    roi_center,
    roi_from_points,
    roi_is_empty,
)
from .types import XY, SomeShape, shape_, xy_

SomeAffine = Union[Affine, np.ndarray]
AffineX = TypeVar("AffineX", np.ndarray, Affine)


class PointTransform(Protocol):
    """
    Invertible point transform.
    """

    def __call__(self, pts: Sequence[XY[float]]) -> Sequence[XY[float]]:
        ...  # pragma: nocover

    @property
    def back(self) -> "PointTransform":
        ...  # pragma: nocover

    @property
    def linear(self) -> Optional[Affine]:
        ...  # pragma: nocover


class LinearPointTransform:
    """
    Point transform within the same projection.
    """

    def __init__(self, A: Affine, back: Optional["LinearPointTransform"] = None):
        self.A = A
        self._back = back

    @property
    def linear(self) -> Optional[Affine]:
        return self.A

    @property
    def back(self) -> "LinearPointTransform":
        if self._back is not None:
            return self._back
        back = LinearPointTransform(~self.A, self)
        self._back = back
        return back

    def __call__(self, pts: Sequence[XY[float]]) -> Sequence[XY[float]]:
        A = self.A
        return [xy_(A * pt.xy) for pt in pts]


class GbxPointTransform:
    """
    Point Transform between two pixel planes.

    Maps pixel coordinates from one geo referenced image to another.

    1. Input is source pixel coordinate

    2. Compute coordinate in CRS units using linear transform of the source image

    3. Project point to the CRS of the destination image

    4. Compute destination image pixel coordinate by using inverse of the
       linear transform of the destination image
    """

    def __init__(
        self, src: GeoBox, dst: GeoBox, back: Optional["GbxPointTransform"] = None
    ):
        assert src.crs is not None and dst.crs is not None
        self._src = src
        self._dst = dst
        self._back = back
        self._tr = src.crs.transformer_to_crs(dst.crs)

    @property
    def back(self) -> "GbxPointTransform":
        if self._back is not None:
            return self._back
        back = GbxPointTransform(self._dst, self._src, self)
        self._back = back
        return back

    @property
    def linear(self) -> Optional[Affine]:
        return None

    def __call__(self, pts: Sequence[XY[float]]) -> Sequence[XY[float]]:
        # pix_src -> X -> X' -> pix_dst
        # inv(dst.A)*to_crs(src.A*pt)
        A = self._src.transform
        B = ~(self._dst.transform)

        pts = [A * pt.xy for pt in pts]
        xx = [pt[0] for pt in pts]
        yy = [pt[1] for pt in pts]
        xx, yy = self._tr(xx, yy)
        return [xy_(B * (x, y)) for x, y in zip(xx, yy)]


@dataclass
class ReprojectInfo:
    """
    Describes computed data loading parameters.

    For scale direction is: "scale > 1 --> shrink src to fit dst".
    """

    roi_src: NormalizedROI
    """Section of the source image to load."""

    roi_dst: NormalizedROI
    """Section of the destination image to update."""

    is_st: bool
    """True when related by simple translation and scale."""

    scale: float
    """Scale change as a single number."""

    scale2: XY[float]
    """Scale change per axis."""

    transform: PointTransform
    """Mapping from src pixels to destination pixels."""


def stack_xy(pts: Sequence[XY[float]]) -> np.ndarray:
    """Turn into an ``Nx2`` ndarray of floats in X,Y order."""
    return np.vstack([pt.xy for pt in pts])


def unstack_xy(pts: np.ndarray) -> List[XY[float]]:
    """Turn ``Nx2`` array in X,Y order into a list of XY points."""
    assert pts.ndim == 2 and pts.shape[1] == 2
    return [xy_(pt) for pt in pts]


def decompose_rws(A: AffineX) -> Tuple[AffineX, AffineX, AffineX]:
    """
     Compute decomposition Affine matrix sans translation into Rotation, Shear and Scale.

     Find matrices ``R,W,S`` such that ``A = R W S`` and

     .. code-block:

        R [ca -sa]  W [1, w]  S [sx,  0]
          [sa  ca]    [0, 1]    [ 0, sy]

    .. note:

       There are ambiguities for negative scales.

       * ``R(90)*S(1,1) == R(-90)*S(-1,-1)``

       * ``(R*(-I))*((-I)*S) == R*S``


     :return: Rotation, Sheer, Scale ``2x2`` matrices
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


def affine_from_pts(X: Sequence[XY[float]], Y: Sequence[XY[float]]) -> Affine:
    """
    Given points ``X,Y`` compute ``A``, such that: ``Y = A*X``.

    Needs at least 3 points.
    """

    assert len(X) == len(Y)
    assert len(X) >= 3

    n = len(X)

    YY = stack_xy(Y)
    XX = np.ones((n, 3), dtype="float64")
    for i, pt in enumerate(X):
        XX[i, :2] = pt.xy

    mm, *_ = linalg.lstsq(XX, YY, rcond=-1)
    a, d, b, e, c, f = mm.ravel()

    return Affine(a, b, c, d, e, f)


def get_scale_from_linear_transform(A: Affine) -> XY[float]:
    """
    Given a linear transform compute scale change.

    1. Y = A*X + t
    2. Extract scale components of A

    Returns (sx, sy), where sx > 0, sy > 0
    """
    _, _, S = decompose_rws(A)
    return xy_(abs(S.a), abs(S.e))


def get_scale_at_point(
    pt: XY[float], tr: PointTransform, r: Optional[float] = None
) -> XY[float]:
    """
    Given an arbitrary locally linear transform estimate scale change around a point.

    1. Approximate ``Y = tr(X)`` as ``Y = A*X + t`` in the neighbourhood of pt, for ``X,Y in R2``

    2. Extract scale components of ``A``


    :param pt: estimate transform around this point
    :param r:  radius around the point (default is 1)
    :param tr: Point transforming function ``Sequence[XY[float]] -> Sequence[XY[float]]``

    :return: ``(sx, sy)`` where ``sx > 0, sy > 0``
    """
    pts0 = [(0, 0), (-1, 0), (0, -1), (1, 0), (0, 1)]
    x0, y0 = pt.xy
    if r is None:
        XX = [xy_(float(x + x0), float(y + y0)) for x, y in pts0]
    else:
        XX = [xy_(float(x * r + x0), float(y * r + y0)) for x, y in pts0]
    YY = tr(XX)
    A = affine_from_pts(XX, YY)
    return get_scale_from_linear_transform(A)


def _same_crs_pix_transform(src: GeoBox, dst: GeoBox) -> LinearPointTransform:
    assert src.crs == dst.crs
    _fwd = (~dst.transform) * src.transform  # src -> dst
    return LinearPointTransform(_fwd)


def compute_axis_overlap(
    Ns: int, Nd: int, s: float, t: float
) -> Tuple[NormalizedSlice, NormalizedSlice]:
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


def box_overlap(
    src_shape: SomeShape, dst_shape: SomeShape, ST: Affine, tol: float
) -> Tuple[NormalizedROI, NormalizedROI]:
    """
    Compute overlap between two image planes.

    Given two image planes whose coordinate systems are related via scale and
    translation only, find overlapping regions within both.

    :param src_shape:
      Shape of source image plane

    :param dst_shape:
      Shape of destination image plane

    :param ST:
      Affine transform with only scale/translation, direction is: Xsrc = ST*Xdst

    :param tol:
      Sub-pixel translation tolerance that's scaled by resolution.
    """

    src_shape = shape_(src_shape)
    dst_shape = shape_(dst_shape)

    (sx, _, tx, _, sy, ty, *_) = ST

    sy = snap_scale(sy)
    sx = snap_scale(sx)

    ty = maybe_int(ty, tol)
    tx = maybe_int(tx, tol)

    s0, d0 = compute_axis_overlap(src_shape.y, dst_shape.y, sy, ty)
    s1, d1 = compute_axis_overlap(src_shape.x, dst_shape.x, sx, tx)
    return (s0, s1), (d0, d1)


def native_pix_transform(src: GeoBox, dst: GeoBox) -> PointTransform:
    """

    direction: from src to dst
    .back: goes the other way
    .linear: None|Affine linear transform src->dst if transform is linear (i.e. same CRS)
    """
    # Special case CRS_in == CRS_out
    if src.crs == dst.crs:
        return _same_crs_pix_transform(src, dst)

    return GbxPointTransform(src, dst)


def compute_reproject_roi(
    src: GeoBox,
    dst: GeoBox,
    tol: float = 0.05,
    padding: Optional[int] = None,
    align: Optional[int] = None,
) -> ReprojectInfo:
    """
    Compute reprojection information.

    Given two GeoBoxes find the region within the source GeoBox that overlaps
    with the destination GeoBox, and also compute the scale factor (>1 means
    shrink). Scale is chosen such that if you apply it to the source image
    before reprojecting, then reproject will have roughly no scale component.

    So we are breaking up reprojection into two stages:

    1. Scale in the native pixel CRS
    2. Reprojection (possibly non-linear with CRS change)

    .. code-block::

       - src[roi] -> scale      -> reproject -> dst  (using native pixels)
       - src(scale)[roi(scale)] -> reproject -> dst  (using overview image)

    Here ``roi`` is "minimal", padding is configurable though, so you only read what you need.
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

    :param src:
      Geobox of the source image

    :param dst:
      Geobox of the destination image

    :param padding:
      Optional padding in source pixels

    :param align:
      Optional pixel alignment in pixels, used on both source and destination.

    :param tol:
      Sub-pixel translation tolerance as pixel fraction.

    :returns:
      An instance of ``ReprojectInfo`` class.
    """
    pts_per_side = 5

    def compute_roi(
        src: GeoBox,
        dst: GeoBox,
        tr: PointTransform,
        pts_per_side: int,
        padding: int,
        align: Optional[int],
    ):
        _XY = tr.back(unstack_xy(gbox_boundary(dst, pts_per_side)))
        roi_src = roi_from_points(stack_xy(_XY), src.shape, padding, align=align)

        if roi_is_empty(roi_src):
            return (roi_src, np.s_[0:0, 0:0])

        # project src roi back into dst and compute roi from that
        xy = tr(unstack_xy(roi_boundary(roi_src, pts_per_side)))
        roi_dst = roi_from_points(
            stack_xy(xy), dst.shape, padding=0
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
        center_pt = xy_(roi_center(roi_src)[::-1])
        scale2 = get_scale_at_point(center_pt, tr)

    # change scale direction to be a shrink by factor
    scale2 = scale2.map(lambda s: 1.0 / s)
    scale = min(scale2.xy)

    return ReprojectInfo(
        roi_src=roi_src,
        roi_dst=roi_dst,
        scale=scale,
        scale2=scale2,
        is_st=is_st,
        transform=tr,
    )
