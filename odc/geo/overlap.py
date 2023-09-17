# This file is part of the Open Data Cube, see https://opendatacube.org for more information
#
# Copyright (c) 2015-2020 ODC Contributors
# SPDX-License-Identifier: Apache-2.0
import math
from dataclasses import dataclass
from typing import Callable, Literal, Optional, Protocol, Sequence, Tuple, Union

import numpy as np
from affine import Affine

from .crs import SomeCRS
from .gcp import GCPGeoBox
from .geobox import GeoBox, GeoBoxBase, gbox_boundary
from .math import (
    affine_from_pts,
    decompose_rws,
    is_affine_st,
    is_almost_int,
    maybe_int,
    snap_affine,
    stack_xy,
    unstack_xy,
)
from .roi import (
    NormalizedROI,
    NormalizedSlice,
    roi_boundary,
    roi_center,
    roi_from_points,
    roi_is_empty,
    scaled_up_roi,
)
from .types import XY, SomeResolution, SomeShape, res_, shape_, xy_


class PointTransform(Protocol):
    """
    Invertible point transform.
    """

    def __call__(self, pts: Sequence[XY[float]]) -> Sequence[XY[float]]:
        ...

    @property
    def back(self) -> "PointTransform":
        ...

    @property
    def linear(self) -> Optional[Affine]:
        ...


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

    def __repr__(self) -> str:
        return f"LinearPointTransform(\n  {self.A!r})"


class GbxPointTransform:
    """
    Point Transform between two pixel planes.

    Maps pixel coordinates from one geo referenced image to another.

    1. Input is source pixel coordinate

    2. Compute coordinate in CRS units using pix2wld transform of the source image

    3. Project point to the CRS of the destination image

    4. Compute destination image pixel coordinate by using wld2pix mapping of the
       destination image
    """

    def __init__(
        self,
        src: GeoBoxBase,
        dst: GeoBoxBase,
        back: Optional["GbxPointTransform"] = None,
    ):
        assert src.crs is not None and dst.crs is not None
        self._src = src
        self._dst = dst
        self._back = back
        self._tr = src.crs.transformer_to_crs(dst.crs)
        self._clamps: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None
        if src.crs.geographic:
            self._clamps = ((-180, 180), (-90, 90))

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

        xx, yy = np.asarray([pt.xy for pt in pts]).T
        xx, yy = self._src.pix2wld(xx, yy)

        if self._clamps is not None:
            # for global datasets in 4326 pixel edges sometimes reach just outside
            # of the valid region due to rounding errors when creating tiff files
            # those coordinates can then not be converted properly to destintation crs
            range_x, range_y = self._clamps
            xx = np.clip(xx, *range_x)
            yy = np.clip(yy, *range_y)

        xx, yy = self._dst.wld2pix(*self._tr(xx, yy))
        return [xy_(x, y) for x, y in zip(xx, yy)]

    def __repr__(self) -> str:
        return f"GbxPointTransform({self._src!r}, {self._dst!r})"


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

    paste_ok: bool
    """
    When ``True`` source can be pasted into destination directly.

    * Must have same projection

    * Must have same pixel size, or same pixel size after shrinking
      source with integer scaling

    * Sub-pixel translation between the source and the destination images must be lower
      than the requested tolerance

    """

    read_shrink: int
    """
    Highest allowed integer shrink factor on the read side.

    Used to pick overview level for reading. A value of ``3`` means you can
    shrink every ``3x3`` pixel block of the source image down to a single pixel,
    and still have higher resolution than requested.
    """

    scale: float
    """Scale change as a single number. (the min of scale2)"""

    scale2: XY[float]
    """Full 2D Scale change as an XY."""

    transform: PointTransform
    """Mapping from src pixels to destination pixels."""


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
    src_shape: SomeShape, dst_shape: SomeShape, ST: Affine
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
      Affine transform with only scale/translation, direction is: ``Xsrc = ST*Xdst``

    :returns: ``(src_roi, dst_roi)``
    """

    src_shape = shape_(src_shape)
    dst_shape = shape_(dst_shape)

    (sx, _, tx, _, sy, ty, *_) = ST

    s0, d0 = compute_axis_overlap(src_shape.y, dst_shape.y, sy, ty)
    s1, d1 = compute_axis_overlap(src_shape.x, dst_shape.x, sx, tx)
    return (s0, s1), (d0, d1)


def native_pix_transform(src: GeoBoxBase, dst: GeoBoxBase) -> PointTransform:
    """

    direction: from src to dst
    .back: goes the other way
    .linear: None|Affine linear transform src->dst if transform is linear (i.e. same CRS)
    """
    # Special case CRS_in == CRS_out
    if isinstance(src, GeoBox) and isinstance(dst, GeoBox) and src.crs == dst.crs:
        return _same_crs_pix_transform(src, dst)

    return GbxPointTransform(src, dst)


def _pick_read_scale(scale: float, tol: float = 1e-3) -> int:
    assert scale > 0
    # scale < 1 --> 1
    # Else: scale down to nearest integer, unless we can scale up by less than tol
    #
    # 2.999999 -> 3
    # 2.8 -> 2
    # 0.3 -> 1

    if scale < 1:
        return 1

    # if close to int from below snap to it
    scale = maybe_int(scale, tol)

    # otherwise snap to nearest integer below
    return int(scale)


def _can_paste(
    A: Affine, stol: float = 1e-3, ttol: float = 1e-2
) -> Tuple[bool, Optional[str]]:
    """
    Check if can read (possibly with scale) and paste, or do we need to read then reproject.

    :param A: Coordinate mapping from dst to src ``X_src = A*X_dst``

    :returns: (True, None) if one can just read and paste
    :returns: (False, Reason) if pasting is not possible, so need to reproject after reading
    """

    if not is_affine_st(A):
        return (False, "has rotation or shear")

    sx, sy = get_scale_from_linear_transform(A).xy
    scale = min(sx, sy)

    if not is_almost_int(scale, stol):  # non-integer scaling
        return False, "non-integer scale"

    read_scale = _pick_read_scale(scale)
    # A_ maps coords from `dst` to `src.overview[scale]`
    A_ = Affine.scale(1 / read_scale, 1 / read_scale) * A

    (sx, _, tx, _, sy, ty, *_) = A_  # tx, ty are in dst pixel space

    # Expect identity for scale change
    if any(abs(abs(s) - 1) > stol for s in (sx, sy)):  # not equal scaling across axis?
        return False, "sx!=sy, probably"

    # Check if sub-pixel translation within bounds
    if not all(is_almost_int(t, ttol) for t in (tx, ty)):
        return False, "sub-pixel translation"

    return True, None


def _relative_rois(
    src: GeoBoxBase,
    dst: GeoBoxBase,
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

    # `padding=0` is to avoid adding padding twice
    roi_dst = roi_from_points(stack_xy(xy), dst.shape, padding=0)
    return (roi_src, roi_dst)


def compute_reproject_roi(
    src: GeoBoxBase,
    dst: GeoBoxBase,
    ttol: float = 0.05,
    stol: float = 1e-3,
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

    .. code-block:: text

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

    If padding is ``None`` "appropriate" padding will be used depending on the
    transform between ``src<>dst``:

    * No padding beyond sub-pixel alignment if Scale+Translation

    * 1 pixel source padding in all other cases

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

    :param stol:
      Scale tolerance for pasting

    :returns:
      An instance of ``ReprojectInfo`` class.
    """
    # pylint: disable=too-many-locals

    pts_per_side = 5

    tr = native_pix_transform(src, dst)

    if tr.linear is None:
        padding = 1 if padding is None else padding
        roi_src, roi_dst = _relative_rois(
            src, dst, tr, pts_per_side=pts_per_side, padding=padding, align=align
        )

        if not roi_is_empty(roi_dst):
            center_pt = xy_(roi_center(roi_dst)[::-1])
            scale2 = get_scale_at_point(center_pt, tr.back)
            scale = min(scale2.xy)
            read_shrink = _pick_read_scale(scale)
        else:
            scale = 0
            scale2 = XY(x=0, y=0)
            read_shrink = 1

        return ReprojectInfo(
            roi_src=roi_src,
            roi_dst=roi_dst,
            scale=scale,
            scale2=scale2,
            paste_ok=False,
            read_shrink=read_shrink,
            transform=tr,
        )

    # Same projection case
    #
    A = tr.back.linear  # dst->src
    scale2 = get_scale_from_linear_transform(A)
    scale = min(scale2.xy)
    read_shrink = _pick_read_scale(scale)

    paste_ok = False
    tight_ok = align in (None, 0) and padding in (0, None)

    if tight_ok:
        paste_ok, _ = _can_paste(A, ttol=ttol, stol=stol)

    if paste_ok:
        if read_shrink == 1:
            A_ = snap_affine(A, ttol=ttol, stol=stol)
            roi_src, roi_dst = box_overlap(src.shape, dst.shape, A_)
        else:
            # compute overlap in scaled down image, then upscale source overlap
            assert isinstance(src, GeoBox)
            _src = src.zoom_out(read_shrink)
            A_ = snap_affine(Affine.scale(1 / read_shrink) * A, ttol=ttol, stol=stol)
            roi_src, roi_dst = box_overlap(_src.shape, dst.shape, A_)
            roi_src = scaled_up_roi(roi_src, read_shrink)
    else:
        padding = 1 if padding is None else padding
        roi_src, roi_dst = _relative_rois(
            src, dst, tr, pts_per_side=2, padding=padding, align=align
        )

    return ReprojectInfo(
        roi_src=roi_src,
        roi_dst=roi_dst,
        scale=scale,
        scale2=scale2,
        paste_ok=paste_ok,
        read_shrink=read_shrink,
        transform=tr,
    )


def compute_output_geobox(
    gbox: Union[GeoBox, GCPGeoBox],
    crs: SomeCRS,
    *,
    resolution: Union[SomeResolution, Literal["auto", "fit", "same"]] = "auto",
    tight: bool = False,
    round_resolution: Union[None, bool, Callable[[float, str], float]] = None,
) -> GeoBox:
    """
    Compute output ``GeoBox``.

    Find best fitting, axis aligned GeoBox in a different coordinate reference given source
    ``GeoBox`` on input.

    :param gbox:
       Source geobox.

    :param crs:
       Desired CRS of the output

    :param resolution:

       * "same" use exactly the same resolution as src
       * "fit" use center pixel to determine scale change between the two
       * | "auto" is to use the same resolution on the output if CRS units are the same
         |  between the source and destination and otherwise use "fit"
       * Else resolution in the units of the output crs

    :param tight:
      By default output pixel grid is adjusted to align pixel edges to X/Y axis, suppling
      ``tight=True`` produces unaligned geobox on the output.

    :param round_resolution: ``round_resolution(res: float, units: str) -> float``

    :return:
       Similar resolution, axis aligned geobox that fully encloses source one but in a different
       projection.
    """
    # pylint: disable=too-many-locals
    src_crs = gbox.crs
    assert src_crs is not None

    bbox = gbox.footprint(crs, buffer=0.9, npoints=100).boundingbox
    dst_crs = bbox.crs
    assert dst_crs is not None

    if (
        dst_crs == src_crs
        and resolution in ("auto", "same")
        and isinstance(gbox, GeoBox)
    ):
        return gbox

    same_units = src_crs.units == dst_crs.units

    if resolution == "same":
        res = gbox.resolution
    elif resolution == "auto" and same_units:
        res = gbox.resolution
    elif resolution in ("fit", "auto"):
        # get initial resolution by computing 1x1 bounding box of the center pixel
        #
        cp = gbox.center_pixel
        cp_bbox = cp.extent.to_crs(dst_crs).boundingbox
        dst_ = GeoBox.from_bbox(cp_bbox, dst_crs, shape=(1, 1), tight=True)

        # further adjust that via fitting
        #  Y = tr(X) => Y ~= s*X + t
        # where X is in `dst_` and Y is in `gbox`
        # .. so a better fit resolution is `_dst.res / s`
        # .. but we want square pixels, so pick average between x and y
        sx, sy = get_scale_at_point(xy_(0.5, 0.5), native_pix_transform(dst_, cp)).xy

        # always produces square pixels on output with inverted Y axis
        avg_res = (abs(dst_.resolution.x / sx) + abs(dst_.resolution.y / sy)) / 2
        if round_resolution is not None:
            if isinstance(round_resolution, bool):
                if round_resolution:
                    avg_res = round(avg_res, 0)
            else:
                avg_res = round_resolution(avg_res, dst_crs.units[0])
        res = res_(avg_res)
    else:
        if isinstance(resolution, str):
            raise ValueError(
                f"Resolution ought to be one of: same,auto,fit, not '{resolution}'"
            )

        res = res_(resolution)

    return GeoBox.from_bbox(bbox, dst_crs, resolution=res, tight=tight)
