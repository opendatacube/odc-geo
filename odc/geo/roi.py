# This file is part of the Open Data Cube, see https://opendatacube.org for more information
#
# Copyright (c) 2015-2020 ODC Contributors
# SPDX-License-Identifier: Apache-2.0
"""
Tools for dealing with ROIs (Regions of Interest).

In this context ROI is a 2d slice of an image. For example a top left corner of 10 pixels square
will have an ROI that can be constructed with :py:func:`numpy.s_` like this: ``s_[0:10, 0:10]``.
"""
import math
from collections import abc
from typing import Optional, Tuple, Union, overload

import numpy as np

from .math import align_down, align_up
from .types import (
    NdROI,
    NormalizedROI,
    NormalizedSlice,
    Shape2d,
    SomeIndex2d,
    SomeShape,
    SomeSlice,
    T,
    iyx_,
    shape_,
)

# This is numeric code, short names make sense in this context, so disabling
# "invalid name" checks for the whole file
# pylint: disable=invalid-name


class WindowFromSlice:
    """Translate numpy slices to rasterio window tuples."""

    # pylint: disable=too-few-public-methods

    def __getitem__(self, roi):
        if roi is None:
            return None

        if not isinstance(roi, abc.Sequence) or len(roi) != 2:
            raise ValueError("Need 2d roi")

        row, col = roi
        return (
            (0 if row.start is None else row.start, row.stop),
            (0 if col.start is None else col.start, col.stop),
        )


w_ = WindowFromSlice()


class Tiles:
    """
    Partition box into tiles.

    Turns ``row, col`` index into a 2d ROI of the original box.
    """

    def __init__(self, base_shape: SomeShape, tile_shape: SomeShape) -> None:
        tile_shape = shape_(tile_shape)
        base_shape = shape_(base_shape)
        self._tile_shape = tile_shape
        self._base_shape = base_shape
        ny, nx = (
            int(math.ceil(float(N) / n)) for N, n in zip(base_shape.yx, tile_shape.yx)
        )
        self._shape = shape_((ny, nx))

    def __getitem__(self, idx: SomeIndex2d) -> NormalizedROI:
        def _slice(i, N, n) -> NormalizedSlice:
            _in = i * n
            if 0 <= _in < N:
                return slice(_in, min(_in + n, N))
            raise IndexError(f"Index {idx} is out of range")

        idx = iyx_(idx)

        ir, ic = (
            _slice(i, N, n)
            for i, N, n in zip(idx.yx, self._base_shape.yx, self._tile_shape.yx)
        )
        return (ir, ic)

    def tile_shape(self, idx: SomeIndex2d) -> Shape2d:
        """
        Query shape for a given tile.

        :param idx: ``(row, col)`` chunk index
        :returns: shape of a tile (edge tiles might be smaller)
        :raises: :py:class:`IndexError` when index is outside of ``[(0,0) -> .shape)``.
        """
        idx = iyx_(idx)

        def _sz(i: int, n: int, tile_sz: int, total_sz: int) -> int:
            if 0 <= i < n - 1:  # not edge tile
                return tile_sz
            if i == n - 1:  # edge tile
                return total_sz - (i * tile_sz)
            # out of index case
            raise IndexError(f"Index {idx} is out of range")

        ny, nx = map(
            _sz, idx.yx, self._shape.yx, self._tile_shape.yx, self._base_shape.yx
        )
        return shape_((ny, nx))

    @property
    def shape(self) -> Shape2d:
        """Number of tiles along each dimension."""
        return self._shape

    @property
    def base(self) -> Shape2d:
        """Base shape."""
        return self._base_shape


def polygon_path(x: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Points along axis aligned polygon.

    A little bit like :py:func:`numpy.meshgrid`, except returns only boundary points and limited to
    a 2d case only.

    .. rubric:: Examples

    .. code-block::

       [0,1] - unit square
       [0,1], [0,1] => [[0, 1, 1, 0, 0],
                        [0, 0, 1, 1, 0]])

       # three points per X, two point per Y side
       [0,1,2], [7,9] => [[0, 1, 2, 2, 1, 0, 0],
                          [7, 7, 7, 9, 9, 9, 7]]

    """

    if y is None:
        y = x

    return np.vstack(
        [
            np.vstack([x, np.full_like(x, y[0])]).T,
            np.vstack([np.full_like(y, x[-1]), y]).T[1:],
            np.vstack([x, np.full_like(x, y[-1])]).T[::-1][1:],
            np.vstack([np.full_like(y, x[0]), y]).T[::-1][1:],
        ]
    ).T


def roi_boundary(roi: NormalizedROI, pts_per_side: int = 2) -> np.ndarray:
    """
    Get boundary points from a 2d roi.

    roi needs to be in the normalised form, i.e. no open-ended start/stop,

    :returns:
      ``Nx2 <float32>`` array of ``X,Y`` points on the perimeter of the envelope defined by ``roi``

    .. seealso:: :py:func:`~odc.geo.roi.roi_normalise`
    """
    yy, xx = roi
    xx = np.linspace(xx.start, xx.stop, pts_per_side, dtype="float32")
    yy = np.linspace(yy.start, yy.stop, pts_per_side, dtype="float32")

    return polygon_path(xx, yy).T[:-1]


def scaled_down_roi(roi: NormalizedROI, scale: int) -> NormalizedROI:
    """
    Compute ROI for a scaled down image.

    Given a crop region of the original image compute equivalent crop in the overview image.

    :param roi: ROI in the original image
    :param scale: integer scale to get scaled down image
    :return: ROI in the scaled down image
    """
    s1, s2 = (slice(s.start // scale, align_up(s.stop, scale) // scale) for s in roi)
    return (s1, s2)


def scaled_up_roi(
    roi: NormalizedROI, scale: int, shape: Optional[SomeShape] = None
) -> NormalizedROI:
    """
    Compute ROI for a scaled up image.

    Given a crop region in the original image compute equivalent crop in the upsampled image.

    :param roi: ROI in the original image
    :param scale: integer scale to get scaled up image
    :param shape: Clamp to that shape is supplied
    :return: ROI in the scaled up image
    """
    s1, s2 = (slice(s.start * scale, s.stop * scale) for s in roi)
    if shape is not None:
        s1, s2 = (
            slice(min(dim, s.start), min(dim, s.stop))
            for s, dim in zip([s1, s2], shape_(shape))
        )
    return (s1, s2)


def scaled_down_shape(shape: Tuple[int, ...], scale: int) -> Tuple[int, ...]:
    """
    Compute shape of the overview image.

    :param shape: Original shape
    :param scale: Shrink factor
    :return: shape of the overview image
    """
    return tuple(align_up(s, scale) // scale for s in shape)


def roi_shape(roi: NdROI) -> Tuple[int, ...]:
    """
    Shape of an array after cropping with ``roi``.

    Same as ``xx[roi].shape``.
    """

    def slice_dim(s: SomeSlice) -> int:
        if isinstance(s, int):
            return 1
        _out = s.stop
        if _out is None:
            raise ValueError(
                "Can't determine shape of the slice with open right-hand side."
            )
        if s.start is None:
            return _out
        return _out - s.start

    if not isinstance(roi, tuple):
        roi = (roi,)

    return tuple(slice_dim(s) for s in roi)


def roi_is_empty(roi: NdROI) -> bool:
    """
    Check if ROI is "empty".

    ROI is empty if any dimension is 0 elements wide.
    """
    return any(d <= 0 for d in roi_shape(roi))


def roi_is_full(roi: NdROI, shape: Union[int, Tuple[int, ...]]) -> bool:
    """
    Check if ROI covers the entire region.

    :returns: ``True`` if ``roi`` covers region from ``(0,..) -> shape``
    :returns: ``False`` if ``roi`` actually crops an image
    """

    def slice_full(s: SomeSlice, n: int) -> bool:
        if isinstance(s, int):
            return n == 1
        return s.start in (0, None) and s.stop in (n, None)

    if not isinstance(roi, tuple):
        roi = (roi,)

    if not isinstance(shape, tuple):
        shape = (shape,)

    return all(slice_full(s, n) for s, n in zip(roi, shape))


def _fill_if_none(x: Optional[T], val_if_none: T) -> T:
    return val_if_none if x is None else x


def _norm_slice_or_error(s: SomeSlice) -> NormalizedSlice:
    if isinstance(s, int):
        start = s
        stop = s + 1
        step = None
    else:
        start = _fill_if_none(s.start, 0)

        if s.stop is None:
            raise ValueError("Can't process open ended slice")

        stop = s.stop
        step = s.step

    if stop < 0 or start < 0:
        raise ValueError("Can't process negative offset slice")

    return slice(start, stop, step)


def _norm_slice(s: SomeSlice, n: int) -> NormalizedSlice:
    if isinstance(s, int):
        return slice(s, s + 1)
    start = _fill_if_none(s.start, 0)
    stop = _fill_if_none(s.stop, n)
    start, stop = (x if x >= 0 else n + x for x in (start, stop))
    return slice(start, stop, s.step)


# fmt: off
@overload
def roi_normalise(roi: SomeSlice, shape: Union[int, Tuple[int]]) -> NormalizedSlice: ...
@overload
def roi_normalise( roi: Tuple[SomeSlice, ...], shape: Tuple[int, ...]
) -> Tuple[NormalizedSlice, ...]: ...
# fmt: on


def roi_normalise(
    roi: NdROI, shape: Union[int, Tuple[int, ...]]
) -> Union[NormalizedSlice, Tuple[NormalizedSlice, ...]]:
    """
    Normalise ROI.

    Fill in missing ``.start/.stop``, also deal with negative values, which are treated as offsets
    from the end.

    ``.step`` parameter is left unchanged.

    .. rubric:: Example

    .. code-block::

       np.s_[:3, 4:  ], (10, 20) => np.s_[0:3, 4:20]
       np.s_[:3,  :-3], (10, 20) => np.s_[0:3, 0:17]

    """
    if not isinstance(roi, abc.Sequence):
        if isinstance(shape, abc.Sequence):
            (shape,) = shape
        return _norm_slice(roi, shape)

    if not isinstance(shape, abc.Sequence):
        shape = (shape,)

    return tuple(_norm_slice(s, n) for s, n in zip(roi, shape))


# fmt: off
@overload
def roi_pad(roi: SomeSlice, pad: int, shape: int) -> NormalizedSlice: ...
@overload
def roi_pad(roi: Tuple[SomeSlice,...], pad: int, shape: Tuple[int, ...]) -> Tuple[NormalizedSlice, ...]: ...
# fmt: on


def roi_pad(
    roi: NdROI, pad: int, shape: Union[int, Tuple[int, ...]]
) -> Union[NormalizedSlice, Tuple[NormalizedSlice, ...]]:
    """
    Pad ROI on each side, with clamping.

    Returned ROI is guaranteed to be within ``(0,..) -> shape``.
    """

    def pad_slice(s: SomeSlice, n: int) -> NormalizedSlice:
        s = _norm_slice(s, n)
        return slice(max(0, s.start - pad), min(n, s.stop + pad))

    if not isinstance(roi, abc.Sequence):
        if isinstance(shape, abc.Sequence):
            (shape,) = shape
        return pad_slice(roi, shape)

    if not isinstance(shape, abc.Sequence):
        shape = (shape,)

    return tuple(pad_slice(s, n) for s, n in zip(roi, shape))


# fmt: off
@overload
def roi_intersect(a: SomeSlice, b: SomeSlice) -> NormalizedSlice: ...
@overload
def roi_intersect(a: Tuple[SomeSlice, ...], b: Tuple[SomeSlice, ...]) -> Tuple[NormalizedSlice, ...]: ...
# fmt: on


def roi_intersect(
    a: NdROI, b: NdROI
) -> Union[NormalizedSlice, Tuple[NormalizedSlice, ...]]:
    """
    Compute intersection of two ROIs.

    .. rubric:: Examples

    .. code-block::

       s_[1:30], s_[20:40] => s_[20:30]
       s_[1:10], s_[20:40] => s_[10:10]

       # works for N dimensions
       s_[1:10, 11:21], s_[8:12, 10:30] => s_[8:10, 11:21]

    """

    def slice_intersect(a: SomeSlice, b: SomeSlice) -> NormalizedSlice:
        a = _norm_slice_or_error(a)
        b = _norm_slice_or_error(b)

        if a.stop < b.start:
            return slice(a.stop, a.stop)
        if a.start > b.stop:
            return slice(a.start, a.start)

        _in = max(a.start, b.start)
        _out = min(a.stop, b.stop)

        return slice(_in, _out)

    if not isinstance(a, abc.Sequence):
        if isinstance(b, abc.Sequence):
            (b,) = b
        return slice_intersect(a, b)

    if not isinstance(b, abc.Sequence):
        b = (b,)

    return tuple(slice_intersect(sa, sb) for sa, sb in zip(a, b))


# fmt: off
@overload
def roi_center(roi: SomeSlice) -> float: ...
@overload
def roi_center(roi: Tuple[SomeSlice, ...]) -> Tuple[float, ...]: ...
# fmt: on


def roi_center(roi: NdROI) -> Union[float, Tuple[float, ...]]:
    """Return center point of an ``roi``."""

    def slice_center(s: SomeSlice) -> float:
        s = _norm_slice_or_error(s)
        return (s.start + s.stop) * 0.5

    if not isinstance(roi, abc.Sequence):
        return slice_center(roi)

    return tuple(slice_center(s) for s in roi)


def roi_from_points(
    xy: np.ndarray,
    shape: SomeShape,
    padding: int = 0,
    align: Optional[int] = None,
) -> NormalizedROI:
    """
    Build ROI from sample points.

    Compute envelope around a bunch of points and return it as an ROI (tuple of
    row/col slices)

    Returned roi is clipped ``(0,0) --> shape``, so it won't stick outside of the
    valid region.
    """
    shape = shape_(shape)

    def to_roi(*args):
        return tuple(slice(int(v[0]), int(v[1])) for v in args)

    assert len(shape) == 2
    assert xy.ndim == 2 and xy.shape[1] == 2

    ny, nx = shape

    _in = np.floor(xy.min(axis=0)).astype("int32") - padding
    _out = np.ceil(xy.max(axis=0)).astype("int32") + padding

    if align is not None:
        _in = align_down(_in, align)
        _out = align_up(_out, align)

    xx = np.asarray([_in[0], _out[0]])
    yy = np.asarray([_in[1], _out[1]])

    xx = np.clip(xx, 0, nx, out=xx)
    yy = np.clip(yy, 0, ny, out=yy)

    return to_roi(yy, xx)
