# This file is part of the Open Data Cube, see https://opendatacube.org for more information
#
# Copyright (c) 2015-2020 ODC Contributors
# SPDX-License-Identifier: Apache-2.0
import collections.abc

import numpy as np

from .math import align_down, align_up

# This is numeric code, short names make sense in this context, so disabling
# "invalid name" checks for the whole file
# pylint: disable=invalid-name


class WindowFromSlice:
    """Translate numpy slices to rasterio window tuples."""

    # pylint: disable=too-few-public-methods

    def __getitem__(self, roi):
        if roi is None:
            return None

        if not isinstance(roi, collections.abc.Sequence) or len(roi) != 2:
            raise ValueError("Need 2d roi")

        row, col = roi
        return (
            (0 if row.start is None else row.start, row.stop),
            (0 if col.start is None else col.start, col.stop),
        )


w_ = WindowFromSlice()


def polygon_path(x, y=None):
    """A little bit like numpy.meshgrid, except returns only boundary values and
    limited to 2d case only.

    Examples:
      [0,1], [3,4] =>
      array([[0, 1, 1, 0, 0],
             [3, 3, 4, 4, 3]])

      [0,1] =>
      array([[0, 1, 1, 0, 0],
             [0, 0, 1, 1, 0]])
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


def roi_boundary(roi, pts_per_side=2):
    """
    Get boundary points from a 2d roi.

    roi needs to be in the normalised form, i.e. no open-ended start/stop, see roi_normalise

    :returns: Nx2 float32 array of X,Y points on the perimeter of the envelope defined by `roi`
    """
    yy, xx = roi
    xx = np.linspace(xx.start, xx.stop, pts_per_side, dtype="float32")
    yy = np.linspace(yy.start, yy.stop, pts_per_side, dtype="float32")

    return polygon_path(xx, yy).T[:-1]


def scaled_down_roi(roi, scale: int):
    return tuple(slice(s.start // scale, align_up(s.stop, scale) // scale) for s in roi)


def scaled_up_roi(roi, scale: int, shape=None):
    roi = tuple(slice(s.start * scale, s.stop * scale) for s in roi)
    if shape is not None:
        roi = tuple(
            slice(min(dim, s.start), min(dim, s.stop)) for s, dim in zip(roi, shape)
        )
    return roi


def scaled_down_shape(shape, scale: int):
    return tuple(align_up(s, scale) // scale for s in shape)


def roi_shape(roi):
    def slice_dim(s):
        return s.stop if s.start is None else s.stop - s.start

    if isinstance(roi, slice):
        roi = (roi,)

    return tuple(slice_dim(s) for s in roi)


def roi_is_empty(roi):
    return any(d <= 0 for d in roi_shape(roi))


def roi_is_full(roi, shape):
    """
    Check if ROI covers the entire region.

    :returns: True if roi covers region from (0,..) -> shape
              False otherwise
    """

    def slice_full(s, n):
        return s.start in (0, None) and s.stop in (n, None)

    if isinstance(roi, slice):
        roi = (roi,)
        shape = (shape,)

    return all(slice_full(s, n) for s, n in zip(roi, shape))


def roi_normalise(roi, shape):
    """
    Fill in missing .start/.stop, also deal with negative values, which are
    treated as offsets from the end.

    .step parameter is left unchanged.

    Example:
          np.s_[:3, 4:  ], (10, 20) -> np._s[0:3, 4:20]
          np.s_[:3,  :-3], (10, 20) -> np._s[0:3, 0:17]

    """

    def fill_if_none(x, val_if_none):
        return val_if_none if x is None else x

    def norm_slice(s, n):
        start = fill_if_none(s.start, 0)
        stop = fill_if_none(s.stop, n)
        start, stop = (x if x >= 0 else n + x for x in (start, stop))
        return slice(start, stop, s.step)

    if not isinstance(shape, collections.abc.Sequence):
        shape = (shape,)

    if isinstance(roi, slice):
        return norm_slice(roi, shape[0])

    return tuple(norm_slice(s, n) for s, n in zip(roi, shape))


def roi_pad(roi, pad, shape):
    """
    Pad ROI on each side, with clamping (0,..) -> shape
    """

    def pad_slice(s, n):
        return slice(max(0, s.start - pad), min(n, s.stop + pad))

    if isinstance(roi, slice):
        return pad_slice(roi, shape)

    return tuple(pad_slice(s, n) for s, n in zip(roi, shape))


def roi_intersect(a, b):
    """
    Compute intersection of two ROIs
    """

    def slice_intersect(a, b):
        if a.stop < b.start:
            return slice(a.stop, a.stop)
        if a.start > b.stop:
            return slice(a.start, a.start)

        _in = max(a.start, b.start)
        _out = min(a.stop, b.stop)

        return slice(_in, _out)

    if isinstance(a, slice):
        if not isinstance(b, slice):
            b = b[0]
        return slice_intersect(a, b)

    b = (b,) if isinstance(b, slice) else b

    return tuple(slice_intersect(sa, sb) for sa, sb in zip(a, b))


def roi_center(roi):
    """Return center point of roi"""

    def slice_center(s):
        return (s.start + s.stop) * 0.5

    if isinstance(roi, slice):
        return slice_center(roi)

    return tuple(slice_center(s) for s in roi)


def roi_from_points(xy, shape, padding=0, align=None):
    """
    Compute envelope around a bunch of points and return it as roi (tuple of
    row/col slices)

    Returned roi is clipped (0,0) --> shape, so it won't stick outside of the
    valid region.
    """

    def to_roi(*args):
        return tuple(slice(v[0], v[1]) for v in args)

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
