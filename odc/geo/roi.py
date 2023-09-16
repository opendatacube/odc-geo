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
from typing import List, Optional, Protocol, Sequence, Tuple, Union, overload

import numpy as np

from .math import align_down, align_up, edge_index
from .types import (
    ROI,
    Chunks2d,
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


class RoiTiles(Protocol):
    """
    Abstraction for 2d slice/shape/chunks lookup.
    """

    def __getitem__(self, idx: Union[SomeIndex2d, ROI]) -> Tuple[slice, slice]:
        ...

    def crop(self, roi: ROI) -> "RoiTiles":
        ...

    def tile_shape(self, idx: SomeIndex2d) -> Shape2d:
        ...

    @property
    def shape(self) -> Shape2d:
        ...

    @property
    def base(self) -> Shape2d:
        ...

    @property
    def chunks(self) -> Chunks2d:
        ...

    def locate(self, pix: SomeIndex2d) -> Tuple[int, int]:
        ...

    def __dask_tokenize__(self):
        ...


def norm_slice_2d(
    idx: Union[SomeIndex2d, ROI], shape: Tuple[int, int]
) -> NormalizedROI:
    if isinstance(idx, tuple):
        return roi_normalise(idx, shape)
    return roi_normalise(iyx_(idx).yx, shape)


def _fmt_shape(shape):
    n1, n2 = shape.yx
    if max(n1, n2) > 10_000:
        return f"{n1:_d}x{n2:_d}"
    return f"{n1:d}x{n2:d}"


def clip_tiles(
    tiles: RoiTiles, selection: Sequence[Tuple[int, int]]
) -> Tuple["RoiTiles", Tuple[slice, slice], List[Tuple[int, int]]]:
    """
    Compute cropped version of tiles from a selection of tile indexes.

    :return: Cropped RoiTiles object, crop that was applied and tile indexes
             changed to the cropped address space.
    """
    ii = np.asarray(selection)
    y1, x1 = ii.min(axis=0).tolist()
    y2, x2 = ii.max(axis=0).tolist()
    roi = np.s_[y1 : y2 + 1, x1 : x2 + 1]
    sel_new = [(y - y1, x - x1) for y, x in selection]
    return tiles.crop(roi), roi, sel_new


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

    def crop(self, roi: ROI) -> "Tiles":
        base_shape = roi_shape(self[roi])
        return Tiles(base_shape, self._tile_shape)

    def __getitem__(self, idx: Union[SomeIndex2d, ROI]) -> Tuple[slice, slice]:
        def _slice(i: NormalizedSlice, N: int, n: int) -> slice:
            _in = i.start * n
            _out = i.stop * n

            if 0 <= _in < N and _out < N + n:
                return slice(_in, min(_out, N))
            raise IndexError(f"Index {idx} is out of range")

        idx = norm_slice_2d(idx, self._shape.yx)

        ir, ic = (
            _slice(i, N, n)
            for i, N, n in zip(idx, self._base_shape.yx, self._tile_shape.yx)
        )
        return (ir, ic)

    def tile_shape(self, idx: SomeIndex2d) -> Shape2d:
        """
        Query shape for a given tile.

        :param idx: ``(row, col)`` chunk index, supports numpy style indexing.
        :returns: shape of a tile (edge tiles might be smaller)
        :raises: :py:class:`IndexError` when index is outside of ``[(0,0) -> .shape)``.
        """
        idx = iyx_(idx)

        def _sz(i: int, n: int, tile_sz: int, total_sz: int) -> int:
            if i < 0:  # numpy style index from the right
                i = n + i
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

    @property
    def chunks(self) -> Chunks2d:
        """Dask compatible chunk rerpesentation."""
        NY, NX = self.shape.yx
        ny, nx = self.tile_shape((0, 0)).yx
        ny_, nx_ = self.tile_shape((NY - 1, NX - 1))

        return (
            (ny,) * (NY - 1) + (ny_,),
            (nx,) * (NX - 1) + (nx_,),
        )

    def locate(self, pix: SomeIndex2d) -> Tuple[int, int]:
        """Tile index from pixel coordinate."""
        NY, NX = self._base_shape.yx
        y, x = iyx_(pix).yx
        if y < 0 or y >= NY or x < 0 or x >= NX:
            raise IndexError()
        ny, nx = self.tile_shape((0, 0)).yx
        return (y // ny, x // nx)

    def __dask_tokenize__(self):
        return (
            "odc.geo.roi.Tiles",
            *self._shape,
            *self._tile_shape,
        )

    def __str__(self) -> str:
        b1, b2, b3 = map(_fmt_shape, [self._shape, self._tile_shape, self._base_shape])
        return f"Tiles: {b1}|{b2}px => {b3}px"

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, __value: object) -> bool:
        if __value is self:
            return True
        if not isinstance(__value, Tiles):
            return False
        return (
            self._base_shape == __value._base_shape
            and self._tile_shape == __value._tile_shape
        )


class VariableSizedTiles:
    """
    Partition box into tiles of varying sizes.

    Turns ``row, col`` index into a 2d ROI of the original box.
    """

    __slots__ = ("_offsets",)

    def __init__(self, chunks: Chunks2d) -> None:
        self._offsets = tuple(
            np.asarray([0, *idx], dtype="int32").cumsum(dtype="int32") for idx in chunks
        )

    def crop(self, roi: ROI) -> "VariableSizedTiles":
        roi = roi_normalise(roi, self.shape.yx)
        y, x = (ch[s.start : s.stop] for ch, s in zip(self.chunks, roi))
        return VariableSizedTiles((y, x))

    def __getitem__(self, idx: Union[SomeIndex2d, ROI]) -> Tuple[slice, slice]:
        idx = norm_slice_2d(idx, self.shape.yx)
        y, x = (
            slice(int(a[i.start]), int(a[i.stop])) for a, i in zip(self._offsets, idx)
        )
        return (y, x)

    def tile_shape(self, idx: SomeIndex2d) -> Shape2d:
        """
        Query shape for a given tile.

        :param idx: ``(row, col)`` chunk index
        :returns: shape of a tile
        :raises: :py:class:`IndexError` when index is outside of ``[(0,0) -> .shape)``.
        """
        idx = iyx_(idx)
        ny, nx = (int(a[i + 1]) - int(a[i]) for a, i in zip(self._offsets, idx.yx))
        return Shape2d(x=nx, y=ny)

    @property
    def shape(self) -> Shape2d:
        """Number of tiles along each dimension."""
        ny, nx = (len(idx) - 1 for idx in self._offsets)
        return Shape2d(x=nx, y=ny)

    @property
    def base(self) -> Shape2d:
        """Base shape."""
        ny, nx = (int(idx[-1]) for idx in self._offsets)
        return Shape2d(x=nx, y=ny)

    @property
    def chunks(self) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        """Dask compatible chunk rerpesentation."""
        y, x = (tuple(np.diff(idx).tolist()) for idx in self._offsets)
        return (y, x)

    def locate(self, pix: SomeIndex2d) -> Tuple[int, int]:
        """Tile index from pixel coordinate."""
        NY, NX = self.base.yx
        y, x = iyx_(pix).yx
        if y < 0 or y >= NY or x < 0 or x >= NX:
            raise IndexError()
        y, x = (
            # 1: because offsets start from 0, n1, n1+n2, ...
            int(np.searchsorted(bins[1:], pix, "right"))
            for bins, pix in zip(self._offsets, (y, x))
        )
        return (y, x)

    def __dask_tokenize__(self):
        return (
            "odc.geo.roi.VariableSizedTiles",
            *self._offsets,
        )

    def __str__(self) -> str:
        b1, b2, b3 = map(_fmt_shape, [self.shape, self.tile_shape((0, 0)), self.base])
        return f"Tiles: {b1}|chunked {b2}px => {b3}px"

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, __value: object) -> bool:
        if __value is self:
            return True
        if not isinstance(__value, VariableSizedTiles):
            return False

        for a, b in zip(self._offsets, __value._offsets):
            if a.shape != b.shape:
                return False
            if (a != b).any():
                return False
        return True


def roi_tiles(shape: SomeShape, how: Union[SomeShape, Chunks2d]) -> RoiTiles:
    if isinstance(how, (tuple, list)) and isinstance(how[0], (tuple, list)):
        y, x = (tuple(i) for i in how)
        return VariableSizedTiles((y, x))
    return Tiles(shape, how)


def polygon_path(
    x: np.ndarray,
    y: Optional[np.ndarray] = None,
    closed: bool = True,
) -> np.ndarray:
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

    :returns:
      A ``2xN`` array, ``x, y = polygon_path(...)``

    """

    if not isinstance(x, np.ndarray):
        x = np.asarray(x)

    if y is None:
        y = x

    if not isinstance(y, np.ndarray):
        y = np.asarray(y)

    shape = (len(y), len(x))
    iy, ix = np.asarray(list(edge_index(shape, closed=closed)), dtype="uint32").T
    return np.vstack([x[np.newaxis, ix], y[np.newaxis, iy]])


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

    return polygon_path(xx, yy, closed=False).T


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


# fmt: off
@overload
def roi_shape(roi: ROI) -> Tuple[int, int]: ...
@overload
def roi_shape(roi: NdROI) -> Tuple[int, ...]: ...
# fmt: on


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
        if s < 0:
            s = n + s
        return slice(s, s + 1)
    start = _fill_if_none(s.start, 0)
    stop = _fill_if_none(s.stop, n)
    start, stop = (x if x >= 0 else n + x for x in (start, stop))
    return slice(start, stop, s.step)


def slice_intersect3(a: SomeSlice, b: SomeSlice) -> Tuple[slice, slice, slice]:
    """
    Compute overlap 3 way.

    Compute part of a that overlaps b, part of b that overlaps a and the region
    of the original region covered by the overlap.

    :returns: ``a', b', ab'``, such that ``X[a][a'] == X[b][b'] == X[ab']``
    """
    a = _norm_slice_or_error(a)
    b = _norm_slice_or_error(b)
    na = a.stop - a.start
    nb = b.stop - b.start

    if a.stop < b.start:
        return slice(na, na), slice(0, 0), slice(a.stop, a.stop)
    if a.start > b.stop:
        return slice(0, 0), slice(nb, nb), slice(a.start, a.start)

    _in = max(a.start, b.start)
    _out = min(a.stop, b.stop)

    return (
        slice(_in - a.start, _out - a.start),
        slice(_in - b.start, _out - b.start),
        slice(_in, _out),
    )


def roi_intersect3(
    a: Tuple[SomeSlice, ...], b: Tuple[SomeSlice, ...]
) -> Tuple[Tuple[slice, ...], Tuple[slice, ...], Tuple[slice, ...]]:
    """
    Compute overlap 3 way.

    Compute part of a that overlaps b, part of b that overlaps a and the region
    of the original region covered by the overlap.

    :returns: ``a', b', ab'``, such that ``X[a][a'] == X[b][b'] == X[ab']``
    """
    assert len(a) == len(b)
    aa, bb, cc = zip(*[slice_intersect3(a_, b_) for a_, b_ in zip(a, b)])
    return aa, bb, cc


# fmt: off
@overload
def roi_normalise(roi: SomeSlice, shape: Union[int, Tuple[int]]) -> NormalizedSlice: ...
@overload
def roi_normalise(roi: ROI, shape: Tuple[int, int]) -> NormalizedROI: ...
@overload
def roi_normalise(roi: Tuple[SomeSlice, ...], shape: Tuple[int, ...]
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
def roi_pad(roi: Tuple[SomeSlice, ...], pad: int, shape: Tuple[int, ...]) -> Tuple[NormalizedSlice, ...]: ...
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

    # keep finite points only
    #  if any points are not finite remove them
    ok_mask = np.isfinite(xy)
    if not ok_mask.all():
        keep = ok_mask.T[0] * ok_mask.T[1]
        xy = xy[keep, :]

    if xy.shape[0] == 0:
        return np.s_[0:0, 0:0]

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
