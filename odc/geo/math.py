# This file is part of the Open Data Cube, see https://opendatacube.org for more information
#
# Copyright (c) 2015-2020 ODC Contributors
# SPDX-License-Identifier: Apache-2.0
"""
Various mathy helpers.

Minimal dependencies in this module.
"""
from math import floor, fmod
from typing import Literal, Optional, Tuple, Union

import numpy as np
from affine import Affine

from .types import XY, SomeResolution, res_


def maybe_zero(x: float, tol: float) -> float:
    """Turn almost zeros to actual zeros."""
    if abs(x) < tol:
        return 0
    return x


def split_float(x: float) -> Tuple[float, float]:
    """
    Split float number into whole and fractional parts.

    Adding the two numbers back together should result in the original value.
    Fractional part is always in the ``(-0.5, +0.5)`` interval, and whole part
    is equivalent to ``round(x)``.

    :param x: floating point number
    :return: ``whole, fraction``
    """
    x_part = fmod(x, 1.0)
    x_whole = x - x_part
    if x_part > 0.5:
        x_part -= 1
        x_whole += 1
    elif x_part < -0.5:
        x_part += 1
        x_whole -= 1
    return (x_whole, x_part)


def maybe_int(x: float, tol: float) -> Union[int, float]:
    """
    Turn almost ints to actual ints.

    pass through other values unmodified.
    """

    x_whole, x_part = split_float(x)

    if abs(x_part) < tol:  # almost int
        return int(x_whole)
    return x


def snap_scale(s: float, tol: float = 1e-6) -> float:
    """
    Snap scale.

    Snap scale to the nearest integer or simple fractions in the form ``1/<int>`` if within
    tolerance.

    :return: ``s`` if too far from snap
    :return: snapped version of ``s`` when within ``tol`` of the integer scale.
    """
    if abs(s) >= 1 - tol:
        return maybe_int(s, tol)

    # Check of s is 0
    if abs(s) < tol:
        return s

    # Check for simple fractions
    s_inv = 1 / s
    s_inv_snapped = maybe_int(s_inv, tol)
    if s_inv_snapped is s_inv:
        return s
    return 1 / s_inv_snapped


def align_down(x: int, align: int) -> int:
    """
    Align integer down.

    :return:
      ``y`` such that ``y % align == 0`` and ``y <= x`` and ``(x - y) < align``
    """
    return x - (x % align)


def align_up(x: int, align: int) -> int:
    """
    Align integer up.

    :return:
      ``y`` such that ``y % align == 0`` and ``y >= x`` and ``(y - x) < align``
    """
    return align_down(x + (align - 1), align)


def clamp(x, lo, up):
    """Clamp ``x`` to be ``lo <= x <= up``."""
    assert lo <= up
    return lo if x < lo else up if x > up else x


def is_almost_int(x: float, tol: float) -> bool:
    """
    Check if number is close enough to an integer.

    :param x: number to check
    :param tol: tolerance to use.
    """
    x = abs(fmod(x, 1))
    if x > 0.5:
        x = 1 - x
    return x < tol


def data_resolution_and_offset(
    data, fallback_resolution: Optional[float] = None
) -> Tuple[float, float]:
    """
    Compute resolution and offset from x/y axis data.

    Only uses first two coordinate values, assumes that data is regularly
    sampled.

    :returns: ``(resolution, offset)``
    """
    if data.size < 2:
        if data.size < 1:
            raise ValueError("Can't calculate resolution for empty data")
        if fallback_resolution is None:
            raise ValueError("Can't calculate resolution with data size < 2")
        res = fallback_resolution
    else:
        _res = (data[data.size - 1] - data[0]) / (data.size - 1.0)
        res = _res.item()

    off = data[0] - 0.5 * res
    return res, off.item()


def affine_from_axis(
    xx, yy, fallback_resolution: Optional[SomeResolution] = None
) -> Affine:
    """
    Compute Affine transform from axis.

    Transform direction is from pixel coordinates to CRS units of ``X/Y`` axis.

    :param xx:
       ``X`` axis coordinates

    :param yy:
       ``Y`` axis coordinates

    :param fallback_resolution:
       Resolution to assume for single element axis.

    .. code-block:: text

        (0,0) in pixel space is defined as top left corner of the top left pixel
          |
          V 0   1
          +---+---+
        0 |   |   |
          +---+---+
        1 |   |   |
          +---+---+

    Only uses first two coordinate values, assumes that data is regularly
    sampled.

    :raises:
       :py:class:`ValueError` when any axis is empty.

    :raises:
       :py:class:`ValueError` when any axis has single value and fallback
       resolution was not supplied.
    """
    frx: Optional[float] = None
    fry: Optional[float] = None

    if fallback_resolution is not None:
        frx, fry = res_(fallback_resolution).xy

    xres, xoff = data_resolution_and_offset(xx, frx)
    yres, yoff = data_resolution_and_offset(yy, fry)

    return Affine.translation(xoff, yoff) * Affine.scale(xres, yres)


def apply_affine(
    A: Affine, x: np.ndarray, y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Broadcast ``A*(x_i, y_i)`` across all elements of ``x/y``.

    Arrays could be in any shape (usually 2d image).

    :param A: Affine transform to apply
    :param x: x coordinates (any shape)
    :param y: y coordinates (same shape as ``x``)
    :returns: Transformed coordinate as two arrays of the same shape as input ``(x', y')``
    """

    shape = x.shape

    A = np.asarray(A).reshape(3, 3)
    t = A[:2, -1].reshape((2, 1))
    A = A[:2, :2]

    x, y = A @ np.vstack([x.ravel(), y.ravel()]) + t
    x, y = (a.reshape(shape) for a in (x, y))
    return (x, y)


def split_translation(t: XY[float]) -> Tuple[XY[float], XY[float]]:
    """
    Split translation into pixel aligned and sub-pixel parts.

    Subpixel translation is guaranteed to be in ``[-0.5, +0.5]`` range.

    .. code-block:: python

       x + t = x + t_whole + t_subpix

    :param t: Translation as :py:class:`~odc.geo.XY`

    :returns: ``(t_whole, t_subpix)``
    """
    _tt = t.map(split_float)
    whole = _tt.map(lambda x: x[0])
    part = _tt.map(lambda x: x[1])
    return whole, part


def is_affine_st(A: Affine, tol: float = 1e-10) -> bool:
    """
    Check if transfrom is pure scale and translation.

    :return: ``True`` if Affine transform has scale and translation components only
    :return: ``False`` if there is non-zero rotation or skew
    """
    (_, wx, _, wy, _, _, *_) = A

    return abs(wx) < tol and abs(wy) < tol


def snap_affine(
    A: Affine, ttol: float = 1e-3, stol: float = 1e-6, tol: float = 1e-8
) -> Affine:
    """
    Snap scale and translation parts to integer when close enough.

    When scale is less than 1 then attempt snapping to ``1/<int>``.

    If input has rotation/shear component then return unchanged.

    :param A: Affine matrix
    :param ttol: translation tolerance, defaults to 1e-3
    :param stol: scale tolerance, defaults to 1e-6
    :param tol: rotation tolerance, defaults to 1e-10

    :return: Adjusted Affine matrix.
    :return: Input Affine matrix when rotation/shear is present.
    """
    sx, wx, tx, wy, sy, ty, *_ = A

    # has rotation component
    if abs(wx) > tol or abs(wy) > tol:
        return A

    sx_ = snap_scale(sx, stol)
    sy_ = snap_scale(sy, stol)
    tx_ = maybe_int(tx, ttol)
    ty_ = maybe_int(ty, ttol)
    return Affine(sx_, 0, tx_, 0, sy_, ty_)


class Bin1D:
    """
    Class for translating continous coordinates to bin index.

    Binning is defined using following parameters:

    :param sz: Bin size (positive floating point number)
    :param origin: Location of the left edge of bin ``0``
    :param direction: Direction of the bin index ``+1|-1``
    """

    __slots__ = ("sz", "origin", "direction")

    def __init__(self, sz: float, origin: float = 0.0, direction: Literal[1, -1] = 1):
        """
        Construct :py:class:`~odc.geo.math.Bin1D` object.

        :param sz:
           Size of each bin, must be positive
        :param origin:
           Location of the left edge of bin ``0``, defaults to ``0.0``
        :param direction:
           Default is to increment bin index left to right, supply ``-1`` to go the other way
        """
        assert direction in (-1, 1)
        assert sz > 0
        self.sz = sz
        self.origin = origin
        self.direction = direction

    def __getitem__(self, idx: int) -> Tuple[float, float]:
        """Convert index to interval."""
        _x = idx * self.sz * self.direction + self.origin
        return _x, _x + self.sz

    def bin(self, x: float) -> int:
        """Lookup bin index that ``x`` falls into."""
        ix = floor((x - self.origin) / self.sz)
        return int(self.direction * ix)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Bin1D):
            return False
        return (
            self.sz == other.sz
            and self.origin == other.origin
            and self.direction == other.direction
        )

    # pylint: disable=redefined-builtin
    @staticmethod
    def from_sample_bin(
        idx: int, bin: Tuple[float, float], direction: Literal[1, -1] = 1
    ) -> "Bin1D":
        """
        Construct :py:class:`~odc.geo.math.Bin1D` from a sample.

        :param idx:
           Index of a sample bin
        :param bin:
           ``x0, x1`` bin edges
        :param direction:
           Default is to increment bin index left to right, supply ``-1`` to go the other way
        """
        x0, x1 = bin
        assert x0 < x1
        sz = x1 - x0
        origin = x0 - sz * idx * direction

        return Bin1D(sz, origin, direction=direction)
