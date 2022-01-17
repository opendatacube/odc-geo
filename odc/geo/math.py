# This file is part of the Open Data Cube, see https://opendatacube.org for more information
#
# Copyright (c) 2015-2020 ODC Contributors
# SPDX-License-Identifier: Apache-2.0
from math import fmod
from typing import Tuple, Union

import numpy as np
from affine import Affine


def maybe_zero(x: float, tol: float) -> float:
    """Turn almost zeros to actual zeros"""
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
    :return: whole, fraction
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
    """Turn almost ints to actual ints, pass through other values unmodified"""

    x_whole, x_part = split_float(x)

    if abs(x_part) < tol:  # almost int
        return int(x_whole)
    return x


def snap_scale(s, tol=1e-6):
    """Snap scale to the nearest integer and simple fractions in the form 1/<int>"""
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
    return x - (x % align)


def align_up(x: int, align: int) -> int:
    return align_down(x + (align - 1), align)


def clamp(x, lo, up):
    """
    clamp x to be lo <= x <= up
    """
    assert lo <= up
    return lo if x < lo else up if x > up else x


def is_almost_int(x: float, tol: float):
    """
    Check if number is close enough to an integer
    """
    x = abs(fmod(x, 1))
    if x > 0.5:
        x = 1 - x
    return x < tol


def data_resolution_and_offset(data, fallback_resolution=None):
    """Compute resolution and offset from x/y axis data.

    Only uses first two coordinate values, assumes that data is regularly
    sampled.

    Returns
    =======
    (resolution: float, offset: float)
    """
    if data.size < 2:
        if data.size < 1:
            raise ValueError("Can't calculate resolution for empty data")
        if fallback_resolution is None:
            raise ValueError("Can't calculate resolution with data size < 2")
        res = fallback_resolution
    else:
        res = (data[data.size - 1] - data[0]) / (data.size - 1.0)
        res = res.item()

    off = data[0] - 0.5 * res
    return res, off.item()


def affine_from_axis(xx, yy, fallback_resolution=None):
    """ Compute Affine transform from pixel to real space given X,Y coordinates.

        :param xx: X axis coordinates
        :param yy: Y axis coordinates
        :param fallback_resolution: None|float|(resx:float, resy:float) resolution to
                                    assume for single element axis.

        (0, 0) in pixel space is defined as top left corner of the top left pixel
            \
            `` 0   1
             +---+---+
           0 |   |   |
             +---+---+
           1 |   |   |
             +---+---+

        Only uses first two coordinate values, assumes that data is regularly
        sampled.

        raises ValueError when any axis is empty
        raises ValueError when any axis has single value and fallback resolution was not supplied.
    """
    if fallback_resolution is not None:
        if isinstance(fallback_resolution, (float, int)):
            frx, fry = fallback_resolution, fallback_resolution
        else:
            frx, fry = fallback_resolution
    else:
        frx, fry = None, None

    xres, xoff = data_resolution_and_offset(xx, frx)
    yres, yoff = data_resolution_and_offset(yy, fry)

    return Affine.translation(xoff, yoff) * Affine.scale(xres, yres)


def apply_affine(
    A: Affine, x: np.ndarray, y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    broadcast A*(x_i, y_i) across all elements of x/y arrays in any shape (usually 2d image)
    """

    shape = x.shape

    A = np.asarray(A).reshape(3, 3)
    t = A[:2, -1].reshape((2, 1))
    A = A[:2, :2]

    x, y = A @ np.vstack([x.ravel(), y.ravel()]) + t
    x, y = (a.reshape(shape) for a in (x, y))
    return (x, y)


def split_translation(t):
    """
    Split translation into pixel aligned and sub-pixel components.

    Subpixel translation is guaranteed to be in [-0.5, +0.5] range.

    >  x + t = x + t_whole + t_subpix

    :param t: (float, float)

    :returns: (t_whole: (float, float), t_subpix: (float, float))
    """

    _tt = [split_float(x) for x in t]
    return tuple(t[0] for t in _tt), tuple(t[1] for t in _tt)


def is_affine_st(A, tol=1e-10):
    """
    True if Affine transform has scale and translation components only.
    """
    (_, wx, _, wy, _, _, *_) = A

    return abs(wx) < tol and abs(wy) < tol
