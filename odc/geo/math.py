# This file is part of the Open Data Cube, see https://opendatacube.org for more information
#
# Copyright (c) 2015-2020 ODC Contributors
# SPDX-License-Identifier: Apache-2.0
"""
Various mathy helpers.

Minimal dependencies in this module.
"""
from math import ceil, floor, fmod, isfinite, log2
from typing import (
    Any,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np
from affine import Affine
from numpy.polynomial.polynomial import polygrid2d, polyval2d

from .types import XY, Resolution, SomeResolution, SomeShape, res_, resxy_, shape_, xy_

AffineX = TypeVar("AffineX", np.ndarray, Affine)


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
    if not isfinite(x):
        return (x, 0)

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
    if not isfinite(x):
        return x

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


def align_up_pow2(x: int) -> int:
    """
    Align up to the nearest power of 2.

    :return:
       Smallest ``y`` such that ``y >= x`` and ``y == 2**n`` for some integer ``n``.
    """
    if x <= 0:
        return 1
    return 2 ** int(ceil(log2(x)))


def align_down_pow2(x: int) -> int:
    """
    Align down to the nearest power of 2.

    :return:
       Largest ``y`` such that ``y <= x`` and ``y == 2**n`` for some integer ``n``.
    """
    y = align_up_pow2(x)
    if y > x:
        y = y // 2
    return y


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
    if not isfinite(x):
        return False

    x = abs(fmod(x, 1))
    if x > 0.5:
        x = 1 - x
    return x < tol


def _snap_edge_pos(x0: float, x1: float, res: float, tol: float) -> Tuple[float, int]:
    assert res > 0
    assert x1 >= x0
    _x0 = floor(maybe_int(x0 / res, tol))
    _x1 = ceil(maybe_int(x1 / res, tol))
    nx = max(1, _x1 - _x0)
    return _x0 * res, nx


def _snap_edge(x0: float, x1: float, res: float, tol: float) -> Tuple[float, int]:
    assert x1 >= x0
    if res > 0:
        return _snap_edge_pos(x0, x1, res, tol)
    _tx, nx = _snap_edge_pos(x0, x1, -res, tol)
    tx = _tx + nx * (-res)
    return tx, nx


def snap_grid(
    x0: float, x1: float, res: float, off_pix: Optional[float] = 0, tol: float = 1e-6
) -> Tuple[float, int]:
    """
    Compute grid snapping for single axis.

    :param x0: In point ``x0 <= x1``
    :param x1: Out point ``x0 <= x1``
    :param res: Pixel size and direction (can be negative)
    :param off_pix:
       Pixel fraction to align to ``x=0``.
       0 - edge aligned
       0.5 - center aligned
       None - don't snap

    :return: ``tx, nx`` that defines 1-d grid, such that ``x0`` and ``x1`` are within edge pixels.
    """
    assert (off_pix is None) or (0 <= off_pix < 1)
    if off_pix is None:
        if res > 0:
            nx = ceil(maybe_int((x1 - x0) / res, tol))
            return x0, max(1, nx)
        nx = ceil(maybe_int((x1 - x0) / (-res), tol))
        return x1, max(nx, 1)

    off = off_pix * abs(res)
    _tx, nx = _snap_edge(x0 - off, x1 - off, res, tol)
    return _tx + off, nx


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


     :return: Rotation, Shear, Scale ``2x2`` matrices
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

    WS = np.linalg.cholesky(A.T @ A).T
    R = A @ np.linalg.inv(WS)

    if np.linalg.det(R) < 0:
        R[:, -1] *= -1
        WS[-1, :] *= -1

    ss = np.diag(WS)
    S = np.diag(ss)
    W = WS @ np.diag(1.0 / ss)

    return R, W, S


def stack_xy(pts: Sequence[XY[float]]) -> np.ndarray:
    """Turn into an ``Nx2`` ndarray of floats in X,Y order."""
    return np.vstack([pt.xy for pt in pts])


def unstack_xy(pts: np.ndarray) -> List[XY[float]]:
    """Turn ``Nx2`` array in X,Y order into a list of XY points."""
    assert pts.ndim == 2 and pts.shape[1] == 2
    return [xy_(pt) for pt in pts]


def norm_xy(
    pts: np.ndarray, out: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, Affine]:
    """
    Normalize ``Nx2`` points.

    Scale and translate such that mean is at 0 and mean distance from 0 is sqrt(2).

    :returns: Normalized array and affine matrix of the normalization.
    """
    assert pts.ndim == 2 and pts.shape[1] == 2
    assert out is None or out.shape == pts.shape

    _mean = pts.mean(axis=0)
    XX = np.subtract(pts, _mean, out=out)

    sx = (((XX**2).sum(axis=1) * 0.5) ** -0.5).mean()
    XX *= sx

    tx, ty = -_mean * sx
    A = Affine(sx, 0, tx, 0, sx, ty)

    return XX, A


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

    mm, *_ = np.linalg.lstsq(XX, YY, rcond=-1)
    a, d, b, e, c, f = mm.ravel()

    return Affine(a, b, c, d, e, f)


def resolution_from_affine(A: Affine) -> Resolution:
    """
    Compute resolution from Affine matrix.

    Deals with rotated case when needed.
    """
    if is_affine_st(A):
        rx, _, _, _, ry, *_ = A
        return resxy_(rx, ry)
    _, _, A_ = decompose_rws(A)
    rx, _, _, _, ry, *_ = A_
    return resxy_(rx, ry)


def edge_index(shape: SomeShape, closed: bool = False) -> Iterator[Tuple[int, int]]:
    """
    Like ``ndindex`` in numpy but for edge locations and limited to 2d.

    Returns a sequence of indexes along the edge of a 2d array of a given shape.
    Order is fixed, starting from ``(0, 0), (0, 1)...``. See example below

    .. code-block::

       # for shape=(10, 8)
       (0, 0), (0, 1) ... (0, 7),
       (1, 7), (2, 7) ... (9, 7),
       (9, 6), (9, 5) ... (9, 0),
       (8, 0), (7, 0) ... (1, 0),
       (0, 0) # Back to (0, 0) if closed=True

    """
    nx, ny = shape_(shape).xy
    ix, iy = 0, 0
    for ix in range(nx):
        yield (0, ix)
    for iy in range(1, ny):
        yield (iy, ix)
    for ix in range(ix - 1, -1, -1):
        yield (iy, ix)
    for iy in range(iy - 1, 0, -1):
        yield (iy, ix)
    if closed:
        yield (0, 0)


def quasi_random_r2(
    n: int,
    shape: Optional[SomeShape] = None,
    offset: int = 0,
) -> np.ndarray:
    """
    Generate quasi-random set of points.

    Returns quasi-random points in ``[0, 1)`` range. If ``shape=(ny, nx)`` is
    supplied, points are scaled to ``[0, nx), [0, ny)`` range.

    :param n: Number of points to generate
    :param offset: Generate from that offset in the sequence
    :returns: ``nx2`` numpy array

    References: http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/
    """
    idx = np.arange(offset, offset + n, dtype="float32")
    aa = (0.7548776662466927, 0.5698402909980532)
    xy = np.fmod(np.outer(idx, aa), 1)

    if shape is not None:
        nx, ny = shape_(shape).wh
        xy[:, 0] *= nx
        xy[:, 1] *= ny

    return xy


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


class Poly2d:
    """
    Wrapper around polyval2d with input normalization.
    """

    def __init__(self, cc: np.ndarray, A: Affine) -> None:
        assert cc.shape in [(3, 3, 2), (2, 2, 2)]
        tol = 1e-6
        self._cc = cc
        self._A = A
        self._safe_to_grid = False

        sx, zx, tx, zy, sy, ty, *_ = A
        if abs(zx) < tol and abs(zy) < tol:
            self._norm = lambda x, y: (np.polyval([sx, tx], x), np.polyval([sy, ty], y))
            self._safe_to_grid = True
        else:
            self._norm = lambda x, y: A * (x, y)

    def __call__(self, x: Any, y: Any = None) -> Any:
        """
        Evaluate at points (x, y).

        When y is None, x is assumed to be Nx2 array.
        """
        if y is None:
            return self.__call__(x[..., 0], x[..., 1]).T
        x, y = self._norm(x, y)
        return polyval2d(x, y, self._cc)

    def grid2d(self, x: Any, y: Any) -> Any:
        """
        Evaluate on the Cartesian product of x and y.
        """
        assert self._safe_to_grid

        x, y = self._norm(x, y)
        return polygrid2d(x, y, self._cc)

    def with_input_transform(self, A: Affine) -> "Poly2d":
        """
        Make a new transform by chaining a linear mapping ``A`` on input side.

        usefull when cropping original image with GCPs.
        """
        return Poly2d(self._cc, self._A * A)

    @staticmethod
    def fit(aa: np.ndarray, bb: np.ndarray) -> "Poly2d":
        """
        Fit 2d polynomial that minimizes ``poly(aa) - bb``.

        Where ``aa``,``bb`` are point correspondences of ``Nx2`` shape.
        """
        assert aa.shape[1] == 2
        assert aa.shape == bb.shape
        N = aa.shape[0]
        if N < 3:
            raise ValueError(f"Need at least 3 points, got {N}")

        aa_, Ain = norm_xy(aa)
        bb_, Ab = norm_xy(bb)

        if N >= 9:
            return Poly2d._fit9(aa_, Ain, bb_, Ab)

        if N >= 4:
            return Poly2d._fit4(aa_, Ain, bb_, Ab)

        return Poly2d._fit3(aa_, Ain, bb_, Ab)

    @staticmethod
    def _fit9(aa: np.ndarray, Ain: Affine, bb: np.ndarray, Ab: Affine) -> "Poly2d":
        N = aa.shape[0]
        assert N >= 9
        x, y = aa.T
        AA = np.empty((N, 9), dtype="float64")

        #   1,     y,     y^2
        #   x,   x*y,   x*y^2
        # x^2, x^2*y, x^2*y^2

        AA[:, 0] = 1
        AA[:, 1] = y
        AA[:, 2] = y * y

        AA[:, 3] = x
        AA[:, 4] = x * y
        AA[:, 5] = AA[:, 4] * y

        AA[:, 6] = x * x
        AA[:, 7] = AA[:, 6] * y
        AA[:, 8] = AA[:, 7] * y

        cc, *_ = np.linalg.lstsq(AA, bb, rcond=-1)

        # denorm output side, assumes `sx==sy`
        s, _, tx, _, _, ty, *_ = ~Ab
        cc = cc * s
        cc[0, :2] += (tx, ty)

        cc = cc.reshape(3, 3, 2)

        return Poly2d(cc, Ain)

    @staticmethod
    def _fit4(aa: np.ndarray, Ain: Affine, bb: np.ndarray, Ab: Affine) -> "Poly2d":
        N = aa.shape[0]
        assert N >= 4
        x, y = aa.T
        AA = np.empty((N, 4), dtype="float64")

        #   1,     y
        #   x,   x*y

        AA[:, 0] = 1
        AA[:, 1] = y

        AA[:, 2] = x
        AA[:, 3] = x * y

        cc, *_ = np.linalg.lstsq(AA, bb, rcond=-1)
        assert cc.shape == (4, 2)

        # denorm output side, assumes `sx==sy`
        s, _, tx, _, _, ty, *_ = ~Ab
        cc = cc * s
        cc[0, :2] += (tx, ty)

        cc = cc.reshape(2, 2, 2)

        return Poly2d(cc, Ain)

    @staticmethod
    def _fit3(aa: np.ndarray, Ain: Affine, bb: np.ndarray, Ab: Affine) -> "Poly2d":
        N = aa.shape[0]
        assert N >= 3
        x, y = aa.T
        AA = np.empty((N, 3), dtype="float64")

        #   1,   y
        #   x,   _

        # 1, y, x
        AA[:, 0] = 1
        AA[:, 1] = y
        AA[:, 2] = x

        cc, *_ = np.linalg.lstsq(AA, bb, rcond=-1)
        assert cc.shape == (3, 2)

        # denorm output side, assumes `sx==sy`
        s, _, tx, _, _, ty, *_ = ~Ab
        cc = cc * s
        cc[0, :2] += (tx, ty)
        cc = np.vstack([cc, np.asarray([0, 0])]).reshape(2, 2, 2)

        return Poly2d(cc, Ain)
