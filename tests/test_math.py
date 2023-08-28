import math
from typing import Tuple

import numpy as np
import pytest
from affine import Affine

from odc.geo import XY, resxy_, xy_
from odc.geo.math import (
    Bin1D,
    Poly2d,
    affine_from_axis,
    align_down,
    align_down_pow2,
    align_up,
    align_up_pow2,
    apply_affine,
    data_resolution_and_offset,
    is_almost_int,
    maybe_int,
    maybe_zero,
    quasi_random_r2,
    resolution_from_affine,
    snap_affine,
    snap_grid,
    snap_scale,
    split_float,
    split_translation,
)
from odc.geo.testutils import mkA


def test_math_ops():
    assert align_up(32, 16) == 32
    assert align_up(31, 16) == 32
    assert align_up(17, 16) == 32
    assert align_up(9, 3) == 9
    assert align_up(8, 3) == 9

    assert align_down(32, 16) == 32
    assert align_down(31, 16) == 16
    assert align_down(17, 16) == 16
    assert align_down(9, 3) == 9
    assert align_down(8, 3) == 6

    assert maybe_zero(0, 1e-5) == 0
    assert maybe_zero(1e-8, 1e-5) == 0
    assert maybe_zero(-1e-8, 1e-5) == 0
    assert maybe_zero(0.1, 1e-2) == 0.1

    assert maybe_int(37, 1e-6) == 37
    assert maybe_int(37 + 1e-8, 1e-6) == 37
    assert maybe_int(37 - 1e-8, 1e-6) == 37
    assert maybe_int(3.4, 1e-6) == 3.4
    assert str(maybe_int(float("nan"), 1e-6)) == str(float("nan"))
    assert str(maybe_int(float("inf"), 1e-6)) == str(float("inf"))
    assert str(maybe_int(float("-inf"), 1e-6)) == str(float("-inf"))

    assert is_almost_int(129, 1e-6)
    assert is_almost_int(129 + 1e-8, 1e-6)
    assert is_almost_int(-129, 1e-6)
    assert is_almost_int(-129 + 1e-8, 1e-6)
    assert is_almost_int(0.3, 1e-6) is False
    assert is_almost_int(float("nan"), 1e-6) is False
    assert is_almost_int(float("inf"), 1e-6) is False
    assert is_almost_int(float("-inf"), 1e-6) is False


def test_snap_scale():
    assert snap_scale(0) == 0

    assert snap_scale(1 + 1e-6, 1e-2) == 1
    assert snap_scale(1 - 1e-6, 1e-2) == 1

    assert snap_scale(0.5 + 1e-8, 1e-3) == 0.5
    assert snap_scale(0.5 - 1e-8, 1e-3) == 0.5
    assert snap_scale(0.5, 1e-3) == 0.5

    assert snap_scale(0.6, 1e-8) == 0.6

    assert snap_scale(3.478, 1e-6) == 3.478


def test_data_res():
    xx = np.asarray([1, 2, 3, 4])
    assert data_resolution_and_offset(xx) == (1, 0.5)
    assert data_resolution_and_offset(xx[1:]) == (1, xx[1] - 1 / 2)
    assert data_resolution_and_offset(xx[:1], 1) == (1, 0.5)

    with pytest.raises(ValueError):
        data_resolution_and_offset(xx[:1])

    with pytest.raises(ValueError):
        data_resolution_and_offset(xx[:0])


def test_affine_from_axis():
    res = 10
    x0, y0 = 111, 212
    xx = np.arange(11) * res + x0 + res / 2
    yy = np.arange(13) * res + y0 + res / 2

    assert affine_from_axis(xx, yy) == Affine(res, 0, x0, 0, res, y0)

    assert affine_from_axis(xx, yy[::-1]) == Affine(
        res, 0, x0, 0, -res, yy[-1] + res / 2
    )

    # equivalent to y:-res, x:+res
    assert affine_from_axis(xx[:1], yy[:1], res) == Affine(
        res, 0, x0, 0, -res, y0 + res
    )
    assert affine_from_axis(xx[:1], yy[:1], resxy_(res, res)) == Affine(
        res, 0, x0, 0, res, y0
    )


def _check_bin(b: Bin1D, idx, tol=1e-8, nsteps=10):
    if isinstance(idx, int):
        idx = [idx]

    for _idx in idx:
        _in, _out = b[_idx]
        for x in np.linspace(_in + tol, _out - tol, nsteps):
            assert b.bin(x) == _idx


def test_bin1d_basic():
    b = Bin1D(sz=10, origin=20)
    assert b[0] == (20, 30)
    assert b[1] == (30, 40)
    assert b[-1] == (10, 20)
    assert b.bin(20) == 0
    assert b.bin(10) == -1
    assert b.bin(20 - 0.1) == -1

    for idx in [-3, -1, 0, 1, 2, 11, 23]:
        assert Bin1D.from_sample_bin(idx, b[idx], b.direction) == b

    _check_bin(b, [-3, 5])

    b = Bin1D(sz=10, origin=20, direction=-1)
    assert b[0] == (20, 30)
    assert b[-1] == (30, 40)
    assert b[1] == (10, 20)
    assert b.bin(20) == 0
    assert b.bin(10) == 1
    assert b.bin(20 - 0.1) == 1
    _check_bin(b, [-3, 5])

    assert Bin1D(10) == Bin1D(10, 0)
    assert Bin1D(10) == Bin1D(10, 0, 1)
    assert Bin1D(11) != Bin1D(10, 0, 1)
    assert Bin1D(10, 3) != Bin1D(10, 0, 1)
    assert Bin1D(10, 0, -1) != Bin1D(10, 0, 1)

    for idx in [-3, -1, 0, 1, 2, 11, 23]:
        assert Bin1D.from_sample_bin(idx, b[idx], b.direction) == b

    assert Bin1D(10) != ["something"]


def test_bin1d():
    _ii = [-3, -1, 0, 1, 2, 7]
    _check_bin(Bin1D(13.3, 23.5), _ii)
    _check_bin(Bin1D(13.3, 23.5, -1), _ii)


def test_apply_affine():
    A = mkA(rot=10, scale=(3, 1.3), translation=(-100, +2.3))
    xx, yy = np.meshgrid(np.arange(13), np.arange(11))

    xx_, yy_ = apply_affine(A, xx, yy)

    assert xx_.shape == xx.shape
    assert yy_.shape == xx.shape

    xy_expect = [A * (x, y) for x, y in zip(xx.ravel(), yy.ravel())]
    xy_got = list(zip(xx_.ravel(), yy_.ravel()))

    np.testing.assert_array_almost_equal(xy_expect, xy_got)


def test_split_translation():
    def verify(
        a: Tuple[XY[float], XY[float]],
        b: Tuple[XY[float], XY[float]],
    ):
        assert a[0].xy == pytest.approx(b[0].xy)
        assert a[1].xy == pytest.approx(b[1].xy)

    def tt(
        tx: float, ty: float, e_whole: Tuple[float, float], e_part: Tuple[float, float]
    ):
        expect = xy_(e_whole), xy_(e_part)
        rr = split_translation(xy_(tx, ty))
        verify(rr, expect)

    # fmt: off
    assert split_translation(xy_( 1,  2)) == (xy_( 1,  2), xy_(0, 0))
    assert split_translation(xy_(-1, -2)) == (xy_(-1, -2), xy_(0, 0))
    tt( 1.3, 2.5 , ( 1, 2), ( 0.3,  0.5 ))
    tt( 1.1, 2.6 , ( 1, 3), ( 0.1, -0.4 ))
    tt(-1.1, 2.8 , (-1, 3), (-0.1, -0.2 ))
    tt(-1.9, 2.05, (-2, 2), (+0.1,  0.05))
    tt(-1.5, 2.45, (-1, 2), (-0.5,  0.45))
    # fmt: on


def test_snap_affine():
    A = mkA(rot=0.1)
    assert snap_affine(A) is A

    assert snap_affine(mkA(translation=(10, 20))) == mkA(translation=(10, 20))

    assert snap_affine(mkA(translation=(10.1, 20.1)), ttol=0.2) == mkA(
        translation=(10, 20)
    )

    assert snap_affine(
        mkA(scale=(3.3, 4.2), translation=(10.1, 20.1)), ttol=0.2
    ) == mkA(scale=(3.3, 4.2), translation=(10, 20))

    assert snap_affine(
        mkA(scale=(3 + 1e-6, 4 - 1e-6), translation=(10.1, 20.1)), ttol=0.2, stol=1e-3
    ) == mkA(scale=(3, 4), translation=(10, 20))

    assert snap_affine(
        mkA(scale=(1 / 2 + 1e-8, 1 / 3 - 1e-8), translation=(10.1, 20.1)),
        ttol=0.2,
        stol=1e-3,
    ) == mkA(scale=(1 / 2, 1 / 3), translation=(10, 20))


@pytest.mark.parametrize(
    "left, right, res, off, expect",
    [
        (20, 30, 10, 0, (20, 1)),
        (20, 30.5, 10, 0, (20, 2)),
        (20, 31.5, 10, 0, (20, 2)),
        (20, 30, 10, 3, (13, 2)),
        (20, 30, -10, 0, (30, 1)),
        (19.5, 30, -10, 0, (30, 2)),
        (18.5, 30, -10, 0, (30, 2)),
        (20, 30, -10, 3, (33, 2)),
    ],
)
def test_snap_grid(left, right, res, off, expect):
    assert snap_grid(left, right, res, off / abs(res)) == expect


def test_snap_grid_tol():
    assert snap_grid(0.95, 10.12, 1, 0, 0.1) == (1.0, 10)


def test_res_affine():
    assert resolution_from_affine(mkA(scale=(2, 3))).xy == (2, 3)
    assert resolution_from_affine(mkA(rot=10, scale=(2, 3))).xy == pytest.approx((2, 3))
    assert resolution_from_affine(mkA(rot=-45, scale=(10, -10))).xy == pytest.approx(
        (10, -10)
    )


@pytest.mark.parametrize(
    "ab, expect_a, expect_b",
    [
        (1.0, 1, 0),
        (1.3, 1, 0.3),
        (0.99, 1, -0.01),
        (0.6, 1, -0.4),
        (100.6, 101, -0.4),
        (2000.3, 2000, 0.3),
        (float("nan"), float("nan"), 0),
        (float("inf"), float("inf"), 0),
    ],
)
def test_split_float(ab, expect_a, expect_b):
    a, b = split_float(ab)

    assert b == pytest.approx(expect_b)
    if np.isnan(expect_a):
        assert np.isnan(a)
        return

    assert a == pytest.approx(expect_a)
    assert (a + b) == pytest.approx(ab)


@pytest.mark.parametrize(
    "n",
    [3, 1, 10, 100],
)
@pytest.mark.parametrize("ny", [30, 101])
@pytest.mark.parametrize("nx", [20, 104])
@pytest.mark.parametrize("offset", [0, 1, 103])
def test_quasi_random_r2(n, ny, nx, offset):
    xx = quasi_random_r2(n)
    assert xx.shape == (n, 2)
    assert xx.min() >= 0
    assert xx.max() < 1

    np.testing.assert_array_equal(xx, quasi_random_r2(n))

    yy = quasi_random_r2(n, offset=offset)
    assert yy.shape == (n, 2)
    assert yy.min() >= 0
    assert yy.max() < 1
    np.testing.assert_array_equal(yy, quasi_random_r2(n, offset=offset))
    if offset == 0:
        np.testing.assert_array_equal(xx, yy)
    else:
        assert not (xx == yy).all()

    xx = quasi_random_r2(n, shape=(ny, nx))
    assert xx.min() >= 0
    assert xx[:, 0].max() < nx
    assert xx[:, 1].max() < ny


@pytest.mark.parametrize(
    "pts",
    [
        quasi_random_r2(20),
        quasi_random_r2(4, offset=10),
        quasi_random_r2(8, offset=10),
        quasi_random_r2(3, offset=20),
    ],
)
def test_poly2d(pts):
    N = pts.shape[0]
    x, y = pts.T

    p = Poly2d.fit(pts, pts)

    assert p(pts).shape == pts.shape
    assert p(x, y).shape == (2, N)

    np.testing.assert_array_almost_equal(pts, p(pts))

    assert p.grid2d(np.arange(3), np.arange(4)).shape == (2, 3, 4)

    # With input transform check
    A = Affine.scale(0.93) * Affine.translation(10, -13.2)
    p_ = p.with_input_transform(A)
    x_, y_ = apply_affine(~A, x, y)
    xx, yy = p_(x_, y_)
    np.testing.assert_array_almost_equal(x, xx)
    np.testing.assert_array_almost_equal(y, yy)

    # With input transform check, rotation
    A = Affine.rotation(-133)
    p_ = p.with_input_transform(A)
    x_, y_ = apply_affine(~A, x, y)
    xx, yy = p_(x_, y_)
    np.testing.assert_array_almost_equal(x, xx)
    np.testing.assert_array_almost_equal(y, yy)


def test_poly2d_not_enough_points():
    pts = quasi_random_r2(2)

    with pytest.raises(ValueError):
        _ = Poly2d.fit(pts[:2], pts[:2])


@pytest.mark.parametrize(
    "x",
    [
        *[2**n for n in range(20)],
        *[2**n - 1 for n in range(20)],
        *[2**n + 1 for n in range(20)],
    ],
)
def test_align_up_pow2(x: int):
    y = align_up_pow2(x)
    assert isinstance(y, int)
    assert y >= x
    assert 2 ** int(math.log2(y)) == y


@pytest.mark.parametrize(
    "x",
    [
        *[2**n for n in range(20)],
        *[2**n - 1 for n in range(1, 20)],
        *[2**n + 1 for n in range(20)],
    ],
)
def test_align_down_pow2(x: int):
    y = align_down_pow2(x)
    assert isinstance(y, int)
    assert y <= x
    assert 2 ** int(math.log2(y)) == y
