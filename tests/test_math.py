import numpy as np
import pytest
from affine import Affine

from odc.geo import resxy_
from odc.geo.math import (
    Bin1D,
    affine_from_axis,
    align_down,
    align_up,
    data_resolution_and_offset,
    is_almost_int,
    maybe_int,
    maybe_zero,
    snap_scale,
)


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

    assert is_almost_int(129, 1e-6)
    assert is_almost_int(129 + 1e-8, 1e-6)
    assert is_almost_int(-129, 1e-6)
    assert is_almost_int(-129 + 1e-8, 1e-6)
    assert is_almost_int(0.3, 1e-6) is False


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

    assert Bin1D(10) != []


def test_bin1d():
    _ii = [-3, -1, 0, 1, 2, 7]
    _check_bin(Bin1D(13.3, 23.5), _ii)
    _check_bin(Bin1D(13.3, 23.5, -1), _ii)
