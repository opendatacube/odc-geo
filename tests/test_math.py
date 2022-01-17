import numpy as np
import pytest
from affine import Affine

from odc.geo.math import (
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

    assert affine_from_axis(xx[:1], yy[:1], res) == Affine(res, 0, x0, 0, res, y0)
    assert affine_from_axis(xx[:1], yy[:1], (res, res)) == Affine(
        res, 0, x0, 0, res, y0
    )
