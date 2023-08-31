from collections import abc
from typing import Tuple

import pytest

from odc.geo import ixy_, iyx_, res_, resxy_, resyx_, shape_, wh_, xy_, yx_
from odc.geo.geom import point
from odc.geo.types import func2map

# pylint: disable=use-implicit-booleaness-not-comparison


def test_basics():
    assert xy_(0, 2) == xy_([0, 2])
    assert xy_(0, 2) == xy_(tuple([0, 2]))
    assert yx_(0, 2) == xy_(2, 0)
    assert xy_(1, 0).xy == (1, 0)
    assert xy_(1, 0).lonlat == (1, 0)
    assert xy_(1, 0).yx == (0, 1)
    assert yx_([1, 0]).yx == (1, 0)
    assert xy_(1, 0).latlon == (0, 1)
    assert xy_(10, 11).x == 10
    assert xy_(10, 11).y == 11
    assert xy_(10, 11).lon == 10
    assert xy_(10, 11).lat == 11
    assert yx_(xy_(0, 1)) == xy_(yx_(1, 0))

    assert yx_(10, 20).shape == (10, 20)
    assert xy_(10, 20).wh == (10, 20)

    # equality only defined with instances of XY
    assert xy_(10, 20) != [10, 20]

    # hash
    assert len(set([xy_(0, 1), yx_(1, 0)])) == 1

    assert repr(xy_(0, 1)) == "XY(x=0, y=1)"
    assert str(xy_(0, 1)) == "XY(x=0, y=1)"

    assert ixy_(0, 3) == xy_(0, 3)
    assert iyx_(0, 3) == yx_(0, 3)

    tt = ixy_(0, 1)
    assert ixy_(tt.xy) == tt
    assert iyx_(tt.yx) == tt
    assert ixy_(tt) is tt
    assert iyx_(tt) is tt
    assert ixy_(xy_(tt.xy)) == tt
    assert iyx_(yx_(tt.yx)) == tt

    assert repr(ixy_(0, 1)) == "Index2d(x=0, y=1)"
    assert str(ixy_(0, 1)) == "Index2d(x=0, y=1)"

    assert resyx_(-10, 10) == res_(10)
    assert res_(10).x == 10
    assert res_(10).y == -10
    assert resxy_(10, -10) == resyx_(-10, 10)
    assert str(res_(10)) == "Resolution(x=10, y=-10)"
    assert repr(res_(10)) == "Resolution(x=10, y=-10)"


def test_shape2d():
    assert wh_(3, 2) == shape_((2, 3))
    assert wh_(3, 2) == (2, 3)
    wh34 = wh_(3, 4)
    assert shape_(wh34) is wh34
    assert shape_(wh34.yx) == wh34
    assert wh34 == (4, 3)
    assert shape_([4, 5]) == (4, 5)
    assert (4, 5) == shape_([4, 5])
    assert wh_(2, 3) == wh_(4, 6).shrink2()
    assert wh_(2, 3) == wh_(4, 7).shrink2()

    # should unpack like a tuple
    ny, nx = wh34
    assert (nx, ny) == wh34.xy

    # should concat like a tuple
    assert wh34 + (1,) == (4, 3, 1)
    assert ("bob",) + wh34 == ("bob", 4, 3)

    # should support len()
    assert len(wh34) == 2

    # should support indexing and slicing
    assert wh34[0] == 4
    assert wh34[::-1] == (3, 4)

    # should be a Sequence
    assert isinstance(wh34, abc.Sequence)

    # test to string
    assert str(wh34) == "Shape2d(x=3, y=4)"
    assert repr(wh34) == "Shape2d(x=3, y=4)"


def test_bad_inputs():
    # shape is valid of ints only
    with pytest.raises(ValueError):
        _ = xy_(3.1, 2).shape
    # wh is valid of ints only
    with pytest.raises(ValueError):
        _ = xy_(3.1, 2).wh

    # not enough arguments
    for op in (xy_, yx_, ixy_, iyx_):
        with pytest.raises(ValueError):
            _ = op(0)

    # making resolution from tuple should raise an error
    with pytest.raises(ValueError):
        _ = res_((-1, 2))

    with pytest.raises(ValueError):
        _ = shape_(3)


def test_map():
    assert xy_(1, 2).map(lambda x: x + 1) == xy_(2, 3)
    assert xy_(1, 2).map(lambda x: [x]) == xy_([1], [2])


def test_geom_interop():
    assert xy_(1.0, 2.0) == xy_(point(1.0, 2.0, "epsg:4326"))


def test_func_to_map() -> None:
    def aa(idx: int) -> Tuple[int, int]:
        return (idx, idx + 1)

    def first(idx: Tuple[int, ...]) -> int:
        return idx[0]

    AA = func2map(aa)
    assert len(AA) == 0
    assert list(AA) == []
    assert AA[10] == aa(10)

    FF = func2map(first)
    assert FF[3, 2] == 3
    assert FF[100, 3, 4] == 100
