import pytest

from odc.geo import ixy_, iyx_, res_, resxy_, resyx_, xy_, yx_


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
