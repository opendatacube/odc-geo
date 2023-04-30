import numpy as np
import pytest

from odc.geo.roi import (
    RoiTiles,
    Tiles,
    VariableSizedTiles,
    _norm_slice_or_error,
    clip_tiles,
    polygon_path,
    roi_boundary,
    roi_center,
    roi_from_points,
    roi_intersect,
    roi_is_empty,
    roi_is_full,
    roi_normalise,
    roi_pad,
    roi_shape,
    roi_tiles,
    scaled_down_roi,
    scaled_down_shape,
    scaled_up_roi,
    slice_intersect3,
    w_,
)


def test_roi_tools():
    s_ = np.s_

    assert roi_shape(s_[2:4, 3:4]) == (2, 1)
    assert roi_shape(s_[:4, :7]) == (4, 7)
    assert roi_shape(s_[3, :7]) == (1, 7)

    assert roi_is_empty(s_[:4, :5]) is False
    assert roi_is_empty(s_[1:1, :10]) is True
    assert roi_is_empty(s_[7:3, :10]) is True

    assert roi_is_empty(s_[:3]) is False
    assert roi_is_empty(s_[4:4]) is True

    assert roi_is_full(s_[:3], 3) is True
    assert roi_is_full(s_[:3, 0:4], (3, 4)) is True
    assert roi_is_full(s_[:, 0:4], (33, 4)) is True
    assert roi_is_full(s_[1:3, 0:4], (3, 4)) is False
    assert roi_is_full(s_[1:3, 0:4], (2, 4)) is False
    assert roi_is_full(s_[0:4, 0:4], (3, 4)) is False
    assert roi_is_full(s_[0], 1) is True
    assert roi_is_full(s_[2], 3) is False

    roi = s_[0:8, 0:4]
    roi_ = scaled_down_roi(roi, 2)
    assert roi_shape(roi_) == (4, 2)
    assert scaled_down_roi(scaled_up_roi(roi, 3), 3) == roi

    assert scaled_down_shape(roi_shape(roi), 2) == roi_shape(scaled_down_roi(roi, 2))

    assert roi_shape(scaled_up_roi(roi, 10000, (40, 50))) == (40, 50)

    for bad_roi in [np.s_[1:], np.s_[:], np.s_[-3:]]:
        with pytest.raises(ValueError):
            _ = roi_shape(bad_roi)

    assert roi_normalise(s_[3:4], 40) == s_[3:4]
    assert roi_normalise(s_[3], 40) == s_[3:4]
    assert roi_normalise(s_[:4], (40,)) == s_[0:4]
    assert roi_normalise(s_[:], (40,)) == s_[0:40]
    assert roi_normalise(s_[:-1], (3,)) == s_[0:2]
    assert roi_normalise(-1, (3,)) == s_[2:3]
    assert roi_normalise(s_[-1:], (3,)) == s_[2:3]
    assert roi_normalise((s_[:-1],), 3) == (s_[0:2],)
    assert roi_normalise(s_[-2:-1, :], (10, 20)) == s_[8:9, 0:20]
    assert roi_normalise(s_[-2:-1, :, 3:4], (10, 20, 100)) == s_[8:9, 0:20, 3:4]
    assert roi_center(s_[0:3]) == 1.5
    assert roi_center(s_[0:2, 0:6]) == (1, 3)


def test_roi_from_points():
    roi = np.s_[0:2, 4:13]
    xy = roi_boundary(roi)

    assert xy.shape == (4, 2)
    assert roi_from_points(xy, (2, 13)) == roi

    xy = np.asarray(
        [
            [0.2, 1.9],
            [10.3, 21.2],
            [float("nan"), 11],
            [float("inf"), float("-inf")],
        ]
    )
    assert roi_from_points(xy, (100, 100)) == np.s_[1:22, 0:11]
    assert roi_from_points(xy, (5, 7)) == np.s_[1:5, 0:7]
    assert roi_from_points(xy[2:, :], (3, 3)) == np.s_[0:0, 0:0]


@pytest.mark.parametrize(
    "a,b",
    [
        np.s_[0:3, :3],
        np.s_[:4, 2:6],
        np.s_[4:13, 5:17],
        np.s_[10:13, 3:7],
        np.s_[10:13, 13:17],
        np.s_[10:13, 14:17],
    ],
)
def test_slice_intersect3(a: slice, b: slice):
    assert isinstance(a.stop, int)
    assert isinstance(b.stop, int)
    _a, _b, _ab = slice_intersect3(a, b)

    (na,) = roi_shape(a)
    (nb,) = roi_shape(b)

    assert _a.start <= _a.stop
    assert 0 <= _a.start <= na
    assert 0 <= _a.stop <= na

    assert _b.start <= _b.stop
    assert 0 <= _b.start <= nb
    assert 0 <= _b.stop <= nb

    X = np.arange(max(a.stop, b.stop))
    np.testing.assert_array_equal(X[a][_a], X[b][_b])
    np.testing.assert_array_equal(X[a][_a], X[_ab])


def test_roi_intersect():
    s_ = np.s_
    roi = s_[0:2, 4:13]

    assert roi_intersect(roi, roi) == roi
    assert roi_intersect(s_[0:3], s_[1:7]) == s_[1:3]
    assert roi_intersect(s_[0:3], (s_[1:7],)) == s_[1:3]
    assert roi_intersect((s_[0:3],), s_[1:7]) == (s_[1:3],)

    assert roi_intersect(s_[4:7, 5:6], s_[0:1, 7:8]) == s_[4:4, 6:6]


def test_roi_pad():
    s_ = np.s_
    assert roi_pad(s_[0:4], 1, 4) == s_[0:4]
    assert roi_pad(s_[0:4], 1, (4,)) == s_[0:4]
    assert roi_pad((s_[0:4],), 1, 4) == (s_[0:4],)

    assert roi_pad(s_[0:4, 1:5], 1, (4, 6)) == s_[0:4, 0:6]
    assert roi_pad(s_[2:3, 1:5], 10, (7, 9)) == s_[0:7, 0:9]
    assert roi_pad(s_[3, 0, :2], 1, (100, 100, 100)) == s_[2:5, 0:2, 0:3]


def test_norm_slice_or_error():
    s_ = np.s_
    assert _norm_slice_or_error(s_[0]) == s_[0:1]
    assert _norm_slice_or_error(s_[3]) == s_[3:4]
    assert _norm_slice_or_error(s_[:3]) == s_[0:3]
    assert _norm_slice_or_error(s_[10:100:3]) == s_[10:100:3]

    for bad in [np.s_[1:], np.s_[:-3], np.s_[-3:], np.s_[-2:10], -3]:
        with pytest.raises(ValueError):
            _ = _norm_slice_or_error(bad)


def test_window_from_slice():
    s_ = np.s_

    assert w_[None] is None
    assert w_[s_[:3, 4:5]] == ((0, 3), (4, 5))
    assert w_[s_[0:3, :5]] == ((0, 3), (0, 5))
    assert w_[list(s_[0:3, :5])] == ((0, 3), (0, 5))

    for roi in [s_[:3], s_[:3, :4, :5], 0]:
        with pytest.raises(ValueError):
            _ = w_[roi]


def test_polygon_path():
    pp = polygon_path([0, 1])
    assert pp.shape == (2, 5)
    assert set(pp.ravel()) == {0, 1}

    pp2 = polygon_path([0, 1], [0, 1])
    assert (pp2 == pp).all()

    pp = polygon_path([0, 1], [2, 3])
    assert set(pp[0].ravel()) == {0, 1}
    assert set(pp[1].ravel()) == {2, 3}


def test_tiles():
    tt = Tiles((10, 20), (3, 7))
    assert tt.tile_shape((0, 0)) == (3, 7)
    assert tt.tile_shape((3, 2)) == (1, 6)
    assert tt.shape.yx == (4, 3)
    assert tt.base.yx == (10, 20)
    assert tt.chunks == ((3, 3, 3, 1), (7, 7, 6))

    assert tt[0, 0] == np.s_[0:3, 0:7]
    assert tt[3, 2] == np.s_[9:10, 14:20]
    assert tt[-1, -1] == tt[3, 2]
    assert tt[-4, -3] == tt[0, 0]
    assert tt.tile_shape((-1, -1)) == tt.tile_shape((3, 2))
    assert tt.tile_shape((-4, -3)) == tt.tile_shape((0, 0))

    assert tt[0, 0] == tt[:1, :1]
    assert tt[:, :] == np.s_[0:10, 0:20]
    assert tt[0:2, 0] == np.s_[0:6, 0:7]
    assert tt[1:4, -1:] == np.s_[3:10, 14:20]
    assert tt[1:, -1:] == np.s_[3:10, 14:20]

    assert tt == tt
    assert tt != Tiles((1, 1), (1, 1))
    assert tt != {"": 3}

    tt_ = VariableSizedTiles(tt.chunks)
    assert tt_.shape == tt.shape
    assert tt_.tile_shape((0, 0)) == (3, 7)
    assert tt_.tile_shape((3, 2)) == (1, 6)
    assert tt_.shape.yx == (4, 3)
    assert tt_.base.yx == (10, 20)
    assert tt_.chunks == ((3, 3, 3, 1), (7, 7, 6))
    assert tt[-1, -1] == tt[3, 2]
    assert tt[-4, -3] == tt[0, 0]
    assert tt.tile_shape((-1, -1)) == tt.tile_shape((3, 2))
    assert tt.tile_shape((-4, -3)) == tt.tile_shape((0, 0))

    assert isinstance(roi_tiles(tt.shape, (1, 2)), Tiles)
    assert isinstance(roi_tiles(tt.shape, tt.shape), Tiles)
    assert isinstance(roi_tiles(tt.shape, tt.chunks), VariableSizedTiles)

    # smoke test repr/str
    assert isinstance(repr(tt), str)
    assert isinstance(repr(tt_), str)
    assert "100_123x200_321" in repr(Tiles((100_123, 200_321), (1000, 1000)))


@pytest.mark.parametrize(
    "chunks",
    [
        ((1,), (2,)),
        ((1, 10, 3), (2, 5, 7, 11, 2)),
    ],
)
def test_varsz_tiles(chunks):
    iy, ix = chunks
    tt = VariableSizedTiles(chunks)

    assert tt.shape == (len(iy), len(ix))
    assert tt.base == (sum(iy), sum(ix))
    assert tt == tt
    assert (tt == "") is False
    assert tt != [9]

    # test comparison with different chunk sizes
    _chunks = tuple(ch + (1,) for ch in chunks)
    assert tt != VariableSizedTiles(_chunks)

    # test comparison with different chunk values
    _chunks = tuple((ch[0] + 3, *ch[1:]) for ch in chunks)
    assert tt != VariableSizedTiles(_chunks)

    for idx in np.ndindex(tt.shape.yx):
        y, x = idx
        assert tt.tile_shape(idx) == (iy[y], ix[x])
        assert (
            tt[idx]
            == np.s_[
                sum(iy[:y]) : sum(iy[: y + 1]),
                sum(ix[:x]) : sum(ix[: x + 1]),
            ]
        )
        # test slice version
        assert tt[y, x] == tt[y : y + 1, x : x + 1]
        assert tt[y, x] == tt[y, x : x + 1]
        assert tt[y, x] == tt[y : y + 1, x]
        assert tt[:y, x:] == np.s_[0 : sum(iy[:y]), sum(ix[:x]) : sum(ix)]

    assert isinstance(tt.__dask_tokenize__(), tuple)


@pytest.mark.parametrize(
    "tile",
    [
        VariableSizedTiles(((10, 1, 30), (2, 4, 5, 6))),
        VariableSizedTiles(((10, 1, 30), (2, 4, 7, 13))),
        Tiles((104, 201), (11, 23)),
    ],
)
@pytest.mark.parametrize(
    "roi",
    [
        np.s_[:1, :1],
        np.s_[1:2, 2:],
        np.s_[-1:, 0],
    ],
)
def test_tiles_crop(tile, roi):
    assert isinstance(tile.crop(roi), type(tile))

    # tile.chunks[roi] == tile[roi].chunks
    _roi = roi_normalise(roi, tile.shape.yx)
    expect_chunks = tuple(ch[s.start : s.stop] for ch, s in zip(tile.chunks, _roi))
    assert tile.crop(roi).chunks == expect_chunks


@pytest.mark.parametrize(
    "tiles",
    [
        VariableSizedTiles(((10, 1, 30), (2, 4, 5, 6))),
        VariableSizedTiles(((10, 1, 30), (2, 4, 7, 13))),
        Tiles((104, 201), (11, 23)),
    ],
)
def test_clip_tiles(tiles: RoiTiles):
    ny, nx = tiles.shape.yx
    tl, tr, br, bl = (0, 0), (0, nx - 1), (ny - 1, nx - 1), (ny - 1, 0)

    tt, roi, idx = clip_tiles(tiles, [tl, tr, br, bl])
    assert tt == tiles
    assert roi == np.s_[0:ny, 0:nx]
    assert idx == [tl, tr, br, bl]


@pytest.mark.parametrize(
    "tiles",
    [
        VariableSizedTiles(((10, 1, 30), (2, 4, 5, 6))),
        VariableSizedTiles(((10, 1, 30), (2, 4, 7, 13))),
        Tiles((104, 201), (11, 23)),
    ],
)
def test_locate(tiles: RoiTiles):
    NY, NX = tiles.base.yx
    ny, nx = tiles.shape.yx

    # check all four corners
    assert tiles.locate((0, 0)) == (0, 0)
    assert tiles.locate((NY - 1, NX - 1)) == (ny - 1, nx - 1)
    assert tiles.locate((0, NX - 1)) == (0, nx - 1)
    assert tiles.locate((NY - 1, 0)) == (ny - 1, 0)

    for idx in [(-1, 0), (NY, 0), (NY, NX), (NY * 1000, 1)]:
        with pytest.raises(IndexError):
            _ = tiles.locate(idx)
