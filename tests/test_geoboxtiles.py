import numpy as np
import pytest
from affine import Affine

from odc.geo import MaybeCRS, geom, wh_
from odc.geo.geobox import GeoBox, GeoboxTiles
from odc.geo.testutils import epsg3857


@pytest.mark.parametrize(
    "iso3, crs, resolution",
    [
        ("AUS", "epsg:4326", 0.1),
        ("AUS", "epsg:3577", 10_000),
        ("AUS", "epsg:3857", 10_000),
        ("NZL", "epsg:3857", 5_000),
    ],
)
def test_geoboxtiles_intersect(
    country: geom.Geometry, resolution: float, crs: MaybeCRS
):
    assert isinstance(country, geom.Geometry)
    assert country.crs == crs

    geobox = GeoBox.from_geopolygon(country, resolution=resolution, tight=True)
    assert geobox.crs == crs

    gbt = GeoboxTiles(geobox, (11, 13))
    assert gbt.base is geobox
    mm = gbt.grid_intersect(gbt)

    assert len(gbt.chunks) == 2
    assert (sum(gbt.chunks[0]), sum(gbt.chunks[1])) == gbt.base.shape

    for iy, ix in np.ndindex(gbt.shape.yx):
        idx = (iy, ix)
        assert idx in mm
        assert len(mm[idx]) == 1
        assert idx in mm[idx]

    gbt2 = GeoboxTiles(geobox.pad(2).to_crs(6933), (13, 11))
    mm = gbt.grid_intersect(gbt2)
    assert set(mm) == set(np.ndindex(gbt.shape.yx))

    gbt3 = GeoboxTiles(geobox.pad(2).rotate(1), (13, 11))
    mm = gbt.grid_intersect(gbt3)
    assert set(mm) == set(np.ndindex(gbt.shape.yx))


@pytest.mark.parametrize("use_chunks", [False, True])
def test_gbox_tiles(use_chunks):
    A = Affine.identity()
    H, W = (300, 200)
    h, w = (10, 20)
    gbox = GeoBox(wh_(W, H), A, epsg3857)
    tt = GeoboxTiles(gbox, (h, w))
    assert tt.shape == (300 / 10, 200 / 20)
    assert tt.base is gbox

    if use_chunks:
        tt = GeoboxTiles(gbox, tt.roi.chunks)

    # smoke test textual repr
    assert isinstance(str(tt), str)
    assert isinstance(repr(tt), str)

    # check ==
    assert tt == tt
    assert tt != "?"
    if use_chunks:
        assert tt == GeoboxTiles(tt.base, tt.chunks)
    else:
        assert tt == GeoboxTiles(tt.base, (h, w))

    assert tt[0, 0] == gbox[0:h, 0:w]
    assert tt[0, 1] == gbox[0:h, w : w + w]

    assert tt[4, 1].shape == (h, w)

    H, W = (11, 22)
    h, w = (10, 9)
    gbox = GeoBox(wh_(W, H), A, epsg3857)
    tt = GeoboxTiles(gbox, (h, w))
    assert tt.shape == (2, 3)
    assert tt[1, 2] == gbox[10:11, 18:22]

    # check .roi
    assert tt.base[tt.roi[1, 2]] == tt[1, 2]

    for idx in [tt.shape, (-33, 1)]:
        with pytest.raises(IndexError):
            _ = tt[idx]

        with pytest.raises(IndexError):
            tt.chunk_shape(idx)

    cc = np.zeros(tt.shape, dtype="int32")
    for idx in tt.tiles(gbox.extent):
        cc[idx] += 1
    np.testing.assert_array_equal(cc, np.ones(tt.shape))

    assert list(tt.tiles(gbox[:h, :w].extent)) == [(0, 0)]
    assert list(tt.tiles(gbox[:h, :w].extent.boundingbox)) == [(0, 0)]
    assert list(tt.tiles(gbox[:h, :w].extent.to_crs("epsg:4326"))) == [(0, 0)]

    (H, W) = (11, 22)
    (h, w) = (10, 20)
    tt = GeoboxTiles(GeoBox(wh_(W, H), A, epsg3857), (h, w))
    assert tt.chunk_shape((0, 0)) == (h, w)
    assert tt.chunk_shape((0, 1)) == (h, 2)
    assert tt.chunk_shape((1, 1)) == (1, 2)
    assert tt.chunk_shape((1, 0)) == (1, w)

    # check that overhang get's clamped properly
    assert tt.range_from_bbox(gbox.pad(2).boundingbox) == (
        range(0, tt.shape[0]),
        range(0, tt.shape[1]),
    )

    # bounding box in any projection should work
    assert tt.range_from_bbox(gbox.geographic_extent.boundingbox) == (
        range(0, tt.shape[0]),
        range(0, tt.shape[1]),
    )

    assert tt.range_from_bbox(tt[0, 0].extent.boundingbox) == (range(0, 1), range(0, 1))


@pytest.mark.parametrize("use_chunks", [False, True])
def test_gbox_tiles_roi(use_chunks):
    A = Affine.identity()
    H, W = (300, 200)
    h, w = (10, 20)
    gbox = GeoBox(wh_(W, H), A, epsg3857)
    tt = GeoboxTiles(gbox, (h, w))
    assert tt.shape == (H // h, W // w)
    assert tt.base is gbox

    if use_chunks:
        tt = GeoboxTiles(gbox, tt.roi.chunks)

    assert tt[:, :] == gbox
    assert tt.crop[:, :].base == gbox
    assert tt.crop[1, 2].base == tt[1, 2]

    assert tt.clip([(0, 0)]) == (tt.crop[0, 0], [(0, 0)])
    assert tt.clip([(1, 0)]) == (tt.crop[1, 0], [(0, 0)])
    assert tt.clip([(1, 2), (2, 5)]) == (
        tt.crop[1 : 2 + 1, 2 : 5 + 1],
        [(0, 0), (1, 3)],
    )

    for idx in np.ndindex(tt.shape.shape):
        bbox = tt.pix_bbox(idx)
        assert bbox.crs is None
        _idx = list(tt.tiles(bbox))
        assert len(_idx) == 1
        assert _idx[0] == idx
