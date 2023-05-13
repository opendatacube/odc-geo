from typing import Tuple

import numpy as np
import pytest

from odc.geo._blocks import BlockAssembler
from odc.geo.roi import RoiTiles, roi_normalise, roi_shape, roi_tiles


@pytest.mark.parametrize(
    "tiles, idx",
    [
        (
            roi_tiles((101, 203), (11, 13)),
            [(0, 0), (1, 1), (2, 3)],
        )
    ],
)
@pytest.mark.parametrize("dtype", ["int8", "uint16", "float32"])
def test_block_assembler(tiles: RoiTiles, idx, dtype):
    dtype = np.dtype(dtype)
    Z = np.zeros(tiles.base.shape, dtype)
    for v, _i in enumerate(idx, start=1):
        Z[tiles[_i]] = v
    Zf = Z.astype(np.float32).copy()
    Zf[Z == 0] = np.nan

    blocks = {i: Z[tiles[i]].copy() for i in idx}
    ba = BlockAssembler(blocks, tiles.chunks)
    assert ba.shape == Z.shape
    assert ba.ndim == Z.ndim
    assert ba.dtype == dtype

    np.testing.assert_equal(Z, ba.extract(0))

    # for ints default fill value is 0
    if np.issubdtype(dtype, np.integer):
        np.testing.assert_equal(Z, ba.extract())
        np.testing.assert_equal(Z, ba[:, :])
        np.testing.assert_equal(Z[:10, :3], ba[:10, :3])
        np.testing.assert_equal(Z[1:-1, 2:-3], ba[1:-1, 2:-3])
        np.testing.assert_array_equal(ba.extract(dtype="float32"), Zf)

    # for floats default fill value is nan
    if np.issubdtype(dtype, np.floating):
        np.testing.assert_array_equal(Zf, ba.extract())
        np.testing.assert_array_equal(Zf, ba[:, :])
        np.testing.assert_equal(Zf[:10, 3:], ba[:10, 3:])

    # test empty
    ba = BlockAssembler({}, tiles.chunks)
    assert ba.shape == tiles.base.shape
    assert ba.ndim == 2
    assert ba.dtype == "float32"

    # axis in the wrong position
    with pytest.raises(ValueError):
        _ = BlockAssembler(blocks, chunks=tiles.chunks, axis=1)

    ii = idx[-1]

    # ndim mismatch
    bb = dict(blocks.items())
    bb[ii] = bb[ii][..., np.newaxis]
    with pytest.raises(ValueError):
        _ = BlockAssembler(bb, chunks=tiles.chunks)

    # chunk size mismatch
    bb = dict(blocks.items())
    bb[ii] = bb[ii][:-2]
    with pytest.raises(ValueError):
        _ = BlockAssembler(bb, chunks=tiles.chunks)


@pytest.mark.parametrize(
    "bshape, axis",
    [
        ((10, 13), 0),
        ((1, 10, 13), 1),
        ((10, 13, 3), 0),
    ],
)
@pytest.mark.parametrize(
    "yx_crop",
    [
        np.s_[3:4, 5:9],
        np.s_[:4, -3:],
    ],
)
def test_block_planes(bshape: Tuple[int, ...], axis: int, yx_crop):
    ny, nx = bshape[axis : axis + 2]
    NY = ny * 10 + 1
    NX = nx * 9 + min(nx - 1, 2)
    rtiles = roi_tiles((NY, NX), (ny, nx))
    chunks = rtiles.chunks
    ba = BlockAssembler({(0, 0): np.zeros(bshape)}, chunks, axis=axis)
    assert ba.ndim == len(bshape)

    for _roi in ba.planes_yx():
        assert len(_roi) == len(bshape)
        assert ba[_roi].shape == (NY, NX)

        if axis == 1 and len(bshape) == 3:
            assert isinstance(_roi[0], int)
            np.testing.assert_array_equal(ba[_roi[0]], ba[_roi])
        if axis == 0 and len(bshape) == 3:
            assert isinstance(_roi[2], int)
            np.testing.assert_array_equal(ba[:, :, _roi[2]], ba[_roi])

    crop_shape = roi_shape(roi_normalise(yx_crop, (NY, NX)))
    for _roi in ba.planes_yx(yx_crop):
        assert len(_roi) == len(bshape)
        assert ba[_roi].shape == crop_shape
        if axis == 1:
            assert isinstance(_roi[0], int)
            if len(bshape) == 3:
                np.testing.assert_array_equal(ba[_roi[0]][yx_crop], ba[_roi])

    with pytest.raises(IndexError):
        _ = ba[tuple(slice(None) for _ in range(len(bshape) + 1))]
