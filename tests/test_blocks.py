from typing import Tuple

import numpy as np
import pytest

from odc.geo._blocks import BlockAssembler
from odc.geo.roi import RoiTiles, roi_tiles


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

    # for floats default fill value is nan
    if np.issubdtype(dtype, np.floating):
        Zf = Z.copy()
        Zf[Z == 0] = np.nan
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
def test_block_planes(bshape: Tuple[int, ...], axis: int):
    ny, nx = bshape[axis : axis + 2]
    rtiles = roi_tiles((ny * 10 + 1, nx * 10 + min(nx - 1, 2)), (ny, nx))
    chunks = rtiles.chunks
    ba = BlockAssembler({(0, 0): np.zeros(bshape)}, chunks, axis=axis)
    assert ba.ndim == len(bshape)

    for _roi in ba.planes_yx():
        assert len(_roi) == len(bshape)
