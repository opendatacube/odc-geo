from typing import Optional, Tuple

import pytest

from odc.geo._cog import _compute_cog_spec, _num_overviews
from odc.geo.gridspec import GridSpec
from odc.geo.xr import xr_zeros


def test_write_cog():
    gs = GridSpec.web_tiles(2)
    gbox = gs[2, 1]

    img = xr_zeros(gbox, dtype="uint16")
    assert img.odc.geobox == gbox

    img_bytes = img.odc.to_cog(blocksize=32)
    assert isinstance(img_bytes, bytes)

    img_bytes2 = img.odc.write_cog(":mem:", blocksize=32)
    assert len(img_bytes) == len(img_bytes2)

    img_bytes = img.odc.to_cog(blocksize=32, overview_levels=[2])
    assert isinstance(img_bytes, bytes)

    img_bytes2 = img.odc.write_cog(
        ":mem:", blocksize=32, overview_levels=[2], use_windowed_writes=True
    )
    assert len(img_bytes) == len(img_bytes2)


def test_write_cog_ovr():
    gs = GridSpec.web_tiles(2)
    gbox = gs[2, 1]

    img = xr_zeros(gbox, dtype="uint16")
    assert img.odc.geobox == gbox
    ovrs = [img[::2, ::2], img[::4, ::4]]

    img_bytes = img.odc.to_cog(blocksize=32, overviews=ovrs)
    assert isinstance(img_bytes, bytes)

    img_bytes2 = img.odc.write_cog(":mem:", blocksize=32, overviews=ovrs)
    assert len(img_bytes) == len(img_bytes2)


@pytest.mark.parametrize(
    ["block", "dim", "n_expect"],
    [
        (2**5, 2**10, 5),
        (2**5, 2**10 - 1, 5),
        (2**5, 2**10 + 1, 5),
        (256, 78, 0),
        (1024, 1_000_000, -1),
        (512, 3_040_000, -1),
    ],
)
def test_num_overviews(block: int, dim: int, n_expect: int):
    if n_expect >= 0:
        assert _num_overviews(block, dim) == n_expect
    else:
        n = _num_overviews(block, dim)
        assert dim // (2**n) <= block


@pytest.mark.parametrize(
    ("shape", "tshape", "max_pad"),
    [
        [(1024, 2048), (256, 128), None],
    ],
)
def test_cog_spec(
    shape: Tuple[int, int],
    tshape: Tuple[int, int],
    max_pad: Optional[int],
):
    _shape, _tshape, nlevels = _compute_cog_spec(shape, tshape, max_pad=max_pad)
    assert _shape[0] >= shape[0]
    assert _shape[1] >= shape[1]
    assert _tshape[0] >= tshape[0]
    assert _tshape[1] >= tshape[1]
    assert _tshape[0] % 16 == 0
    assert _tshape[1] % 16 == 0

    assert max(_shape) // (2**nlevels) <= max(tshape)

    if max_pad is not None:
        assert _shape[0] - shape[0] <= max_pad
        assert _shape[1] - shape[1] <= max_pad
