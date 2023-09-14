import itertools
from io import BytesIO
from typing import Optional, Tuple

import pytest

from odc.geo._cog import _compute_cog_spec, _num_overviews, make_empty_cog
from odc.geo.gridspec import GridSpec
from odc.geo.types import Unset
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


@pytest.mark.parametrize(
    "shape, blocksize, expect_ax",
    [
        ((800, 600), [400, 200], "YX"),
        ((800, 600, 3), [400, 200], "YXS"),
        ((800, 600, 4), [400, 200], "YXS"),
        ((2, 800, 600), [400, 200], "SYX"),
        ((160, 30), 16, "YX"),
        ((160, 30, 5), 16, "YXS"),
    ],
)
@pytest.mark.parametrize(
    "dtype, compression, expect_predictor",
    [
        ("int16", "deflate", 2),
        ("int16", "zstd", 2),
        ("uint8", "webp", 1),
        ("float32", Unset(), 3),
    ],
)
def test_empty_cog(shape, blocksize, expect_ax, dtype, compression, expect_predictor):
    tifffile = pytest.importorskip("tifffile")
    gbox = GridSpec.web_tiles(0)[0, 0]
    if expect_ax == "SYX":
        gbox = gbox.zoom_to(shape[1:])
        assert gbox.shape == shape[1:]
    else:
        gbox = gbox.zoom_to(shape[:2])
        assert gbox.shape == shape[:2]

    mm = make_empty_cog(
        shape,
        dtype,
        gbox=gbox,
        blocksize=blocksize,
        compression=compression,
    )
    assert isinstance(mm, memoryview)

    f = tifffile.TiffFile(BytesIO(mm))
    assert f.tiff.is_bigtiff

    p = f.pages[0]
    assert p.shape[0] >= shape[0]
    assert p.shape[1] >= shape[1]
    assert p.dtype == dtype
    assert p.axes == expect_ax
    assert p.predictor == expect_predictor

    if isinstance(compression, str):
        compression = compression.upper()
        compression = {"DEFLATE": "ADOBE_DEFLATE"}.get(compression, compression)
        assert p.compression.name == compression
    else:
        # should default to deflate
        assert p.compression == 8

    if expect_ax == "YX":
        assert f.pages[-1].chunked == (1, 1)
    elif expect_ax == "YXS":
        assert f.pages[-1].chunked[:2] == (1, 1)
    elif expect_ax == "SYX":
        assert f.pages[-1].chunked[1:] == (1, 1)

    if not isinstance(blocksize, list):
        blocksize = [blocksize]

    _blocks = itertools.chain(iter(blocksize), itertools.repeat(blocksize[-1]))
    for p, tsz in zip(f.pages, _blocks):
        if isinstance(tsz, int):
            tsz = (tsz, tsz)

        assert p.chunks[0] % 16 == 0
        assert p.chunks[1] % 16 == 0

        assert tsz[0] <= p.chunks[0] < tsz[0] + 16
        assert tsz[1] <= p.chunks[1] < tsz[1] + 16
