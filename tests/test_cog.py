import itertools
from io import BytesIO
from pathlib import Path
from typing import Optional, Tuple

import pytest

from odc.geo.cog import CogMeta, cog_gbox, save_cog_with_dask
from odc.geo.cog._shared import compute_cog_spec, num_overviews
from odc.geo.cog._tifffile import (
    GEOTIFF_TAGS,
    _make_empty_cog,
    _norm_compression_tifffile,
    geotiff_metadata,
)
from odc.geo.geobox import GeoBox
from odc.geo.gridspec import GridSpec
from odc.geo.types import Unset, wh_
from odc.geo.xr import xr_zeros

gbox_globe = GridSpec.web_tiles(0)[0, 0]
_gbox = GeoBox.from_bbox((-10, -20, 15, 30), 4326, resolution=1)

# pylint: disable=redefined-outer-name


@pytest.fixture
def gbox():
    gs = GridSpec.web_tiles(2)
    return gs[2, 1]


def test_write_cog(gbox: GeoBox):
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


def test_write_cog_ovr(gbox: GeoBox):
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
        assert num_overviews(block, dim) == n_expect
    else:
        n = num_overviews(block, dim)
        assert dim // (2**n) <= block


@pytest.mark.parametrize(
    ("shape", "tshape", "max_pad"),
    [
        [(1024, 2048), (256, 128), None],
        [(1024, 2048), (256, 128), 0],
        [(1024 + 11, 2048 + 13), (256, 128), 0],
    ],
)
def test_cog_spec(
    shape: Tuple[int, int],
    tshape: Tuple[int, int],
    max_pad: Optional[int],
):
    _shape, _tshape, nlevels = compute_cog_spec(shape, tshape, max_pad=max_pad)
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
    "gbox",
    [
        gbox_globe.zoom_to(shape)
        for shape in [
            (1024, 2048),
            (17, 19),
            (10_003, 10_003),
            (1 << 20, 1 << 20),
        ]
    ],
)
@pytest.mark.parametrize("kw", [{}, {"tile": 256}, {"nlevels": 5}])
def test_cog_gbox(gbox: GeoBox, kw):
    _gbox = cog_gbox(gbox, **kw)
    assert _gbox[:1, :1] == gbox[:1, :1]
    assert _gbox.shape[0] >= gbox.shape[0]
    assert _gbox.shape[1] >= gbox.shape[1]

    # once expanded has to stay same size
    assert cog_gbox(gbox, **kw) == cog_gbox(cog_gbox(gbox, **kw), **kw)


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
        ("int16", Unset(), 2),
        ("float32", "zstd", 3),
        ("float32", "adobe_deflate", 3),
        ("float32", "lerc", 1),
        ("float32", "lerc_deflate", 1),
        ("float32", "lerc_zstd", 1),
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

    meta, mm = _make_empty_cog(
        shape,
        dtype,
        gbox=gbox,
        blocksize=blocksize,
        compression=compression,
    )
    assert isinstance(mm, memoryview)
    assert meta.axis == expect_ax
    assert meta.dtype == dtype
    assert meta.shape[0] >= gbox.shape[0]
    assert meta.shape[1] >= gbox.shape[1]
    assert meta.gbox is not None
    assert meta.shape == meta.gbox.shape

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
        if compression == "DEFLATE":
            assert p.compression.name == "ADOBE_DEFLATE"
        elif compression.startswith("LERC_"):
            assert p.compression.name == "LERC"
        else:
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


@pytest.mark.parametrize(
    "meta",
    [
        CogMeta("YX", wh_(256, 256), wh_(128, 128), 1, "int16", 8, 1),
        CogMeta("YXS", wh_(500, 256), wh_(128, 128), 3, "uint8", 8, 1),
        CogMeta("SYX", wh_(500, 256), wh_(128, 128), 5, "float32", 8, 1),
    ],
)
def test_cog_meta(meta: CogMeta):
    for idx_flat, idx in enumerate(meta.tidx()):
        assert meta.flat_tile_idx(idx) == idx_flat

    assert meta.num_tiles == (meta.chunked.x * meta.chunked.y * meta.num_planes)
    assert meta.num_tiles == len(list(meta.tidx()))

    assert len(meta.pix_shape) == {"YX": 2, "SYX": 3, "YXS": 3}[meta.axis]
    assert len(meta.pix_shape) == len(meta.chunks)

    if meta.axis in ("YX", "YXS"):
        assert meta.num_planes == 1
        assert meta.nsamples >= meta.num_planes

    if meta.axis == "SYX":
        assert meta.num_planes == meta.nsamples

    layers = meta.flatten()
    for idx in meta.cog_tidx():
        assert isinstance(idx, tuple)
        assert len(idx) == 4
        img_idx, s, y, x = idx
        tidx = (s, y, x)
        assert 0 <= img_idx < len(layers)
        assert layers[img_idx].flat_tile_idx(tidx) <= layers[img_idx].num_tiles

    for bad_idx in [
        (0, -1, 0),
        (-1, 0, 0),
        (0, 0, -1),
        (meta.num_planes, 0, 0),
        (0, meta.chunked.y, 0),
        (0, meta.chunked.y - 1, meta.chunked.x + 2),
    ]:
        with pytest.raises(IndexError):
            _ = meta.flat_tile_idx(bad_idx)


def test_norm_compress():
    predictor, compression, ca = _norm_compression_tifffile("int16")
    assert compression == "ADOBE_DEFLATE"
    assert predictor == 2
    assert ca == {}


@pytest.mark.parametrize(
    "gbox",
    [
        _gbox,
        _gbox.to_crs(3857),
        _gbox.to_crs("ESRI:53010"),
        _gbox.rotate(10),
        _gbox.center_pixel.pad(3),
    ],
)
@pytest.mark.parametrize("nodata", [None, float("nan"), 0, -999])
@pytest.mark.parametrize("gdal_metadata", [None, "<GDALMetadata></GDALMetadata>"])
def test_geotiff_metadata(gbox: GeoBox, nodata, gdal_metadata: Optional[str]):
    assert gbox.crs is not None

    geo_tags, md = geotiff_metadata(gbox, nodata=nodata, gdal_metadata=gdal_metadata)
    assert isinstance(md, dict)
    assert isinstance(geo_tags, list)
    assert len(geo_tags) >= 2
    tag_codes = {code for code, *_ in geo_tags}

    if nodata is not None:
        assert 42113 in tag_codes
    if gdal_metadata is not None:
        assert 42112 in tag_codes

    for code, dtype, count, val in geo_tags:
        assert code in GEOTIFF_TAGS
        assert isinstance(dtype, int)
        assert isinstance(count, int)
        if count > 0:
            assert isinstance(val, (tuple, str))
            if isinstance(val, str):
                assert len(val) + 1 == count
            else:
                assert len(val) == count

    if gbox.axis_aligned:
        assert "ModelPixelScale" in md
    else:
        assert "ModelTransformation" in md

    if gbox.crs.epsg is not None:
        if gbox.crs.projected:
            assert md["GTModelTypeGeoKey"] == 1
            assert md["ProjectedCSTypeGeoKey"] == gbox.crs.epsg
        else:
            assert md["GTModelTypeGeoKey"] == 2
            assert md["GeographicTypeGeoKey"] == gbox.crs.epsg
    else:
        assert md["GTModelTypeGeoKey"] == 32767


@pytest.mark.parametrize("dtype", ["int16", "float32"])
def test_cog_with_dask_smoke_test(gbox: GeoBox, tmp_path: Path, dtype):
    gbox = gbox.zoom_to(1024)
    assert gbox.shape == (1024, 1024)
    n = 512
    nodata = -9999

    fname = str(tmp_path / "cog.tif")
    img = xr_zeros(gbox, dtype, chunks=(n, n), nodata=nodata)
    fut = save_cog_with_dask(img, fname, compression="deflate", level=2)
    assert fut is not None
    rr = fut.compute()
    assert str(rr) == fname

    # YXS
    img = img.odc.colorize()
    assert img.ndim == 3
    assert img.odc.ydim == 0

    fname = str(tmp_path / "cog-rgba.tif")
    fut = save_cog_with_dask(img, fname, compression="deflate", level=2)
    rr = fut.compute()
    assert str(rr) == fname

    # SYX
    img = xr_zeros(gbox, dtype, chunks=(1, n, n), time=["2000", "2001"])
    assert img.ndim == 3
    assert img.odc.ydim == 1
    fname = str(tmp_path / "cog-syx.tif")
    fut = save_cog_with_dask(img, fname, compression="deflate", level=2)
    rr = fut.compute()
    assert str(rr) == fname
