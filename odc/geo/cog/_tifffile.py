# This file is part of the Open Data Cube, see https://opendatacube.org for more information
#
# Copyright (c) 2015-2020 ODC Contributors
# SPDX-License-Identifier: Apache-2.0
"""
Write Cloud Optimized GeoTIFFs from xarrays.
"""
import itertools
from functools import partial
from io import BytesIO
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import xarray as xr

from .._interop import have
from ..geobox import GeoBox
from ..types import MaybeNodata, Shape2d, Unset, shape_
from ._s3 import MultiPartUpload, s3_parse_url
from ._shared import (
    GDAL_COMP,
    GEOTIFF_TAGS,
    CogMeta,
    compute_cog_spec,
    norm_blocksize,
    yaxis_from_shape,
)

if TYPE_CHECKING:
    import dask.bag

# pylint: disable=too-many-locals,too-many-branches,too-many-arguments,too-many-statements,too-many-instance-attributes


def _make_empty_cog(
    shape: Tuple[int, ...],
    dtype: Any,
    gbox: Optional[GeoBox] = None,
    *,
    nodata: MaybeNodata = None,
    gdal_metadata: Optional[str] = None,
    compression: Union[str, Unset] = Unset(),
    compressionargs: Any = None,
    predictor: Union[int, bool, Unset] = Unset(),
    blocksize: Union[int, List[Union[int, Tuple[int, int]]]] = 2048,
    bigtiff: bool = True,
    **kw,
) -> Tuple[CogMeta, memoryview]:
    # pylint: disable=import-outside-toplevel
    have.check_or_error("tifffile", "rasterio", "xarray")
    from tifffile import (
        COMPRESSION,
        FILETYPE,
        PHOTOMETRIC,
        PLANARCONFIG,
        TiffWriter,
        enumarg,
    )

    predictor, compression, compressionargs = _norm_compression_tifffile(
        dtype,
        predictor,
        compression=compression,
        compressionargs=compressionargs,
        kw=kw,
    )
    _compression = enumarg(COMPRESSION, compression.upper())

    if isinstance(blocksize, int):
        blocksize = [blocksize]

    ax, yaxis = yaxis_from_shape(shape, gbox)
    im_shape = shape_(shape[yaxis : yaxis + 2])
    photometric = PHOTOMETRIC.MINISBLACK
    planarconfig = PLANARCONFIG.SEPARATE
    if ax == "YX":
        nsamples = 1
    elif ax == "YXS":
        nsamples = shape[-1]
        planarconfig = PLANARCONFIG.CONTIG
        if nsamples in (3, 4):
            photometric = PHOTOMETRIC.RGB
    else:
        nsamples = shape[0]

    buf = BytesIO()

    opts_common = {
        "dtype": dtype,
        "photometric": photometric,
        "planarconfig": planarconfig,
        "predictor": predictor,
        "compression": _compression,
        "compressionargs": compressionargs,
        "software": False,
        **kw,
    }

    def _sh(shape: Shape2d) -> Tuple[int, ...]:
        if ax == "YX":
            return shape.shape
        if ax == "YXS":
            return (*shape.shape, nsamples)
        return (nsamples, *shape.shape)

    tsz = norm_blocksize(blocksize[-1])
    im_shape, _, nlevels = compute_cog_spec(im_shape, tsz)

    extratags: List[Tuple[int, int, int, Any]] = []
    if gbox is not None:
        gbox = gbox.expand(im_shape)
        extratags, _ = geotiff_metadata(
            gbox, nodata=nodata, gdal_metadata=gdal_metadata
        )
    # TODO: support nodata/gdal_metadata without gbox?

    _blocks = itertools.chain(iter(blocksize), itertools.repeat(blocksize[-1]))

    tw = TiffWriter(buf, bigtiff=bigtiff, shaped=False)
    metas: List[CogMeta] = []

    for tsz, idx in zip(_blocks, range(nlevels + 1)):
        tile = norm_blocksize(tsz)
        meta = CogMeta(
            ax,
            im_shape,
            shape_(tile),
            nsamples,
            dtype,
            int(_compression),
            predictor,
            compressionargs=compressionargs,
            gbox=gbox,
        )

        if idx == 0:
            kw = {**opts_common, "extratags": extratags}
        else:
            kw = {**opts_common, "subfiletype": FILETYPE.REDUCEDIMAGE}

        tw.write(
            itertools.repeat(b""),
            shape=_sh(im_shape),
            tile=tile,
            **kw,
        )

        metas.append(meta)
        im_shape = im_shape.shrink2()
        if gbox is not None:
            gbox = gbox.zoom_to(im_shape)

    meta = metas[0]
    meta.overviews = tuple(metas[1:])

    tw.close()

    return meta, buf.getbuffer()


def _cog_block_compressor(
    block: np.ndarray,
    *,
    tile_shape: Tuple[int, ...] = (),
    encoder: Any = None,
    predictor: Any = None,
    axis: int = 1,
    **kw,
) -> bytes:
    assert block.ndim == len(tile_shape)
    if tile_shape != block.shape:
        pad = tuple((0, want - have) for want, have in zip(tile_shape, block.shape))
        block = np.pad(block, pad, "constant", constant_values=(0,))

    if predictor is not None:
        block = predictor(block, axis=axis)
    if encoder:
        try:
            return encoder(block, **kw)
        except Exception:  # pylint: disable=broad-except
            return b""

    return bytes(block.data)


def _mk_tile_compressor(meta: CogMeta) -> Callable[[np.ndarray], bytes]:
    # pylint: disable=import-outside-toplevel
    have.check_or_error("tifffile")
    from tifffile import TIFF

    tile_shape = meta.chunks
    encoder = TIFF.COMPRESSORS[meta.compression]

    # TODO: handle SYX in planar mode
    axis = 1
    predictor = None
    if meta.predictor != 1:
        predictor = TIFF.PREDICTORS[meta.predictor]

    return partial(
        _cog_block_compressor,
        tile_shape=tile_shape,
        encoder=encoder,
        predictor=predictor,
        axis=axis,
        **meta.compressionargs,
    )


def _compress_cog_tile(encoder, block, idx):
    return [(encoder(block), idx)]


def _compress_tiles(
    xx: xr.DataArray,
    meta: CogMeta,
    scale_idx: int = 0,
    sample_idx: int = 0,
) -> "dask.bag.Bag":
    """
    Compress chunks according to cog spec.

    :returns: Dask bag of tuples ``(data: bytes, idx: (int, int, int, int))}``
    """
    # pylint: disable=import-outside-toplevel
    have.check_or_error("dask")
    from dask.bag import Bag
    from dask.base import quote, tokenize
    from dask.highlevelgraph import HighLevelGraph

    from .._interop import is_dask_collection

    # TODO: deal with SYX planar data
    assert meta.num_planes == 1
    assert meta.axis in ("YX", "YXS")
    assert meta.num_planes == 1
    src_ydim = 0  # for now assume Y,X or Y,X,S

    encoder = _mk_tile_compressor(meta)
    data = xx.data

    if data.chunksize != meta.chunks:
        data = data.rechunk(meta.chunks)

    assert is_dask_collection(data)

    tk = tokenize(
        data,
        scale_idx,
        meta.axis,
        meta.chunks,
        meta.predictor,
        meta.compression,
        meta.compressionargs,
    )
    plane_id = "" if scale_idx == 0 else f"_{scale_idx}"
    plane_id += "" if sample_idx == 0 else f"@{sample_idx}"

    name = f"compress{plane_id}-{tk}"

    src_data_name = data.name

    def block_name(p, y, x):
        if data.ndim == 2:
            return (src_data_name, y, x)
        if src_ydim == 0:
            return (src_data_name, y, x, p)
        return (src_data_name, p, y, x)

    dsk = {}
    for i, (p, y, x) in enumerate(meta.tidx()):
        block = block_name(p, y, x)
        dsk[name, i] = (_compress_cog_tile, encoder, block, quote((scale_idx, p, y, x)))

    nparts = len(dsk)
    dsk = HighLevelGraph.from_collections(name, dsk, dependencies=[data])
    return Bag(dsk, name, nparts)


def _pyramids_from_cog_metadata(
    xx: xr.DataArray,
    cog_meta: CogMeta,
    resampling: Union[str, int] = "nearest",
) -> Tuple[xr.DataArray, ...]:
    out = [xx]

    for mm in cog_meta.overviews:
        gbox = mm.gbox
        out.append(
            out[-1].odc.reproject(gbox, chunks=mm.tile.yx, resampling=resampling)
        )

    return tuple(out)


def _extract_tile_info(
    meta: CogMeta,
    tiles: List[Tuple[int, int, int, int, int]],
    start_offset: int = 0,
) -> List[Tuple[List[int], List[int]]]:
    mm = meta.flatten()
    tile_info = [([0] * m.num_tiles, [0] * m.num_tiles) for m in mm]

    byte_offset = start_offset
    for scale_idx, p, y, x, sz in tiles:
        m = mm[scale_idx]
        b_offsets, b_lengths = tile_info[scale_idx]

        tidx = m.flat_tile_idx((p, y, x))
        if sz != 0:
            b_lengths[tidx] = sz
            b_offsets[tidx] = byte_offset
            byte_offset += sz

    return tile_info


def _build_hdr(
    tiles: List[Tuple[int, Tuple[int, int, int, int]]], meta: CogMeta, hdr0: bytes
) -> bytes:
    # pylint: disable=import-outside-toplevel
    from tifffile import TiffFile

    _tiles = [(*idx, sz) for sz, idx in tiles]
    tile_info = _extract_tile_info(meta, _tiles, len(hdr0))

    _bio = BytesIO(hdr0)
    with TiffFile(_bio, mode="r+", name=":mem:") as tr:
        assert len(tile_info) == len(tr.pages)

        # 324 -- offsets
        # 325 -- byte counts
        for info, page in zip(tile_info, tr.pages):
            tags = page.tags
            offsets, lengths = info
            tags[324].overwrite(offsets)
            tags[325].overwrite(lengths)

    return bytes(_bio.getbuffer())


def _norm_predictor(predictor: Union[int, bool, None], dtype: Any) -> int:
    if predictor is False or predictor is None:
        return 1

    if predictor is True:
        dtype = np.dtype(dtype)
        if dtype.kind == "f":
            return 3
        if dtype.kind in "ui" and dtype.itemsize <= 4:
            return 2
        return 1
    return predictor


def _norm_compression_tifffile(
    dtype: Any,
    predictor: Union[bool, None, int, Unset] = Unset(),
    compression: Union[str, Unset] = Unset(),
    compressionargs: Any = None,
    level: Optional[Union[int, float]] = None,
    kw: Optional[Dict[str, Any]] = None,
) -> Tuple[int, str, Dict[str, Any]]:
    if kw is None:
        kw = {}
    if isinstance(compression, Unset):
        compression = kw.pop("compress", "ADOBE_DEFLATE")
        assert isinstance(compression, str)

    if compressionargs is None:
        compressionargs = {}

    remap = {k.upper(): k for k in kw}

    def opt(name: str, default=None) -> Any:
        k = remap.get(name.upper(), None)
        if k is None:
            return default
        return kw.pop(k, default)

    def _gdal_level(compression: str, default=None) -> Any:
        gdal_level_k = GDAL_COMP.get(compression, None)
        if gdal_level_k is None:
            return default
        return opt(gdal_level_k, default)

    compression = compression.upper()

    if level is None and "level" not in compressionargs:
        # GDAL compat
        level = _gdal_level(compression)

    if level is not None:
        compressionargs["level"] = level

    if compression == "DEFLATE":
        compression = "ADOBE_DEFLATE"
    if compression == "LERC_DEFLATE":
        compression = "LERC"
        compressionargs["compression"] = "deflate"
        if (lvl := _gdal_level("DEFLATE")) is not None:
            compressionargs["compressionargs"] = {"level": lvl}
    elif compression == "LERC_ZSTD":
        compression = "LERC"
        compressionargs["compression"] = "zstd"
        if (lvl := _gdal_level("ZSTD")) is not None:
            compressionargs["compressionargs"] = {"level": lvl}

    if isinstance(predictor, Unset):
        predictor = compression in ("ADOBE_DEFLATE", "ZSTD", "LZMA")

    predictor = _norm_predictor(predictor, dtype)
    return (predictor, compression, compressionargs)


def save_cog_with_dask(
    xx: xr.DataArray,
    dst: str = "",
    *,
    compression: Union[str, Unset] = Unset(),
    compressionargs: Any = None,
    level: Optional[Union[int, float]] = None,
    predictor: Union[int, bool, Unset] = Unset(),
    blocksize: Union[Unset, int, List[Union[int, Tuple[int, int]]]] = Unset(),
    bigtiff: bool = True,
    overview_resampling: Union[int, str] = "nearest",
    aws: Optional[Dict[str, Any]] = None,
    **kw,
):
    # pylint: disable=import-outside-toplevel
    import dask.bag

    from ..xr import ODCExtensionDa

    if aws is None:
        aws = {}

    # normalize compression and remove GDAL compat options from kw
    predictor, compression, compressionargs = _norm_compression_tifffile(
        xx.dtype, predictor, compression, compressionargs, level=level, kw=kw
    )
    xx_odc = xx.odc
    assert isinstance(xx_odc, ODCExtensionDa)
    assert isinstance(xx_odc.geobox, GeoBox) or xx_odc.geobox is None

    ydim = xx_odc.ydim
    data_chunks: Tuple[int, int] = xx.data.chunksize[ydim : ydim + 2]
    if isinstance(blocksize, Unset):
        blocksize = [data_chunks, int(max(*data_chunks) // 2)]

    meta, hdr0 = _make_empty_cog(
        xx.shape,
        xx.dtype,
        xx_odc.geobox,
        predictor=predictor,
        compression=compression,
        compressionargs=compressionargs,
        blocksize=blocksize,
        bigtiff=bigtiff,
        nodata=xx_odc.nodata,
        **kw,
    )
    hdr0 = bytes(hdr0)
    mk_header = partial(_build_hdr, meta=meta, hdr0=hdr0)

    layers = _pyramids_from_cog_metadata(xx, meta, resampling=overview_resampling)

    _tiles: List["dask.bag.Bag"] = []
    for scale_idx, (mm, img) in enumerate(zip(meta.flatten(), layers)):
        # TODO: SYX order
        tt = _compress_tiles(img, mm, scale_idx=scale_idx)
        if tt.npartitions > 20:
            tt = tt.repartition(npartitions=tt.npartitions // 4)
        _tiles.append(tt)

    if dst == "":
        return {
            "meta": meta,
            "hdr0": hdr0,
            "mk_header": mk_header,
            "tiles": _tiles,
            "layers": layers,
        }

    bucket, key = s3_parse_url(dst)
    if not bucket:
        raise ValueError("Currently only support output to S3")

    upload_params = {
        k: aws.pop(k) for k in ["writes_per_chunk", "spill_sz"] if k in aws
    }
    tiles_write_order = _tiles[::-1]
    if len(tiles_write_order) > 4:
        tiles_write_order = [
            dask.bag.concat(tiles_write_order[:4]),
            *tiles_write_order[4:],
        ]

    cleanup = aws.pop("cleanup", False)
    s3_sink = MultiPartUpload(bucket, key, **aws)
    if cleanup:
        s3_sink.cancel("all")
    return s3_sink.upload(tiles_write_order, mk_header=mk_header, **upload_params)


def geotiff_metadata(
    geobox: GeoBox,
    nodata: MaybeNodata = None,
    gdal_metadata: Optional[str] = None,
) -> Tuple[List[Tuple[int, int, int, Any]], Dict[str, Any]]:
    """
    Convert GeoBox to geotiff tags and metadata for :py:mod:`tifffile`.

    .. note::

       Requires :py:mod:`rasterio`, :py:mod:`tifffile` and :py:mod:`xarray`.


    :returns:
       List of TIFF tag tuples suitable for passing to :py:mod:`tifffile` as
       ``extratags=``, and dictionary representation of GEOTIFF tags.

    """
    # pylint: disable=import-outside-toplevel

    if not (have.tifffile and have.rasterio):
        raise RuntimeError(
            "Please install `tifffile` and `rasterio` to use this method"
        )

    from tifffile import TiffFile

    from ..xr import xr_zeros
    from ._rio import to_cog

    buf = to_cog(
        xr_zeros(geobox[:2, :2]), nodata=nodata, compress=None, overview_levels=[]
    )
    tf = TiffFile(BytesIO(buf), mode="r")
    assert tf.geotiff_metadata is not None
    geo_tags: List[Tuple[int, int, int, Any]] = [
        (t.code, t.dtype.value, t.count, t.value)
        for t in tf.pages.first.tags.values()
        if t.code in GEOTIFF_TAGS
    ]

    if gdal_metadata is not None:
        geo_tags.append((42112, 2, len(gdal_metadata) + 1, gdal_metadata))

    return geo_tags, tf.geotiff_metadata
