# This file is part of the Open Data Cube, see https://opendatacube.org for more information
#
# Copyright (c) 2015-2020 ODC Contributors
# SPDX-License-Identifier: Apache-2.0
"""
Write Cloud Optimized GeoTIFFs from xarrays.
"""
from __future__ import annotations

import itertools
from functools import partial
from io import BytesIO
from typing import TYPE_CHECKING, Any, Callable, Optional, Union
from urllib.parse import urlparse
from xml.sax.saxutils import escape as xml_escape

import math
import numpy as np
import xarray as xr


from .._interop import have
from ..geobox import GeoBox
from ..math import resolve_nodata
from ..types import Shape2d, SomeNodata, Unset, shape_
from ._mpu import mpu_write
from ._mpu_fs import MPUFileSink
from ._multipart import MultiPartUploadBase

from ._shared import (
    GDAL_COMP,
    GEOTIFF_TAGS,
    CogMeta,
    compute_cog_spec,
    norm_blocksize,
    yaxis_from_shape,
)

if TYPE_CHECKING:
    import dask.array
    import dask.bag
    from dask.delayed import Delayed

# pylint: disable=too-many-locals,too-many-branches,too-many-arguments,too-many-statements,too-many-instance-attributes


def _render_gdal_metadata(
    band_stats: list[dict[str, float]] | dict[str, float] | None,
    precision: int = 10,
    pad: int = 0,
    eol: str = "",
    gdal_metadata_extra: Optional[list[str]] = None,
) -> str:
    def _item(sample: int, stats: dict[str, float]) -> str:
        return eol.join(
            [
                f'<Item name="STATISTICS_{k.upper()}" sample="{sample:d}">{v:{pad}.{precision}f}</Item>'
                for k, v in stats.items()
            ]
        )

    if band_stats is None:
        band_stats = []
    if isinstance(band_stats, dict):
        band_stats = [band_stats]

    gdal_metadata_extra = [] if gdal_metadata_extra is None else gdal_metadata_extra

    body = eol.join(
        [_item(sample, stats) for sample, stats in enumerate(band_stats)]
        + gdal_metadata_extra
    )
    return eol.join(["<GDALMetadata>", body, "</GDALMetadata>"])


def _unwrap_stats(stats, ndim):
    if ndim == 2:
        return [{k: float(v) for k, v in stats.items()}]

    n = {len(v) for v in stats.values()}.pop()
    return [{k: v[idx] for k, v in stats.items()} for idx in range(n)]


def _stats_from_layer(
    pix: "dask.array.Array", nodata=None, yaxis: int = 0
) -> "Delayed":
    # pylint: disable=import-outside-toplevel
    from dask import array as da
    from dask import delayed

    unwrap = delayed(_unwrap_stats, pure=True, traverse=True)

    axis = (yaxis, yaxis + 1)
    npix = pix.shape[yaxis] * pix.shape[yaxis + 1]

    if nodata is None or np.isnan(nodata):
        dd = pix
        return unwrap(
            {
                "minimum": da.nanmin(dd, axis=axis),
                "maximum": da.nanmax(dd, axis=axis),
                "mean": da.nanmean(dd, axis=axis),
                "stddev": da.nanstd(dd, axis=axis),
                "valid_percent": da.isfinite(dd).sum(axis=axis) * (100 / npix),
            },
            pix.ndim,
        )

    # Exclude both nodata and invalid (e.g. NaN) values from statistics computation
    dd = da.ma.masked_where((pix == nodata) | ~(np.isfinite(pix)), pix)
    return unwrap(
        {
            "minimum": dd.min(axis=axis),
            "maximum": dd.max(axis=axis),
            "mean": dd.mean(axis=axis),
            "stddev": dd.std(axis=axis),
            "valid_percent": da.isfinite(dd).sum(axis=axis) * (100 / npix),
        },
        pix.ndim,
    )


def _make_empty_cog(
    shape: tuple[int, ...],
    dtype: Any,
    gbox: Optional[GeoBox] = None,
    *,
    nodata: SomeNodata = "auto",
    gdal_metadata: Optional[str] = None,
    compression: Union[str, Unset] = Unset(),
    compressionargs: Any = None,
    predictor: Union[int, bool, Unset] = Unset(),
    blocksize: Union[int, list[Union[int, tuple[int, int]]]] = 2048,
    bigtiff: bool = True,
    **kw,
) -> tuple[CogMeta, memoryview]:
    # pylint: disable=import-outside-toplevel,import-error
    have.check_or_error("tifffile", "rasterio", "xarray")
    from tifffile import (
        COMPRESSION,
        FILETYPE,
        PHOTOMETRIC,
        PLANARCONFIG,
        TiffWriter,
        enumarg,
    )

    nodata = resolve_nodata(nodata, dtype)

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
    planarconfig: Optional[PLANARCONFIG] = PLANARCONFIG.SEPARATE
    if ax == "YX":
        nsamples = 1
    elif ax == "YXS":
        nsamples = shape[-1]
        planarconfig = PLANARCONFIG.CONTIG
        if nsamples in (3, 4):
            photometric = PHOTOMETRIC.RGB
    else:
        nsamples = shape[0]
        if nsamples == 1:
            planarconfig = None

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

    def _sh(shape: Shape2d) -> tuple[int, ...]:
        if ax == "YX":
            return shape.shape
        if ax == "YXS":
            return (*shape.shape, nsamples)
        return (nsamples, *shape.shape)

    tsz = norm_blocksize(blocksize[-1])
    im_shape, _, nlevels = compute_cog_spec(im_shape, tsz)

    extratags: list[tuple[int, int, int, Any]] = []
    if gbox is not None:
        gbox = gbox.expand(im_shape)
        extratags, _ = geotiff_metadata(
            gbox, nodata=nodata, gdal_metadata=gdal_metadata
        )
    # TODO: support nodata/gdal_metadata without gbox?

    _blocks = itertools.chain(iter(blocksize), itertools.repeat(blocksize[-1]))

    tw = TiffWriter(buf, bigtiff=bigtiff, shaped=False)
    metas: list[CogMeta] = []

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
            nodata=nodata,
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


def _cog_block_compressor_yxs(
    block: np.ndarray,
    *,
    tile_shape: tuple[int, ...] = (),
    encoder: Any = None,
    predictor: Any = None,
    fill_value: Union[float, int] = 0,
    **kw,
) -> bytes:
    assert block.ndim == len(tile_shape)
    if tile_shape != block.shape:
        pad = tuple((0, want - have) for want, have in zip(tile_shape, block.shape))
        block = np.pad(block, pad, "constant", constant_values=(fill_value,))

    if predictor is not None:
        block = predictor(block, axis=1)
    if encoder:
        try:
            return encoder(block, **kw)
        except Exception:  # pylint: disable=broad-except
            return b""

    return bytes(block.data)


def _cog_block_compressor_syx(
    block: np.ndarray,
    *,
    tile_shape: tuple[int, int] = (0, 0),
    encoder: Any = None,
    predictor: Any = None,
    fill_value: Union[float, int] = 0,
    sample_idx: int = 0,
    **kw,
) -> bytes:
    assert isinstance(block, np.ndarray)

    if block.ndim == 2:
        pass
    elif block.shape[0] == 1:
        block = block[0, :, :]
    else:
        block = block[sample_idx, :, :]

    assert block.ndim == 2
    if tile_shape != block.shape:
        pad = tuple((0, want - have) for want, have in zip(tile_shape, block.shape))
        block = np.pad(block, pad, "constant", constant_values=(fill_value,))

    if predictor is not None:
        block = predictor(block, axis=1)

    if encoder:
        try:
            return encoder(block, **kw)
        except Exception:  # pylint: disable=broad-except
            return b""

    return bytes(block.data)


def _mk_tile_compressor(
    meta: CogMeta, sample_idx: int = 0
) -> Callable[[np.ndarray], bytes]:
    # pylint: disable=import-outside-toplevel,import-error
    have.check_or_error("tifffile")
    from tifffile import TIFF

    tile_shape = meta.chunks
    encoder = TIFF.COMPRESSORS[meta.compression]

    predictor = None
    if meta.predictor != 1:
        predictor = TIFF.PREDICTORS[meta.predictor]

    fill_value: Union[float, int] = 0
    if meta.nodata is not None:
        fill_value = float(meta.nodata) if isinstance(meta.nodata, str) else meta.nodata

    if meta.axis == "SYX":
        return partial(
            _cog_block_compressor_syx,
            tile_shape=meta.tile.yx,
            encoder=encoder,
            predictor=predictor,
            fill_value=fill_value,
            sample_idx=sample_idx,
            **meta.compressionargs,
        )

    return partial(
        _cog_block_compressor_yxs,
        tile_shape=tile_shape,
        encoder=encoder,
        predictor=predictor,
        fill_value=fill_value,
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

    data = xx.data
    assert is_dask_collection(data)

    if meta.axis == "SYX":
        src_ydim = 1
        if data.ndim == 2:
            _chunks: tuple[int, ...] = meta.tile.yx
        elif len(data.chunks[0]) == 1:
            # if 1 single chunk with all "samples", keep it that way
            _chunks = (data.shape[0], *meta.tile.yx)
        else:
            # else have 1 chunk per "sample"
            _chunks = (1, *meta.tile.yx)

        if data.chunksize != _chunks:
            data = data.rechunk(_chunks)
    else:
        assert meta.num_planes == 1
        src_ydim = 0
        if data.chunksize != meta.chunks:
            data = data.rechunk(meta.chunks)

    encoder = _mk_tile_compressor(meta, sample_idx)

    tk = tokenize(
        data,
        scale_idx,
        meta.axis,
        meta.chunks,
        meta.predictor,
        meta.compression,
        meta.compressionargs,
    )
    cc_id = "" if scale_idx == 0 else f"_{scale_idx}"
    cc_id += "" if meta.num_planes == 1 else f"@{sample_idx}"

    name = f"compress{cc_id}-{tk}"

    src_data_name = data.name

    def block_name(s, y, x):
        if data.ndim == 2:
            return (src_data_name, y, x)
        if src_ydim == 0:
            return (src_data_name, y, x, s)
        if len(data.chunks[0]) == 1:
            return (src_data_name, 0, y, x)
        return (src_data_name, s, y, x)

    dsk: Any = {}
    for i, (s, y, x) in enumerate(meta.tidx(sample_idx)):
        block = block_name(s, y, x)
        dsk[name, i] = (_compress_cog_tile, encoder, block, quote((scale_idx, s, y, x)))

    nparts = len(dsk)
    dsk = HighLevelGraph.from_collections(name, dsk, dependencies=[data])
    return Bag(dsk, name, nparts)


def _pyramids_from_cog_metadata(
    xx: xr.DataArray,
    cog_meta: CogMeta,
    resampling: Union[str, int] = "nearest",
) -> tuple[xr.DataArray, ...]:
    out = [xx]

    for mm in cog_meta.overviews:
        gbox = mm.gbox
        out.append(
            out[-1].odc.reproject(gbox, chunks=mm.tile.yx, resampling=resampling)
        )

    return tuple(out)


def _extract_tile_info(
    meta: CogMeta,
    tiles: list[tuple[int, int, int, int, int]],
    start_offset: int = 0,
) -> list[tuple[list[int], list[int]]]:
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


def _patch_hdr(
    tiles: list[tuple[int, tuple[int, int, int, int]]],
    meta: CogMeta,
    hdr0: bytes,
    stats: Optional[list[dict[str, float]]] = None,
    gdal_metadata_extra: Optional[list[str]] = None,
) -> bytes:
    # pylint: disable=import-outside-toplevel,import-error
    from tifffile import TiffFile, TiffPage

    _tiles = [(*idx, sz) for sz, idx in tiles]
    tile_info = _extract_tile_info(meta, _tiles, 0)

    _bio = BytesIO(hdr0)
    with TiffFile(_bio, mode="r+", name=":mem:") as tr:
        assert len(tile_info) == len(tr.pages)
        if stats is not None or gdal_metadata_extra:
            md_tag = tr.pages.first.tags.get(42112, None)
            assert md_tag is not None
            gdal_metadata = _render_gdal_metadata(
                stats, precision=6, gdal_metadata_extra=gdal_metadata_extra
            )
            md_tag.overwrite(gdal_metadata)

        hdr_sz = len(_bio.getbuffer())

        # 324 -- offsets
        # 325 -- byte counts
        for info, page in zip(tile_info, tr.pages):
            assert isinstance(page, TiffPage)
            tags = page.tags
            offsets, lengths = info
            tags[324].overwrite([off + hdr_sz for off in offsets])
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
    kw: Optional[dict[str, Any]] = None,
) -> tuple[int, str, dict[str, Any]]:
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


def _gdal_sample_description(sample: int, description: str) -> str:
    """Make XML line of GDAL metadata.

    :param band: Sample / band number in data array.
    :param description: Band name in data array.

    :return: GDAL XML metadata line to place in TIFF file.
    """
    # GDAL does double escaping; see frmts/gtiff/geotiff.cpp.
    # We also double escape to maximize compatibility with tools expecting GDAL-generated metadata.
    double_escaped_description = xml_escape(xml_escape(description))
    return f'<Item name="DESCRIPTION" sample="{sample}" role="description">{double_escaped_description}</Item>'


def _band_names(xx: xr.DataArray) -> list[str]:
    if "band" in xx.coords and xx.coords["band"].dtype.type is np.str_:
        return list(xx["band"].values)
    if "long_name" in xx.attrs:
        long_name = xx.attrs["long_name"]
        return [long_name] if isinstance(long_name, str) else long_name
    return []


def _gdal_sample_descriptions(descriptions: list[str]) -> list[str]:
    """Convert band names to GDAL sample descriptions.

    :return: List of GDAL XML metadata lines to place in TIFF file.
    """
    return [
        _gdal_sample_description(sample, description)
        for sample, description in enumerate(descriptions)
    ]


def save_cog_with_dask(
    xx: xr.DataArray,
    dst: str = "",
    *,
    compression: Union[str, Unset] = Unset(),
    compressionargs: Any = None,
    level: Optional[Union[int, float]] = None,
    predictor: Union[int, bool, Unset] = Unset(),
    blocksize: Union[Unset, int, list[Union[int, tuple[int, int]]]] = Unset(),
    bigtiff: bool = True,
    overview_resampling: Union[int, str] = "nearest",
    tile_batching: int | Callable[[int], int] = 4,
    tile_batching_threshold: int = 20,
    merge_compressed_overviews: Optional[int] = 4,
    aws: Optional[dict[str, Any]] = None,
    azure: Optional[dict[str, Any]] = None,
    client: Any = None,
    stats: bool | int = True,
    **kw,
) -> Any:
    """
    Save a Cloud Optimized GeoTIFF to S3, Azure Blob Storage, or file with Dask.

    To optimise performance, pre-chunk your data to the desired COG block sizes.
    Then configure batching and overview merging as needed:

    - `tile_batching` is how many tiles get grouped into a single
      compression task. If it's an integer, that fixed number of tiles are compressed
      together. If it's a callable, it is given the total tile count and returns a
      number of tiles to compress at once. Combining small tile tasks can reduce
      overhead.

    - `tile_batching_threshold` is the minimum number of tiles in a given layer before
      any tile batching occurs. If a layer has fewer tiles than this threshold,
      batching is skipped because overhead might outweigh the benefits.

    - `merge_compressed_overviews` merges n topmost overview layers *after* they have
      been compressed, reducing overhead from having multiple small overview layers.
      If you only have a few overviews, you can disable or lower this. By default,
      it's set to merge up to four of the highest overview layers.

    :param xx: Pixels as :py:class:`xarray.DataArray` backed by Dask.
    :param dst: S3, Azure URL, or file path.
    :param compression: Compression to use, default is ``DEFLATE``.
    :param level: Compression “level”, depends on chosen compression.
    :param predictor: TIFF predictor setting.
    :param compressionargs: Any other compression arguments.
    :param overview_resampling: Resampling method used for computing overviews.
    :param tile_batching: How many tiles to group into each compression task. Can be
                          an integer or a function of total tile count. Default is 4.
    :param tile_batching_threshold: Minimum number of tiles in a layer before batching
                                    is applied. Default is 20.
    :param merge_compressed_overviews: Merges that many highest-level overview layers
                                       after compression into a single chunk. Default is 4.
                                       Set to None or 0 to disable merging.
    :param blocksize: Configure block sizes for main and overview images.
    :param bigtiff: Generate BigTIFF by default, set to ``False`` to disable.
    :param aws: Configure AWS write access.
    :param azure: Azure credentials/config.
    :param client: Dask client.
    :param stats: Set to `False` to disable stats computation. If `True`, uses a
                  mid-level layer to compute stats.

    :returns: Dask delayed.
    """
    # pylint: disable=import-outside-toplevel
    import dask.bag

    from ..xr import ODCExtensionDa

    aws = aws or {}
    azure = azure or {}

    upload_params = {
        k: kw.pop(k, None) for k in ["writes_per_chunk", "spill_sz"] if k in kw
    }
    parts_base = kw.pop("parts_base", None)

    # Normalize compression settings and remove GDAL compat options from kw
    predictor, compression, compressionargs = _norm_compression_tifffile(
        xx.dtype, predictor, compression, compressionargs, level=level, kw=kw
    )

    xx_odc = xx.odc
    assert isinstance(xx_odc, ODCExtensionDa)
    assert isinstance(xx_odc.geobox, GeoBox) or xx_odc.geobox is None

    ydim = xx_odc.ydim
    data_chunks: tuple[int, int] = xx.data.chunksize[ydim : ydim + 2]
    if isinstance(blocksize, Unset):
        # As a default guess, keep the existing chunk size, and for overviews
        # use half that dimension as a block. (Placeholder logic)
        blocksize = [data_chunks, int(max(*data_chunks) // 2)]

    # Metadata
    band_names = _band_names(xx)
    sample_descriptions_metadata = _gdal_sample_descriptions(band_names)
    gdal_metadata = None if stats is False and not band_names else ""

    # Prepare COG metadata and header
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
        gdal_metadata=gdal_metadata,
        **kw,
    )
    hdr0 = bytes(hdr0)

    if band_names and len(band_names) != meta.nsamples:
        raise ValueError(
            f"Found {len(band_names)} band names ({band_names}), expected {meta.nsamples} bands."
        )

    layers = _pyramids_from_cog_metadata(xx, meta, resampling=overview_resampling)

    # Default stats: pick a mid-level overview
    if stats is True:
        stats = len(layers) // 2

    _stats: "Delayed" | None = None
    if stats is not False:
        _stats = _stats_from_layer(
            layers[stats].data, nodata=xx_odc.nodata, yaxis=xx_odc.ydim
        )

    # Prepare tiles for each pyramid level and band
    _tiles: list["dask.bag.Bag"] = []
    for scale_idx, (mm, img) in enumerate(zip(meta.flatten(), layers)):
        for sample_idx in range(meta.num_planes):
            tt = _compress_tiles(img, mm, scale_idx=scale_idx, sample_idx=sample_idx)
            # Apply tile batching if the total tile count >= threshold
            num_tiles = tt.npartitions
            if num_tiles >= tile_batching_threshold:
                if callable(tile_batching):
                    batch_size = tile_batching(num_tiles)
                else:
                    batch_size = tile_batching
                if batch_size > 1:
                    new_parts = math.ceil(num_tiles / batch_size)
                    tt = tt.repartition(npartitions=new_parts)
            _tiles.append(tt)

    if dst == "":
        return {
            "meta": meta,
            "hdr0": hdr0,
            "tiles": _tiles,
            "layers": layers,
            "_stats": _stats,
        }

    # Determine output type and initiate uploader
    parsed_url = urlparse(dst)
    if parsed_url.scheme == "s3":
        if have.botocore:
            from ._s3 import S3MultiPartUpload, s3_parse_url

            bucket, key = s3_parse_url(dst)
            uploader: MultiPartUploadBase = S3MultiPartUpload(bucket, key, **aws)
        else:
            raise RuntimeError("Please install `boto3` to use S3")
    elif parsed_url.scheme == "az":
        if have.azure:
            from ._az import AzMultiPartUpload

            assert azure is not None
            assert "account_url" in azure
            assert "credential" in azure

            uploader = AzMultiPartUpload(
                account_url=azure["account_url"],
                container=parsed_url.netloc,
                blob=parsed_url.path.lstrip("/"),
                credential=azure["credential"],
            )
        else:
            raise RuntimeError("Please install `azure-storage-blob` to use Azure")
    else:
        # Assume local disk
        write = MPUFileSink(dst, parts_base=parts_base)
        return mpu_write(
            _tiles[::-1],
            write,
            mk_header=_patch_hdr,
            user_kw={
                "meta": meta,
                "hdr0": hdr0,
                "stats": _stats,
                "gdal_metadata_extra": sample_descriptions_metadata,
            },
            **upload_params,
        )

    # Reverse tile order for writing: highest overview first
    tiles_write_order = _tiles[::-1]

    # Optionally merge the top N overview levels into a single sub-stream.
    # Each overview layer typically uses its own sub-stream and needs
    # a portion of the upload’s chunk budget (e.g., S3 has a 10,000-chunk limit).
    # Combining multiple small overviews into one sub-stream lowers the total
    # number of sub-streams, leaving more chunk space for other layers.
    if (
        merge_compressed_overviews
        and len(tiles_write_order) > merge_compressed_overviews
    ):
        tiles_write_order = [
            dask.bag.concat(tiles_write_order[:merge_compressed_overviews]),
            *tiles_write_order[merge_compressed_overviews:],
        ]

    # Upload tiles
    return uploader.upload(
        tiles_write_order,
        mk_header=_patch_hdr,
        user_kw={
            "meta": meta,
            "hdr0": hdr0,
            "stats": _stats,
            "gdal_metadata_extra": sample_descriptions_metadata,
        },
        client=client,
        **upload_params,
    )


def geotiff_metadata(
    geobox: GeoBox,
    nodata: SomeNodata = "auto",
    gdal_metadata: Optional[str] = None,
) -> tuple[list[tuple[int, int, int, Any]], dict[str, Any]]:
    """
    Convert GeoBox to geotiff tags and metadata for :py:mod:`tifffile`.

    .. note::

       Requires :py:mod:`rasterio`, :py:mod:`tifffile` and :py:mod:`xarray`.


    :returns:
       List of TIFF tag tuples suitable for passing to :py:mod:`tifffile` as
       ``extratags=``, and dictionary representation of GEOTIFF tags.

    """
    # pylint: disable=import-outside-toplevel,import-error

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

    def _dtype_as_int(dtype) -> int:
        if isinstance(dtype, int):
            return dtype
        return dtype.value

    geo_tags: list[tuple[int, int, int, Any]] = [
        (t.code, _dtype_as_int(t.dtype), t.count, t.value)
        for t in tf.pages.first.tags.values()
        if t.code in GEOTIFF_TAGS
    ]

    if gdal_metadata is not None:
        geo_tags.append((42112, 2, len(gdal_metadata) + 1, gdal_metadata))

    return geo_tags, tf.geotiff_metadata
