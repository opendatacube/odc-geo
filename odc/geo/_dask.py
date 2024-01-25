from functools import partial
from typing import Any, Dict, Optional, Sequence, Tuple, Union
from uuid import uuid4

import dask.array as da
import numpy as np
from dask.highlevelgraph import HighLevelGraph

from ._blocks import BlockAssembler
from .gcp import GCPGeoBox
from .geobox import GeoBox, GeoboxTiles
from .warp import Nodata, Resampling, _rio_reproject, resampling_s2rio


def resolve_fill_value(dst_nodata, src_nodata, dtype):
    dtype = np.dtype(dtype)

    if dst_nodata is not None:
        return dtype.type(dst_nodata)
    if src_nodata is not None:
        return dtype.type(src_nodata)
    if np.issubdtype(dtype, np.floating):
        return dtype.type("nan")
    return dtype.type(0)


def _do_chunked_reproject(
    d2s: Dict[Tuple[int, int], Sequence[Tuple[int, int]]],
    src_gbt: GeoboxTiles,
    dst_gbt: GeoboxTiles,
    dst_idx: Tuple[int, int],
    *blocks: np.ndarray,
    axis: int = 0,
    dtype=None,
    casting="same_kind",
    resampling: Resampling = "nearest",
    src_nodata: Nodata = None,
    dst_nodata: Nodata = None,
    **kwargs,
):
    # pylint: disable=too-many-locals
    src_gbt, src_idx = src_gbt.clip(d2s[dst_idx])
    src_gbox = src_gbt.base
    dst_gbox = dst_gbt[dst_idx]
    assert isinstance(dst_gbox, GeoBox)
    assert isinstance(src_gbox, (GCPGeoBox, GeoBox))

    ba = BlockAssembler(dict(zip(src_idx, blocks)), src_gbt.chunks, axis=axis)
    if dtype is None:
        dtype = ba.dtype

    dst_shape = ba.with_yx(ba.shape, dst_gbox.shape)
    dst = np.zeros(dst_shape, dtype=dtype)

    for src_roi in ba.planes_yx():
        src = ba.extract(src_nodata, dtype=dtype, casting=casting, roi=src_roi)
        dst_roi = ba.with_yx(src_roi, np.s_[:, :])

        _ = _rio_reproject(
            src,
            dst[dst_roi],
            src_gbox,
            dst_gbox,
            resampling=resampling,
            src_nodata=src_nodata,
            dst_nodata=dst_nodata,
            **kwargs,
        )

    return dst


def _dask_rio_reproject(
    src: da.Array,
    s_gbox: Union[GeoBox, GCPGeoBox],
    d_gbox: GeoBox,
    resampling: Resampling,
    src_nodata: Nodata = None,
    dst_nodata: Nodata = None,
    ydim: int = 0,
    chunks: Optional[Tuple[int, int]] = None,
    **kwargs,
) -> da.Array:
    # pylint: disable=too-many-arguments, too-many-locals
    if isinstance(resampling, str):
        resampling = resampling_s2rio(resampling)

    if chunks is None:
        ny, nx = map(int, src.chunksize[ydim : ydim + 2])
        chunks = (ny, nx)

    def with_yx(a, yx):
        return (*a[:ydim], *yx, *a[ydim + 2 :])

    name: str = kwargs.pop("name", "reproject")

    assert isinstance(s_gbox, GeoBox)
    gbt_src = GeoboxTiles(s_gbox, src.chunks[ydim : ydim + 2])
    gbt_dst = GeoboxTiles(d_gbox, chunks)
    d2s_idx = gbt_dst.grid_intersect(gbt_src)

    dst_shape = with_yx(src.shape, d_gbox.shape.yx)
    dst_chunks: Tuple[Tuple[int, ...], ...] = with_yx(src.chunks, gbt_dst.chunks)

    tk = uuid4().hex
    name = f"{name}-{tk}"
    dsk: Any = {}

    proc = partial(
        _do_chunked_reproject,
        d2s_idx,
        gbt_src,
        gbt_dst,
        src_nodata=src_nodata,
        dst_nodata=dst_nodata,
        axis=ydim,
        resampling=resampling,
        **kwargs,
    )
    src_block_keys = src.__dask_keys__()

    fill_value = resolve_fill_value(dst_nodata, src_nodata, src.dtype)

    def _src(idx):
        a = src_block_keys
        for i in idx:
            a = a[i]
        return a

    shape_in_blocks = tuple(map(len, dst_chunks))
    for idx in np.ndindex(shape_in_blocks):
        y, x = idx[ydim : ydim + 2]
        srcs = [with_yx(idx, (y, x)) for y, x in d2s_idx.get((y, x), [])]

        k = (name, *idx)
        if srcs:
            block_deps = tuple(_src(s_idx) for s_idx in srcs)
            dsk[k] = (proc, (y, x), *block_deps)
        else:
            b_shape = tuple(ch[i] for ch, i in zip(dst_chunks, idx))
            dsk[k] = (np.full, b_shape, fill_value, src.dtype)

    dsk = HighLevelGraph.from_collections(name, dsk, dependencies=(src,))

    return da.Array(dsk, name, chunks=dst_chunks, dtype=src.dtype, shape=dst_shape)
