import math

import dask.array as da
import glymur
import numpy as np
import rasterio
from dask.base import quote, tokenize

from .geobox import GeoBox
from .xr import wrap_xr


def _rio_gbox(fname):
    with rasterio.open(fname) as rio:
        return GeoBox.from_rio(rio)


def _j2k_loader(path, roi, rlevel=0):
    # TODO: cache opened file per thread
    j2k = glymur.Jp2k(path)
    (y0, y1), (x0, x1) = ((r.start, r.stop) for r in roi[:2])

    if rlevel > 0:
        s = 1 << rlevel
        y0, x0, y1, x1 = (y0 * s, x0 * s, y1 * s, x1 * s)

    H, W = j2k.shape[:2]
    area = (y0, x0, min(y1, H), min(x1, W))
    return j2k.read(rlevel=rlevel, area=area)


def j2k_load(fname, chunks=(2048, -1), rlevel=0):
    def _toslices(chunks):
        off = 0
        for ch in chunks:
            yield slice(off, off + ch)
            off += ch

    gbox = _rio_gbox(fname)

    tk = tokenize(fname)

    j2k = glymur.Jp2k(fname)
    band = f"j2k-{tk}"
    if j2k.ndim == 3:
        if len(chunks) == 2:
            chunks = (*chunks, -1)

    if rlevel > 0:
        s = 1 << rlevel
        shape = tuple(int(math.ceil(n / s)) for n in j2k.shape[:2]) + j2k.shape[2:]
        gbox = gbox.zoom_to(shape=shape[:2])
    else:
        shape = j2k.shape

    chunks = da.core.normalize_chunks(chunks, shape, dtype=j2k.dtype)

    rois = tuple(tuple(_toslices(ch)) for ch in chunks)
    chunked = tuple(map(len, chunks))
    dsk = {}

    for idx in np.ndindex(chunked):
        _roi = tuple(roi[i] for roi, i in zip(rois, idx))
        dsk[(band, *idx)] = (_j2k_loader, fname, quote(_roi), rlevel)

    return wrap_xr(da.Array(dsk, band, chunks, dtype=j2k.dtype, shape=shape), gbox)
