import math
from threading import local
from warnings import catch_warnings, filterwarnings

import numpy as np

from .geobox import GeoBox
from .roi import roi_normalise
from .xr import wrap_xr

# pylint: disable=import-outside-toplevel,import-error

_LCL = local()


def _rio_gbox(fname):
    import rasterio

    with rasterio.open(fname) as rio:
        return GeoBox.from_rio(rio)


def _j2k_open(path, num_threads=None):
    import glymur

    if num_threads is not None:
        try:
            if num_threads != glymur.get_option("lib.num_threads"):
                glymur.set_option("lib.num_threads", num_threads)
        except RuntimeError:
            pass

    cache = getattr(_LCL, "cache", None)
    if cache is None:
        _LCL.cache = {}
        cache = _LCL.cache
    f_cached = cache.get(path, None)
    if f_cached is not None:
        assert isinstance(f_cached, glymur.Jp2k)
        return f_cached

    with catch_warnings():
        filterwarnings("ignore", category=UserWarning)
        jp2k = glymur.Jp2k(path)
        cache[path] = jp2k
        return jp2k


def _j2k_loader(path, roi, rlevel, num_threads):
    j2k = _j2k_open(path, num_threads=num_threads)

    if roi is None:
        area = None
    else:
        yx_roi = roi_normalise(roi[:2], j2k.shape[:2])
        (y0, y1), (x0, x1) = ((r.start, r.stop) for r in yx_roi)

        if rlevel > 0:
            s = 1 << rlevel
            y0, x0, y1, x1 = (y0 * s, x0 * s, y1 * s, x1 * s)

        H, W = j2k.shape[:2]
        area = (y0, x0, min(y1, H), min(x1, W))

    return j2k.read(rlevel=rlevel, area=area)


def _toslices(chunks):
    off = 0
    for ch in chunks:
        yield slice(off, off + ch)
        off += ch


def _j2k_dask(fname, rlevel, chunks, j2k, num_threads):
    import dask.array as da
    from dask.base import quote, tokenize

    if rlevel > 0:
        shape = (
            tuple(int(math.ceil(n / (1 << rlevel))) for n in j2k.shape[:2])
            + j2k.shape[2:]
        )
    else:
        shape = j2k.shape

    band = f"j2k-{tokenize(fname, rlevel, chunks)}"
    if j2k.ndim == 3:
        if len(chunks) == 2:
            chunks = (*chunks, -1)

    chunks = da.core.normalize_chunks(chunks, shape, dtype=j2k.dtype)

    rois = tuple(tuple(_toslices(ch)) for ch in chunks)
    chunked = tuple(map(len, chunks))
    dsk = {}

    for idx in np.ndindex(chunked):
        _roi = tuple(roi[i] for roi, i in zip(rois, idx))
        dsk[(band, *idx)] = (_j2k_loader, fname, quote(_roi), rlevel, num_threads)

    return da.Array(dsk, band, chunks, dtype=j2k.dtype, shape=shape)


def j2k_load(fname, rlevel=0, *, chunks=None, roi=None, num_threads=None):
    j2k = _j2k_open(fname)
    gbox = _rio_gbox(fname)

    if chunks is None:
        data = _j2k_loader(fname, roi, rlevel=rlevel, num_threads=num_threads)
    else:
        if roi is not None:
            raise NotImplementedError("Do not support dask load with crop")

        data = _j2k_dask(fname, rlevel, chunks, j2k, num_threads)

    gbox = gbox.zoom_to(shape=data.shape[:2])

    return wrap_xr(data, gbox)
