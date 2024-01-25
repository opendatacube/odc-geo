""" Helpers for dealing with RGB(A) images.
"""
import functools
from typing import Any, List, Optional, Tuple

import numpy as np
import xarray as xr

from ._interop import is_dask_collection

# pylint: disable=import-outside-toplevel


def is_rgb(x: xr.DataArray):
    """
    Check if array is RGB(A).
    """
    if x.dtype != "uint8":
        return False
    if x.ndim < 3:
        return False
    if x.shape[-1] not in (3, 4):
        return False
    return True


def _guess_rgb_names(bands: List[str]) -> Tuple[str, str, str]:
    def _candidate(color: str) -> str:
        candidates = [name for name in bands if color in name]
        n = len(candidates)
        if n == 1:
            return candidates[0]

        if n == 0:
            raise ValueError(f'Found no candidate for color "{color}"')
        raise ValueError(f'Found too many candidates for color "{color}"')

    r, g, b = [_candidate(c) for c in ("red", "green", "blue")]
    return (r, g, b)


def _auto_guess_clamp(ds: xr.Dataset) -> Tuple[float, float]:
    # TODO: deal with nodata > 0 case
    return (float(0), max(x.data.max() for x in ds.data_vars.values()))


def _to_u8(x: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    x = np.clip(x, vmin, vmax)

    if x.dtype.kind == "f":
        x = (x - vmin) * (255.0 / (vmax - vmin))
    else:
        x = (x - vmin).astype("uint32") * 255 // (vmax - vmin)
    return x.astype("uint8")


def _np_to_rgba(
    r: np.ndarray,
    g: np.ndarray,
    b: np.ndarray,
    nodata: Optional[float],
    vmin: float,
    vmax: float,
) -> np.ndarray:
    rgba = np.zeros((*r.shape, 4), dtype="uint8")

    if r.dtype.kind == "f":
        valid = ~np.isnan(r)
        if nodata is not None:
            valid = valid * (r != nodata)
    elif nodata is not None:
        valid = r != nodata
    else:
        valid = np.ones(r.shape, dtype=np.bool_)

    rgba[..., 3] = valid.astype("uint8") * (0xFF)
    for idx, band in enumerate([r, g, b]):
        rgba[..., idx] = _to_u8(band, vmin, vmax)

    return rgba


def to_rgba(
    ds: Any,
    bands: Optional[Tuple[str, str, str]] = None,
    *,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> xr.DataArray:
    """
    Convert dataset to RGBA image.

    Given :py:class:`xarray.Dataset` with bands ``red,green,blue`` construct :py:class:`xarray.Datarray`
    containing ``uint8`` rgba image.

    :param ds: :py:class:`xarray.Dataset`
    :param vmin: Defaults to ``0`` when ``vmax`` is supplied.
    :param vmax:
       Configure range, must be supplied for Dask inputs. When not configured
       ``vmin=0, vmax=max(r,g,b))`` is used.

    :param bands: Which bands to use, order should be red,green,blue
    """
    # pylint: disable=too-many-locals
    assert isinstance(ds, xr.Dataset)

    if bands is None:
        try:
            bands = _guess_rgb_names(list(ds.data_vars))
        except ValueError as e:
            raise ValueError(
                f"Unable to automatically guess RGB colours ({e}). "
                f"Manually specify red, green and blue bands using the "
                f"`bands` parameter."
            ) from e

    is_dask = is_dask_collection(ds)
    if vmin is None:
        if vmax is not None:
            vmin = 0

    if vmax is None:
        if is_dask:
            raise ValueError("Must specify clamp for Dask inputs (e.g. vmax, vmin)")
        _vmin, vmax = _auto_guess_clamp(ds[list(bands)])
        vmin = _vmin if vmin is None else vmin

    assert vmin is not None
    assert vmax is not None

    _b = ds[bands[0]]
    nodata = getattr(_b, "nodata", None)
    dims = (*_b.dims, "band")

    r, g, b = (ds[name].data for name in bands)
    if is_dask:
        # pylint: disable=import-outside-toplevel
        from dask import array as da
        from dask.base import tokenize

        assert _b.chunks is not None
        data = da.map_blocks(
            _np_to_rgba,
            r,
            g,
            b,
            nodata,
            vmin,
            vmax,
            name=f"ro_rgba-{tokenize(r, g, b)}",
            dtype=np.uint8,
            chunks=(*_b.chunks, (4,)),
            new_axis=[r.ndim],
        )
    else:
        data = _np_to_rgba(r, g, b, nodata, vmin, vmax)

    coords = dict(_b.coords.items())
    coords.update(band=xr.DataArray(data=["r", "g", "b", "a"], dims=("band",)))

    rgba = xr.DataArray(data, coords=coords, dims=dims)
    return rgba


def _np_colorize(x, cmap, clip):
    if x.dtype == "bool":
        x = x.astype("uint8")
    if clip:
        x = np.clip(x, 0, cmap.shape[0] - 1)
    return cmap[x]


def _matplotlib_colorize(x, cmap, vmin=None, vmax=None, nodata=None, robust=False):
    from matplotlib import colormaps
    from matplotlib.colors import Normalize

    if cmap is None or isinstance(cmap, str):
        cmap = colormaps.get_cmap(cmap)

    if nodata is not None:
        x = np.where(x == nodata, np.float32("nan"), x)

    if robust:
        if x.dtype.kind != "f":
            x = x.astype("float32")
        _vmin, _vmax = np.nanpercentile(x, [2, 98])

        # do not override configured values
        if vmin is None:
            vmin = _vmin
        if vmax is None:
            vmax = _vmax
    elif x.dtype.kind == "f":
        if vmin is None:
            vmin = np.nanmin(x)

        if vmax is None:
            vmax = np.nanmax(x)

    return cmap(Normalize(vmin=vmin, vmax=vmax)(x), bytes=True)


def colorize(
    x: Any,
    cmap=None,
    attrs=None,
    *,
    clip: bool = False,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    robust: Optional[bool] = None,
) -> xr.DataArray:
    """
    Apply colormap to data.

    There are two modes of operation:

    * Map categorical values from ``x`` to RGBA according to ``cmap`` lookup table.
    * Interpolate into RGBA using matplotlib colormaps (needs matplotlib installed)

    .. note::

       When using matplotlib colormaps with Dask inputs one must configure
       vmin/vmax to ensure all chunks are colorized consistently.


    :param x: Input xarray data array (can be Dask)
    :param cmap: Lookup table ``cmap[x] -> RGB(A)`` or matplotlib colormap
    :param vmin: Valid range to colorize
    :param vmax: Valid range to colorize
    :param robust: Use percentiles for clamping ``vmin=2%, vmax=98%``
    :param attrs: xarray attributes table, if not supplied input attributes are copied across
    :param clip: If ``True`` clip values from ``x`` to be in the safe range for ``cmap``.
    """
    # pylint: disable=too-many-locals

    assert isinstance(x, xr.DataArray)
    _is_dask = is_dask_collection(x.data)

    if isinstance(cmap, np.ndarray):
        assert cmap.ndim == 2
        assert cmap.shape[1] in (3, 4)
        cmap_dtype = cmap.dtype
        _impl = functools.partial(_np_colorize, clip=clip)
        nc = cmap.shape[1]
    else:
        # Assume matplotlib

        # default robust=True for float, non-dask inputs when vmin/vmax/robust are not configured
        if (
            vmin is None
            and vmax is None
            and robust is None
            and x.dtype.kind == "f"
            and not _is_dask
        ):
            robust = True
        elif robust is None:
            robust = False

        _impl = functools.partial(
            _matplotlib_colorize,
            vmin=vmin,
            vmax=vmax,
            nodata=getattr(x, "nodata", None),
            robust=robust,
        )
        nc, cmap_dtype = 4, "uint8"

    if attrs is None:
        attrs = {**x.attrs}
        attrs.pop("nodata", None)

    dims = (*x.dims, "band")
    coords = dict(x.coords.items())
    coords["band"] = xr.DataArray(data=["r", "g", "b", "a"][:nc], dims=("band",))

    if _is_dask:
        from dask import array as da
        from dask import delayed
        from dask.base import tokenize

        _cmap = delayed(cmap) if isinstance(cmap, np.ndarray) else cmap

        assert x.chunks is not None
        data = da.map_blocks(
            _impl,
            x.data,
            _cmap,
            name=f"colorize-{tokenize(x, _cmap, clip, vmin, vmax, robust)}",
            meta=np.ndarray((), cmap_dtype),
            chunks=(*x.chunks, (nc,)),
            new_axis=[x.data.ndim],
        )
    else:
        data = _impl(x.data, cmap)

    return xr.DataArray(data=data, dims=dims, coords=coords, attrs=attrs)


def replace_transparent_pixels(
    rgba: np.ndarray, color: Tuple[int, int, int] = (255, 0, 255)
) -> np.ndarray:
    """
    Convert RGBA to RGB.

    Replaces transparent pixels with a given color.
    """
    assert rgba.ndim == 3
    assert rgba.shape[-1] == 4

    m = rgba[..., -1] == 0
    rgb = rgba[..., :3].copy()
    rgb[m] = color
    return rgb
