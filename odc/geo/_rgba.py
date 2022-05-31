""" Helpers for dealing with RGB(A) images.
"""
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import xarray as xr

from ._interop import is_dask_collection


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


def _to_u8(x: np.ndarray, x_min: float, x_max: float) -> np.ndarray:
    x = np.clip(x, x_min, x_max)

    if x.dtype.kind == "f":
        x = (x - x_min) * (255.0 / (x_max - x_min))
    else:
        x = (x - x_min).astype("uint32") * 255 // (x_max - x_min)
    return x.astype("uint8")


def _np_to_rgba(
    r: np.ndarray,
    g: np.ndarray,
    b: np.ndarray,
    nodata: Optional[float],
    clamp: Tuple[float, float],
) -> np.ndarray:
    rgba = np.zeros((*r.shape, 4), dtype="uint8")

    if r.dtype.kind == "f":
        valid = ~np.isnan(r)
        if nodata is not None:
            valid = valid * (r != nodata)
    elif nodata is not None:
        valid = r != nodata
    else:
        valid = np.ones(r.shape, dtype=np.bool8)

    rgba[..., 3] = valid.astype("uint8") * (0xFF)
    for idx, band in enumerate([r, g, b]):
        rgba[..., idx] = _to_u8(band, *clamp)

    return rgba


def to_rgba(
    ds: Any,
    clamp: Optional[Union[float, Tuple[float, float]]] = None,
    bands: Optional[Tuple[str, str, str]] = None,
) -> xr.DataArray:
    """
    Convert dataset to RGBA image.

    Given :py:class:`xarray.Dataset` with bands ``red,green,blue`` construct :py:class:`xarray.Datarray`
    containing ``uint8`` rgba image.

    :param ds: :py:class:`xarray.Dataset`
    :param clamp:
      ``(min_intensity, max_intensity) | max_intensity == (0, max_intensity)``. Can also
       supply ``None`` for non-dask inputs, in which case clamp is set to ``(0, max(r,g,b))``.

    :param bands: Which bands to use, order should be red,green,blue
    """
    # pylint: disable=too-many-locals
    assert isinstance(ds, xr.Dataset)

    if bands is None:
        bands = _guess_rgb_names(list(ds.data_vars))

    is_dask = is_dask_collection(ds)

    if clamp is None:
        if is_dask:
            raise ValueError("Must specify clamp for Dask inputs")
        clamp = _auto_guess_clamp(ds[list(bands)])
    elif not isinstance(clamp, tuple):
        clamp = (float(0), float(clamp))

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
            clamp,
            name=f"ro_rgba-{tokenize(r, g, b)}",
            dtype=np.uint8,
            chunks=(*_b.chunks, (4,)),
            new_axis=[r.ndim],
        )
    else:
        data = _np_to_rgba(r, g, b, nodata, clamp)

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


def colorize(
    x: Any,
    cmap: Any,
    attrs=None,
    *,
    clip: bool = False,
) -> xr.DataArray:
    """
    Map categorical values from ``x`` to RGBA according to ``cmap`` lookup table.

    :param x: Input xarray data array (can be Dask)
    :param cmap: Lookup table ``cmap[x] -> RGB(A)``
    :param attrs: xarray attributes table, if not supplied input attributes are copied across
    :param clip: If ``True`` clip values from ``x`` to be in the safe range for ``cmap``.
    """
    assert isinstance(x, xr.DataArray)
    assert cmap.ndim == 2
    assert cmap.shape[1] in (3, 4)

    if attrs is None:
        attrs = x.attrs

    nc = cmap.shape[1]
    dims = (*x.dims, "band")
    coords = dict(x.coords.items())
    coords["band"] = xr.DataArray(data=["r", "g", "b", "a"][:nc], dims=("band",))

    if is_dask_collection(x.data):
        # pylint: disable=import-outside-toplevel
        from dask import array as da
        from dask import delayed
        from dask.base import tokenize

        _cmap = delayed(cmap)
        assert x.chunks is not None
        data = da.map_blocks(
            _np_colorize,
            x.data,
            _cmap,
            clip,
            name=f"colorize-{tokenize(x, cmap)}",
            dtype=cmap.dtype,
            chunks=(*x.chunks, (nc,)),
            new_axis=[x.data.ndim],
        )
    else:
        data = _np_colorize(x.data, cmap, clip)

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
