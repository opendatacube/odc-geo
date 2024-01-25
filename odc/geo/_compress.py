import base64
import warnings
from typing import Optional, Tuple, Union

import numpy as np
import rasterio
import rasterio.env
import rasterio.session
import xarray as xr

from ._interop import is_dask_collection
from ._rgba import replace_transparent_pixels

# Driver, Compress Option Name
_fmt_info = {
    "png": ("PNG", "zlevel", "image/png"),
    "jpeg": ("JPEG", "quality", "image/jpeg"),
    "webp": ("WEBP", "quality", "image/webp"),
}


def _verify_can_compress(xx: xr.DataArray):
    """Returns error if array dimensions are not suitable for compress"""
    if xx.ndim > 2:
        xx = xx.squeeze()

    if xx.ndim not in (2, 3):
        raise ValueError(
            f"Expected a 2 or 3 dimensional array; got {xx.ndim} dimensions {xx.dims}."
        )


def _compress_image(im: np.ndarray, driver="PNG", **opts) -> bytes:
    if im.ndim > 2:
        im = np.squeeze(im)

    if im.ndim == 3:
        h, w, nc = im.shape  # type: ignore
        bands = np.transpose(im, axes=(2, 0, 1))  # Y,X,B -> B,Y,X
    elif im.ndim == 2:
        (h, w), nc = im.shape, 1
        bands = im.reshape(nc, h, w)
    else:
        raise ValueError(
            f"Expected a 2 or 3 dimensional array; got {im.ndim} dimensions."
        )

    rio_opts = {
        "width": w,
        "height": h,
        "count": nc,
        "driver": driver,
        "dtype": im.dtype,
        **opts,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", rasterio.errors.NotGeoreferencedWarning)

        with rasterio.env.Env(session=rasterio.session.DummySession()):
            with rasterio.MemoryFile() as mem:
                with mem.open(**rio_opts) as dst:
                    dst.write(bands)
                return mem.read()


def compress(
    xx,
    /,
    *args,
    as_data_url=False,
    transparent: Optional[Tuple[int, int, int]] = None,
    **kw,
) -> Union[str, bytes]:
    """
    Save image to RAM in jpeg/png/webp format.

    .. code:: python

       png_bytes = compress(xx)  # default is PNG
       png_bytes = compress(xx, "png", 9) # compression settings

       # - Make data url with JPEG quality of 85
       # - Use black for transparent pixels (JPEG doesn't support transparency)
       # - Result is an ASCII string
       url = compress(xx, "jpeg", 85, as_data_url=True, transparent=(0,0,0))

    :param xx: DataArray to compress
    :param transparent:
      Pixel value to use for transparent pixels, useful for jpeg output.

    """
    # Raise error early if xx has unsuitable dims
    _verify_can_compress(xx)

    fmt = "png"
    opts = {}
    if len(args) >= 1:
        fmt, *_ = args

    driver, opt_name, mime_type = _fmt_info.get(fmt.lower(), (None, "", ""))
    if driver is None:
        raise ValueError(f"Format '{fmt}', is not suported, try jpeg/png/webp")

    if len(args) >= 2:
        _, comp, *_ = args
        opts[opt_name] = comp

    if isinstance(xx, xr.DataArray):
        xx = xx.data

    if is_dask_collection(xx):
        xx = xx.compute()

    if transparent is not None:
        xx = replace_transparent_pixels(xx, transparent)

    _bytes = _compress_image(xx, driver, **opts, **kw)
    if not as_data_url:
        return _bytes

    return f"data:{mime_type};base64," + base64.encodebytes(_bytes).decode("ascii")
