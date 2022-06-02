from typing import Any, Optional, Tuple

import xarray as xr

from ._interop import have
from ._rgba import colorize, is_rgb


# pylint: disable=import-outside-toplevel, redefined-builtin, too-many-locals
def _add_to_folium(url, bounds, map, name=None, **kw):
    assert have.folium

    from folium.raster_layers import ImageOverlay

    img_overlay = ImageOverlay(url, bounds, **kw)
    img_overlay.add_to(map, name=name)
    return img_overlay


def _add_to_ipyleaflet(url, bounds, map, name=None, **kw):
    assert have.ipyleaflet

    from ipyleaflet import ImageOverlay, Map

    assert isinstance(map, Map)

    if name is not None:
        kw.update(name=name)

    img_overlay = ImageOverlay(url=url, bounds=bounds, **kw)
    map.add_layer(img_overlay)

    return img_overlay


def _get_add_to_method(map):
    if map is None:
        return None

    map_module = getattr(map, "__module__", "")

    if "folium" in map_module:
        return _add_to_folium

    if "ipyleaflet" in map_module:
        return _add_to_ipyleaflet

    raise ValueError(f"Not sure how to add image to: '{type(map)}'")


def add_to(
    xx: Any,
    map: Any,
    *,
    name: Optional[str] = None,
    fmt: str = "png",
    max_size: int = 4096,
    # jpeg options:
    transparent_pixel: Optional[Tuple[int, int, int]] = None,
    # RGB conversion parameters
    cmap: Optional[Any] = None,
    clip: bool = False,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    # passed to ImageOverlay constructor
    **kw,
) -> Any:
    """
    Add image to a map.

    If map is not supplied, image data url and bounds are returned instead.

    :param xx: array to display
    :param map:
       Map object, :py:mod:`folium` and :py:mod:`ipyleaflet` are understood, can be ``None``.

    :param name: Image layer name
    :param fmt: compress image format, defaults to "png", can be "webp", "jpeg" as well.
    :param max_size:
       If longest dimension is bigger than this, shrink it down before compression, defaults to 4096
    :param transparent_pixel: Replace transparent pixels with this value, needed for "jpeg".

    :param cmap: If supplied array is not RGB use this colormap to turn it into one
    :param clip: When converting to RGB clip input values to fit ``cmap``.
    :param vmin: Used with matplotlib colormaps
    :param vmax: Used with matplotlib colormapa

    :raises ValueError: when map object is not understood
    :return: ImageLayer that was added to a map
    :return: ``(url, bounds)`` when ``map is None``.

    .. seealso:: :py:meth:`~odc.geo.xr.colorize`, :py:meth:`~odc.geo.xr.to_rgba`

    """
    from .xr import ODCExtensionDa

    assert isinstance(xx, xr.DataArray)
    assert isinstance(xx.odc, ODCExtensionDa)

    _add_to = _get_add_to_method(map)  # raises on error

    need_reproject = False

    gbox = xx.odc.geobox
    assert gbox is not None
    assert gbox.crs is not None

    if gbox.crs.epsg != 3857:
        need_reproject = True
        gbox = gbox.to_crs("epsg:3857", tight=True)

    if max(*gbox.shape) > max_size:
        need_reproject = True
        gbox = gbox.zoom_to(max_size)

    if need_reproject:
        xx = xx.odc.reproject(gbox)

    if not is_rgb(xx):
        xx = colorize(xx, cmap=cmap, clip=clip, vmin=vmin, vmax=vmax)

    compress_opts = [fmt]
    for opt in ["zlevel", "quality"]:
        if (v := kw.pop(opt, None)) is not None:
            compress_opts.append(v)

    url = xx.odc.compress(
        *compress_opts, as_data_url=True, transparent=transparent_pixel
    )
    bounds = gbox.map_bounds()
    if _add_to is None:
        return url, bounds

    return _add_to(url, gbox.map_bounds(), map, name=name, **kw)
