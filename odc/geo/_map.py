from typing import Any, Optional, Tuple, Dict

import xarray as xr

from ._interop import have
from ._rgba import colorize, is_rgb
from .converters import map_crs
from .gcp import GCPGeoBox
from .geobox import GeoBox


# pylint: disable=import-outside-toplevel, redefined-builtin, too-many-locals
def _add_to_folium(url, bounds, map, name=None, **kw):
    assert have.folium

    from folium.raster_layers import ImageOverlay

    img_overlay = ImageOverlay(url, bounds, name=name, **kw)
    img_overlay.add_to(map)
    return img_overlay


def _add_to_ipyleaflet(url, bounds, map, name=None, **kw):
    assert have.ipyleaflet

    from ipyleaflet import ImageOverlay, Map

    assert isinstance(map, Map)

    if name is not None:
        kw.update(name=name)

    img_overlay = ImageOverlay(url=url, bounds=bounds, **kw)
    map.add(img_overlay)

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
    resampling: str = "nearest",
    # jpeg options:
    transparent_pixel: Optional[Tuple[int, int, int]] = None,
    # RGB conversion parameters
    cmap: Optional[Any] = None,
    clip: bool = False,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    robust: Optional[bool] = None,
    # passed to ImageOverlay constructor
    **kw,
) -> Any:
    """
    Add image to an interactive map.

    If map is not supplied, image data url and bounds are returned instead.

    :param xx:
        The :py:class:`~xarray.DataArray` to display
    :param map:
        Map object, :py:mod:`folium` and :py:mod:`ipyleaflet` are
        understood; can also be ``None`` which will return an image data
        url and bounds instead.

    :param name:
        The name of the layer as it will appear in :py:mod:`folium` and
        :py:mod:`ipyleaflet` Layer Controls. The default ``None`` will
        use the input array name (e.g. ``xx.name``) if it exists.
    :param fmt:
        Compress image format. Defaults to "png"; also supports "webp",
        "jpeg".
    :param max_size:
        If longest dimension is bigger than this, shrink it down before
        compression; defaults to 4096.
    :param resampling:
        Custom resampling method to use when reprojecting ``xx`` to the
        map CRS; defaults to "nearest".

    :param transparent_pixel:
        Replace transparent pixels with this value, needed for "jpeg".

    :param cmap:
        If supplied array is not RGB use this colormap to turn it into one.
    :param clip:
        When converting to RGB clip input values to fit ``cmap``.
    :param vmin: Used with matplotlib colormaps
    :param vmax: Used with matplotlib colormaps
    :param robust: Used with matplotlib colormaps, ``vmin=2%, vmax=98%``

    :raises ValueError: when map object is not understood
    :return: ImageLayer that was added to a map
    :return: ``(url, bounds)`` when ``map is None``.

    .. seealso:: :py:meth:`~odc.geo.xr.colorize`, :py:meth:`~odc.geo.xr.to_rgba`

    """
    from .xr import ODCExtensionDa

    assert isinstance(xx, xr.DataArray)
    assert isinstance(xx.odc, ODCExtensionDa)

    _add_to = _get_add_to_method(map)  # raises on error
    _crs = map_crs(map)

    # If array xx has a name (xx.name), use it by default
    if (name is None) and (xx.name is not None):
        name = xx.name if isinstance(xx.name, str) else str(xx.name)

    gbox0 = xx.odc.geobox
    assert gbox0 is not None
    native_crs = gbox0.crs
    assert native_crs is not None
    gbox = gbox0

    if isinstance(gbox, GCPGeoBox):
        if _crs is not None:
            gbox = xx.odc.output_geobox(_crs, tight=True)
        else:
            gbox = xx.odc.output_geobox(native_crs, tight=True)

    if _crs is not None and gbox.crs != _crs:
        gbox = gbox.to_crs(_crs, tight=True)

    if not gbox.axis_aligned:
        gbox = GeoBox.from_bbox(
            gbox.boundingbox, resolution=gbox.resolution, tight=True
        )

    if max(*gbox.shape) > max_size:
        gbox = gbox.zoom_to(max_size)

    if gbox is not gbox0:
        xx = xx.odc.reproject(gbox, resampling=resampling)

    if not is_rgb(xx):
        xx = colorize(xx, cmap=cmap, clip=clip, vmin=vmin, vmax=vmax, robust=robust)

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

    return _add_to(url, bounds, map, name=name, **kw)


# pylint: disable=too-many-arguments, protected-access, anomalous-backslash-in-string
def explore(
    xx: Any,
    map: Optional[Any] = None,
    bands: Optional[Tuple[str, str, str]] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: Optional[Any] = None,
    robust: bool = False,
    tiles: Any = "OpenStreetMap",
    attr: Optional[str] = None,
    layer_control: bool = True,
    resampling: str = "nearest",
    map_kwds: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Any:
    """
    Plot xarray data on an interactive :py:mod:`folium` leaflet map for
    rapid data exploration.

    :py:class:`xarray.Dataset` inputs are automatically converted to
    multi-band RGB plots, while single-band :py:class:`xarray.DataArray`
    inputs can be plotted using matplotlib colormaps (needs matplotlib
    installed).

    :param xx:
        The :py:class:`~xarray.Dataset` or :py:class:`~xarray.DataArray`
        to plot on the map.
    :param map:
        An optional existing :py:mod:`folium` map object to plot into.
        By default, a new map object will be created.
    :param bands:
        Bands used for RGB colours when converting from a
        :py:class:`~xarray.Dataset` (order should be red, green, blue).
        By default, the function will attempt to guess bands
        automatically. Ignored for :py:class:`~xarray.DataArray` inputs.
    :param vmin:
        Lower value used for the color stretch.
    :param vmax:
        Upper value used for the color stretch.
    :param cmap:
        The colormap used to colorise single-band arrays. If not
        provided, this will default to 'viridis'. Ignored for multi-band
        inputs.
    :param robust:
        If ``True`` (and ``vmin`` and ``vmax`` are absent), the colormap
        range will be computed based on 2nd and 98th percentiles,
        minimising the influence of extreme values. Used for single-band
        arrays only; ignored for multi-band inputs.
    :param tiles:
        Map tileset to use for the map basemap. Supports any option
        supported by :py:mod:`folium`, including "OpenStreetMap",
        "CartoDB positron", "CartoDB dark_matter" or a custom XYZ URL.
    :param attr:
        Map tile attribution; only required if passing custom tile URL.
    :param layer_control:
        Whether to add a control to the map to show or hide map layers.
        If a layer control already exists, this will be skipped.
    :param resampling:
        Custom resampling method to use when reprojecting ``xx`` to the
        map CRS; defaults to "nearest".
    :param map_kwds:
        Additional keyword arguments to pass to ``folium.Map()``.
    :param \**kwargs:
        Additional keyword arguments to pass to ``.odc.add_to()``.

    :return: A :py:mod:`folium` map containing the plotted xarray data.
    """
    if not have.folium:
        raise ModuleNotFoundError(
            "'folium' is required but not installed. "
            "Please install it before using `.explore()`."
        )

    from folium import Map, LayerControl

    # Update any supplied kwargs with custom params
    map_kwds = {} if map_kwds is None else map_kwds
    kwargs.update(cmap=cmap, vmin=vmin, vmax=vmax, robust=robust, resampling=resampling)
    map_kwds.update(tiles=tiles, attr=attr)

    # If input is a dataset, convert to an RGBA array
    if isinstance(xx, xr.Dataset):
        xx = xx.odc.to_rgba(bands=bands, vmin=vmin, vmax=vmax)

    # Create folium Map if required
    if map is None:
        map = Map(**map_kwds)

    # Add to map and raise a friendly error if data has unsuitable dims
    try:
        xx.odc.add_to(map, **kwargs)
    except ValueError as e:
        raise ValueError(
            "Only 2D single-band (x, y) or 3D multi-band (x, y, band) "
            "arrays are supported by `.explore()`. Please reduce the "
            "dimensions in your array, for example by using `.isel()` "
            "or `.sel()`: `da.isel(time=0).odc.explore()`."
        ) from e

    # Zoom map to extent of data
    map.fit_bounds(xx.odc.map_bounds())

    # Add a layer control if requested and not already added
    layer_control_added = any(
        isinstance(child, LayerControl) for child in map._children.values()
    )
    if layer_control and not layer_control_added:
        LayerControl().add_to(map)

    return map
