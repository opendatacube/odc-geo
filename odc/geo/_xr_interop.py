# This file is part of the Open Data Cube, see https://opendatacube.org for more information
#
# Copyright (c) 2015-2020 ODC Contributors
# SPDX-License-Identifier: Apache-2.0
"""
Add ``.odc.`` extension to :py:class:`xarray.Dataset` and :class:`xarray.DataArray`.
"""
import functools
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

import numpy
import xarray
from affine import Affine

from ._interop import have, is_dask_collection
from ._rgba import colorize, to_rgba
from .crs import CRS, CRSError, SomeCRS, norm_crs_or_error
from .gcp import GCPGeoBox, GCPMapping
from .geobox import Coordinate, GeoBox
from .geom import Geometry
from .math import affine_from_axis
from .overlap import compute_output_geobox
from .types import Resolution, resxy_, xy_

# pylint: disable=import-outside-toplevel
if have.rasterio:
    from ._cog import to_cog, write_cog
    from ._compress import compress
    from ._map import add_to
    from .warp import rio_reproject

XarrayObject = Union[xarray.DataArray, xarray.Dataset]
XrT = TypeVar("XrT", xarray.DataArray, xarray.Dataset)
F = TypeVar("F", bound=Callable)
SomeGeoBox = Union[GeoBox, GCPGeoBox]

_DEFAULT_CRS_COORD_NAME = "spatial_ref"


@dataclass
class GeoState:
    """
    Geospatial information for xarray object.
    """

    spatial_dims: Optional[Tuple[str, str]] = None
    transform: Optional[Affine] = None
    crs: Optional[CRS] = None
    geobox: Optional[SomeGeoBox] = None
    gcp: Optional[GCPMapping] = None


def _get_crs_from_attrs(obj: XarrayObject, sdims: Tuple[str, str]) -> Optional[CRS]:
    """
    Looks for attribute named ``crs`` containing CRS string.

    - Checks spatials coords attrs
    - Checks data variable attrs
    - Checks dataset attrs

    Returns
    =======
    Content for `.attrs[crs]` usually it's a string
    None if not present in any of the places listed above
    """
    crs_set: Set[CRS] = set()

    def _add_candidate(crs):
        if crs is None:
            return
        if isinstance(crs, str):
            try:
                crs_set.add(CRS(crs))
            except CRSError:
                warnings.warn(f"Failed to parse CRS: {crs}")
        elif isinstance(crs, CRS):
            # support current bad behaviour of injecting CRS directly into
            # attributes in example notebooks
            crs_set.add(crs)
        else:
            warnings.warn(f"Ignoring crs attribute of type: {type(crs)}")

    def process_attrs(attrs):
        _add_candidate(attrs.get("crs", None))
        _add_candidate(attrs.get("crs_wkt", None))

    def process_datavar(x):
        process_attrs(x.attrs)
        for dim in sdims:
            if dim in x.coords:
                process_attrs(x.coords[dim].attrs)

    if isinstance(obj, xarray.Dataset):
        process_attrs(obj.attrs)
        for dv in obj.data_vars.values():
            process_datavar(dv)
    else:
        process_datavar(obj)

    crs = None
    if len(crs_set) > 1:
        warnings.warn("Have several candidates for a CRS")

    if len(crs_set) >= 1:
        crs = crs_set.pop()

    return crs


def spatial_dims(
    xx: Union[xarray.DataArray, xarray.Dataset], relaxed: bool = False
) -> Optional[Tuple[str, str]]:
    """
    Find spatial dimensions of ``xx``.

    Checks for presence of dimensions named:
    ``y, x | latitude, longitude | lat, lon``

    If ``relaxed=True`` and none of the above dimension names are found,
    assume that last two dimensions are spatial dimensions.

    :returns: ``None`` if no dimensions with expected names are found
    :returns: ``('y', 'x') | ('latitude', 'longitude') | ('lat', 'lon')``
    """
    guesses = [("y", "x"), ("latitude", "longitude"), ("lat", "lon")]

    _dims = [str(dim) for dim in xx.dims]
    dims = set(_dims)
    for guess in guesses:
        if dims.issuperset(guess):
            return guess

    if relaxed and len(_dims) >= 2:
        return _dims[-2], _dims[-1]

    return None


def _mk_crs_coord(
    crs: CRS,
    name: str = _DEFAULT_CRS_COORD_NAME,
    gcps=None,
) -> xarray.DataArray:
    # pylint: disable=protected-access

    cf = crs.proj.to_cf()
    epsg = 0 if crs.epsg is None else crs.epsg
    crs_wkt = cf.get("crs_wkt", None) or crs.wkt

    if gcps is not None:
        cf["gcps"] = _gcps_to_json(gcps)

    return xarray.DataArray(
        numpy.asarray(epsg, "int32"),
        name=name,
        dims=(),
        attrs={"spatial_ref": crs_wkt, **cf},
    )


def _gcps_to_json(gcps):
    def _to_feature(p):
        coords = [p.x, p.y] if p.z is None else [p.x, p.y, p.z]

        return {
            "type": "Feature",
            "properties": {
                "id": str(p.id),
                "info": (p.info or ""),
                "row": p.row,
                "col": p.col,
            },
            "geometry": {"type": "Point", "coordinates": coords},
        }

    return {"type": "FeatureCollection", "features": list(map(_to_feature, gcps))}


def _coord_to_xr(name: str, c: Coordinate, **attrs) -> xarray.DataArray:
    """
    Construct xr.DataArray from named Coordinate object.

    This can then be used to define coordinates for ``xr.Dataset|xr.DataArray``
    """
    attrs = dict(units=c.units, resolution=c.resolution, **attrs)
    return xarray.DataArray(
        c.values, coords={name: c.values}, dims=(name,), attrs=attrs
    )


def assign_crs(
    xx: XrT,
    crs: SomeCRS,
    crs_coord_name: str = _DEFAULT_CRS_COORD_NAME,
) -> XrT:
    """
    Assign CRS for a non-georegistered array or dataset.

    Returns a new object with CRS information populated.

    .. code-block:: python

        xx = xr.open_rasterio("some-file.tif")
        print(xx.odc.crs)
        print(xx.astype("float32").crs)


    :param xx: :py:class:`~xarray.Dataset` or :py:class:`~xarray.DataArray`
    :param crs: CRS to assign
    :param crs_coord_name: how to name crs coordinate (defaults to ``spatial_ref``)
    """
    crs = norm_crs_or_error(crs)
    crs_coord = _mk_crs_coord(crs, name=crs_coord_name)
    xx = xx.assign_coords({crs_coord_name: crs_coord})

    if isinstance(xx, xarray.DataArray):
        xx.encoding.update(grid_mapping=crs_coord_name)
    elif isinstance(xx, xarray.Dataset):
        for band in xx.data_vars.values():
            band.encoding.update(grid_mapping=crs_coord_name)

    return xx


def xr_coords(
    gbox: SomeGeoBox, crs_coord_name: Optional[str] = _DEFAULT_CRS_COORD_NAME
) -> Dict[Hashable, xarray.DataArray]:
    """
    Dictionary of Coordinates in xarray format.

    :param crs_coord_name:
       Use custom name for CRS coordinate, default is "spatial_ref". Set to ``None`` to not generate
       CRS coordinate at all.

    :returns:
      Dictionary ``name:str -> xr.DataArray``. Where names are either ``y,x`` for projected or
      ``latitude, longitude`` for geographic.

    """
    attrs = {}
    crs = gbox.crs
    if crs is not None:
        attrs["crs"] = str(crs)

    gcps = None

    if isinstance(gbox, GCPGeoBox):
        coords: Dict[Hashable, xarray.DataArray] = {
            name: _mk_pixel_coord(name, sz, None)
            for name, sz in zip(gbox.dimensions, gbox.shape)
        }
        gcps = gbox.gcps()
    elif gbox.axis_aligned:
        coords = {
            name: _coord_to_xr(name, coord, **attrs)
            for name, coord in gbox.coordinates.items()
        }
    else:
        coords = {
            name: _mk_pixel_coord(name, sz, gbox.transform)
            for name, sz in zip(gbox.dimensions, gbox.shape)
        }

    if crs_coord_name is not None and crs is not None:
        coords[crs_coord_name] = _mk_crs_coord(crs, crs_coord_name, gcps=gcps)

    return coords


def _mk_pixel_coord(
    name: str,
    sz: int,
    transform: Optional[Affine],
) -> xarray.DataArray:
    data = numpy.arange(0.5, sz, dtype="float32")
    xx = xarray.DataArray(
        data, coords={name: data}, dims=(name,), attrs={"units": "pixel"}
    )
    if transform is not None:
        xx.encoding["_transform"] = transform[:6]
    return xx


def _locate_crs_coords(xx: XarrayObject) -> List[xarray.DataArray]:
    grid_mapping = xx.encoding.get("grid_mapping", None)
    if grid_mapping is None:
        grid_mapping = xx.attrs.get("grid_mapping")

    if grid_mapping is not None:
        # Specific mapping is defined via NetCDF/CF convention
        coord = xx.coords.get(grid_mapping, None)
        if coord is None:
            warnings.warn(
                f"grid_mapping={grid_mapping} is not pointing to valid coordinate"
            )
            return []
        return [coord]

    # Find all dimensionless coordinates with `spatial_ref|crs_wkt` attribute present
    return [
        coord
        for coord in xx.coords.values()
        if coord.ndim == 0
        and ("spatial_ref" in coord.attrs or "crs_wkt" in coord.attrs)
    ]


def _extract_crs(crs_coord: xarray.DataArray) -> Optional[CRS]:
    _wkt = crs_coord.attrs.get("spatial_ref", None)  # GDAL convention?
    if _wkt is None:
        _wkt = crs_coord.attrs.get("crs_wkt", None)  # CF convention
    if _wkt is None:
        return None
    try:
        return CRS(_wkt)
    except CRSError:
        return None


def _extract_gcps(crs_coord: xarray.DataArray) -> Optional[GCPMapping]:
    gcps = crs_coord.attrs.get("gcps", None)
    if gcps is None:
        return None
    crs = _extract_crs(crs_coord)
    try:
        wld = Geometry(gcps, crs=crs)
        pix = [
            xy_(f["properties"]["col"], f["properties"]["row"])
            for f in gcps["features"]
        ]
        return GCPMapping(pix, wld)
    except (IndexError, KeyError, ValueError):
        return None


def _extract_transform(src: XarrayObject, sdims: Tuple[str, str]) -> Optional[Affine]:
    if any(dim not in src.coords for dim in sdims):
        # special case of no spatial dims at all
        # happens for GCP sources loaded by rioxarray
        return None

    _yy, _xx = (src[dim] for dim in sdims)
    rx, ry = (coord.attrs.get("resolution", None) for coord in (_xx, _yy))
    fallback_res = None
    if rx is not None and ry is not None:
        fallback_res = resxy_(float(rx), float(ry))

    try:
        transform = affine_from_axis(_xx.values, _yy.values, fallback_res)
    except ValueError:
        # this can fail when any dimension is shorter than 2 elements
        return None

    _pix2world = _xx.encoding.get("_transform", None)
    if _pix2world is not None:
        # non-axis aligned geobox detected
        # adjust transform
        #  world <- pix' <- pix
        transform = Affine(*_pix2world) * transform
    return transform


def _locate_geo_info(src: XarrayObject) -> GeoState:
    # pylint: disable=too-many-locals
    sdims = spatial_dims(src, relaxed=True)
    if sdims is None:
        return GeoState()

    transform = _extract_transform(src, sdims)
    crs: Optional[CRS] = None
    geobox: Optional[SomeGeoBox] = None
    gcp: Optional[GCPMapping] = None

    ny, nx = (src.coords[dim].shape[0] for dim in sdims)

    _crs_coords = _locate_crs_coords(src)
    num_candidates = len(_crs_coords)
    if num_candidates > 0:
        if num_candidates > 1:
            warnings.warn("Multiple CRS coordinates are present")
        crs = _extract_crs(_crs_coords[0])
        gcp = _extract_gcps(_crs_coords[0])
    else:
        # try looking in attributes
        crs = _get_crs_from_attrs(src, sdims)

    if gcp is not None:
        geobox = GCPGeoBox((ny, nx), gcp, transform)
    elif transform is not None:
        geobox = GeoBox((ny, nx), transform, crs)

    return GeoState(
        spatial_dims=sdims, transform=transform, crs=crs, geobox=geobox, gcp=gcp
    )


def _wrap_op(method: F) -> F:
    @functools.wraps(method, assigned=("__doc__",))
    def wrapped(*args, **kw):
        # pylint: disable=protected-access
        _self, *rest = args
        return method(_self._xx, *rest, **kw)

    return wrapped  # type: ignore


def xr_reproject(
    src: XrT,
    how: Union[SomeCRS, GeoBox],
    *,
    resampling: Union[str, int] = "nearest",
    dst_nodata: Optional[float] = None,
    **kw,
) -> XrT:
    """
    Reproject raster to different projection/resolution.

    This method uses :py:mod:`rasterio`.
    """
    if isinstance(src, xarray.DataArray):
        return _xr_reproject_da(
            src, how, resampling=resampling, dst_nodata=dst_nodata, **kw
        )
    return _xr_reproject_ds(
        src, how, resampling=resampling, dst_nodata=dst_nodata, **kw
    )


def _xr_reproject_ds(
    src: Any,
    how: Union[SomeCRS, GeoBox],
    *,
    resampling: Union[str, int] = "nearest",
    dst_nodata: Optional[float] = None,
    **kw,
) -> xarray.Dataset:
    assert isinstance(src, xarray.Dataset)

    if have.rasterio is False:  # pragma: nocover
        raise RuntimeError("Please install `rasterio` to use this method")

    assert isinstance(src.odc, ODCExtensionDs)
    if src.odc.geobox is None:
        raise ValueError("Can not reproject non-georegistered array.")

    if isinstance(how, GeoBox):
        dst_geobox = how
    else:
        dst_geobox = src.odc.output_geobox(how)

    def _maybe_reproject(dv: xarray.DataArray):
        if dv.odc.geobox is None:
            # pass-through data variables without a geobox
            strip_coords = [str(c.name) for c in _locate_crs_coords(dv)]
            if len(strip_coords) > 0:
                dv = dv.drop_vars(strip_coords)
            return dv
        return _xr_reproject_da(
            dv, how=dst_geobox, resampling=resampling, dst_nodata=dst_nodata, **kw
        )

    return src.map(_maybe_reproject)


def _xr_reproject_da(
    src: Any,
    how: Union[SomeCRS, GeoBox],
    *,
    resampling: Union[str, int] = "nearest",
    dst_nodata: Optional[float] = None,
    **kw,
) -> xarray.DataArray:
    """
    Reproject raster to different projection/resolution.

    This method uses :py:mod:`rasterio`.
    """
    assert isinstance(src, xarray.DataArray)

    if have.rasterio is False:  # pragma: nocover
        raise RuntimeError("Please install `rasterio` to use this method")

    assert isinstance(src.odc, ODCExtensionDa)  # for mypy sake
    src_gbox = src.odc.geobox

    if src_gbox is None:
        raise ValueError("Can not reproject non-georegistered array.")

    if isinstance(how, GeoBox):
        dst_geobox = how
    else:
        dst_geobox = src.odc.output_geobox(how)

    if is_dask_collection(src):
        raise NotImplementedError("Dask inputs are not yet supported.")

    # compute destination shape by replacing spatial dimensions shape
    ydim = src.odc.ydim
    assert ydim + 1 == src.odc.xdim
    dst_shape = (*src.shape[:ydim], *dst_geobox.shape, *src.shape[ydim + 2 :])

    dst = numpy.empty(dst_shape, dtype=src.dtype)
    src_nodata = kw.pop("src_nodata", None)
    if src_nodata is None:
        src_nodata = src.odc.nodata
    if dst_nodata is None:
        dst_nodata = src_nodata

    dst = rio_reproject(
        src.values,
        dst,
        src_gbox,
        dst_geobox,
        resampling=resampling,
        src_nodata=src_nodata,
        dst_nodata=dst_nodata,
        ydim=ydim,
        **kw,
    )

    attrs = src.attrs.copy()
    if dst_nodata is None:
        attrs.pop("nodata", None)
        attrs.pop("_FillValue", None)
    else:
        attrs.update(nodata=dst_nodata)

    # new set of coords (replace x,y dims)
    coords = dict((k, v) for k, v in src.coords.items() if k not in src.dims)
    coords.update(xr_coords(dst_geobox))

    dims = (*src.dims[:ydim], *dst_geobox.dimensions, *src.dims[ydim + 2 :])

    return xarray.DataArray(dst, coords=coords, dims=dims, attrs=attrs)


class ODCExtension:
    """
    ODC extension base class.

    Common accessors for both Array/Dataset.
    """

    def __init__(self, state: GeoState):
        self._state = state

    @property
    def spatial_dims(self) -> Optional[Tuple[str, str]]:
        """Return names of spatial dimensions, or ``None``."""
        return self._state.spatial_dims

    @property
    def transform(self) -> Optional[Affine]:
        return self._state.transform

    affine = transform

    @property
    def crs(self) -> Optional[CRS]:
        """Query :py:class:`~odc.geo.crs.CRS`."""
        return self._state.crs

    @property
    def geobox(self) -> Optional[SomeGeoBox]:
        """Query :py:class:`~odc.geo.geobox.GeoBox` or :py:class:`~odc.geo.gcp.GCPGeoBox`."""
        return self._state.geobox

    def output_geobox(self, crs: SomeCRS, **kw) -> GeoBox:
        """
        Compute geobox of this data in other projection.

        ..seealso:: :py:meth:`odc.geo.overlap.compute_output_geobox`
        """
        gbox = self.geobox
        if gbox is None:
            raise ValueError("Not geo registered")

        return compute_output_geobox(gbox, crs, **kw)

    def map_bounds(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """See :py:meth:`odc.geo.geobox.GeoBox.map_bounds`."""
        gbox = self.geobox
        if gbox is None:
            raise ValueError("Not geo registered")

        return gbox.map_bounds()


@xarray.register_dataarray_accessor("odc")
class ODCExtensionDa(ODCExtension):
    """
    ODC extension for :py:class:`xarray.DataArray`.
    """

    def __init__(self, xx: xarray.DataArray):
        ODCExtension.__init__(self, _locate_geo_info(xx))
        self._xx = xx

    @property
    def uncached(self) -> "ODCExtensionDa":
        return ODCExtensionDa(self._xx)

    @property
    def ydim(self) -> int:
        """Index of the Y dimension."""
        if (sdims := self.spatial_dims) is not None:
            return self._xx.dims.index(sdims[0])
        raise ValueError("Can't locate spatial dimensions")

    @property
    def xdim(self) -> int:
        """Index of the X dimension."""
        if (sdims := self.spatial_dims) is not None:
            return self._xx.dims.index(sdims[1])
        raise ValueError("Can't locate spatial dimensions")

    def assign_crs(
        self, crs: SomeCRS, crs_coord_name: str = _DEFAULT_CRS_COORD_NAME
    ) -> xarray.DataArray:
        """See :py:meth:`odc.geo.xr.assign_crs`."""
        return assign_crs(self._xx, crs=crs, crs_coord_name=crs_coord_name)

    @property
    def nodata(self) -> Optional[float]:
        """Extract ``nodata/_FillValue`` attribute if set."""
        attrs = self._xx.attrs
        for k in ["nodata", "_FillValue"]:
            nodata = attrs.get(k, None)
            if nodata is not None:
                return float(nodata)

        return None

    colorize = _wrap_op(colorize)

    if have.rasterio:
        write_cog = _wrap_op(write_cog)
        to_cog = _wrap_op(to_cog)
        compress = _wrap_op(compress)
        reproject = _wrap_op(_xr_reproject_da)
        add_to = _wrap_op(add_to)


@xarray.register_dataset_accessor("odc")
class ODCExtensionDs(ODCExtension):
    """
    ODC extension for :py:class:`xarray.Dataset`.
    """

    def __init__(self, ds: xarray.Dataset):
        ODCExtension.__init__(self, _locate_geo_info(ds))
        self._xx = ds

    @property
    def uncached(self) -> "ODCExtensionDs":
        return ODCExtensionDs(self._xx)

    def assign_crs(
        self, crs: SomeCRS, crs_coord_name: str = _DEFAULT_CRS_COORD_NAME
    ) -> xarray.Dataset:
        return assign_crs(self._xx, crs=crs, crs_coord_name=crs_coord_name)

    def to_rgba(
        self,
        bands: Optional[Tuple[str, str, str]] = None,
        *,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ) -> xarray.DataArray:
        return to_rgba(self._xx, bands=bands, vmin=vmin, vmax=vmax)

    if have.rasterio:
        reproject = _wrap_op(_xr_reproject_ds)


ODCExtensionDs.to_rgba.__doc__ = to_rgba.__doc__


def _xarray_geobox(xx: XarrayObject) -> Optional[GeoBox]:
    if isinstance(xx, xarray.DataArray):
        return xx.odc.geobox
    for dv in xx.data_vars.values():
        geobox = dv.odc.geobox
        if geobox is not None:
            return geobox
    return None


def register_geobox():
    """
    Backwards compatiblity layer for datacube ``.geobox`` property.
    """
    xarray.Dataset.geobox = property(_xarray_geobox)  # type: ignore
    xarray.DataArray.geobox = property(_xarray_geobox)  # type: ignore


def wrap_xr(
    im: Any,
    gbox: SomeGeoBox,
    *,
    time=None,
    nodata=None,
    crs_coord_name: Optional[str] = _DEFAULT_CRS_COORD_NAME,
    **attrs,
) -> xarray.DataArray:
    """
    Wrap xarray around numpy array with CRS and x,y coords.

    :param im: numpy array to wrap, last two axes are Y,X
    :param gbox: Geobox, must same shape as last two axis of ``im``
    :param time: optional time axis value(s), defaults to None
    :param nodata: optional `nodata` value, defaults to None
    :param attrs: Any other attributes to set on the result
    :return: xarray DataArray
    """
    assert im.shape[-2:] == gbox.shape

    prefix_dims: Tuple[str, ...] = ("time",) if im.ndim > 2 else ()

    dims = (*prefix_dims, *gbox.dimensions)
    coords = xr_coords(gbox, crs_coord_name=crs_coord_name)

    if time is not None:
        if not isinstance(time, xarray.DataArray):
            if len(prefix_dims) > 0 and isinstance(time, (str, datetime)):
                time = [time]

            time = xarray.DataArray(time, dims=prefix_dims).astype("datetime64[ns]")

        coords["time"] = time

    if nodata is not None:
        attrs = dict(nodata=nodata, **attrs)

    out = xarray.DataArray(im, coords=coords, dims=dims, attrs=attrs)
    if crs_coord_name is not None:
        out.encoding["grid_mapping"] = crs_coord_name
    return out


def xr_zeros(
    geobox: SomeGeoBox,
    dtype="float64",
    *,
    chunks: Optional[Union[Tuple[int, int], Tuple[int, int, int]]] = None,
    time=None,
    crs_coord_name: Optional[str] = _DEFAULT_CRS_COORD_NAME,
    **kw,
) -> xarray.DataArray:
    """
    Construct geo-registered xarray from a :py:class:`~odc.geo.geobox.GeoBox`.

    :param gbox: Desired footprint and resolution
    :param dtype: Pixel data type
    :param chunks: Create a dask array instead of numpy array
    :param time: When set adds time dimension
    :param crs_coord_name: allows to change name of the crs coordinate variable

    :return: :py:class:`xarray.DataArray` filled with zeros (numpy or dask)
    """
    if time is not None:
        _shape: Tuple[int, ...] = (len(time), *geobox.shape.yx)
    else:
        _shape = geobox.shape.yx

    if chunks is not None:
        from dask import array as da  # pylint: disable=import-outside-toplevel

        return wrap_xr(
            da.zeros(_shape, dtype=dtype, chunks=chunks),
            geobox,
            crs_coord_name=crs_coord_name,
            time=time,
            **kw,
        )

    return wrap_xr(
        numpy.zeros(_shape, dtype=dtype),
        geobox,
        crs_coord_name=crs_coord_name,
        time=time,
        **kw,
    )


def rasterize(
    poly: Geometry,
    how: Union[float, int, Resolution, GeoBox],
    *,
    value_inside: bool = True,
    all_touched: bool = False,
) -> xarray.DataArray:
    """
    Generate raster from geometry.

    This method is a wrapper for :py:meth:`rasterio.features.make_mask`.

    :param poly:
       Geometry shape to rasterize.

    :param how:
        This could be either just resolution or a GeoBox that fully defines output
        raster extent/resolution/projection.

    :param all_touched:
        If ``True``, all pixels touched by geometries will be burned in.  If
        ``False``, only pixels whose center is within the polygon or that
        are selected by Bresenham's line algorithm will be burned in.

    :param value_inside:
        By default pixels inside a polygon will have value of ``True`` and ``False``
        outside, but this can be flipped.

    :return: geo-registered data array
    """
    # pylint: disable=import-outside-toplevel

    if have.rasterio is False:  # pragma: nocover
        raise RuntimeError("Please install `rasterio` to use this method")

    from rasterio.features import geometry_mask

    if isinstance(how, GeoBox):
        geobox = how
    else:
        geobox = GeoBox.from_geopolygon(poly, resolution=how)

    if poly.crs != geobox.crs and geobox.crs is not None:
        poly = poly.to_crs(geobox.crs)

    pix = geometry_mask(
        [poly.geom],
        geobox.shape,
        geobox.transform,
        all_touched=all_touched,
        invert=value_inside,
    )
    return wrap_xr(pix, geobox)
