# This file is part of the Open Data Cube, see https://opendatacube.org for more information
#
# Copyright (c) 2015-2020 ODC Contributors
# SPDX-License-Identifier: Apache-2.0
"""
Add ``.odc.`` extension to :py:class:`xarray.Dataset` and :class:`xarray.DataArray`.
"""
import warnings
from dataclasses import dataclass
from typing import Dict, Hashable, List, Optional, Set, Tuple, TypeVar, Union

import numpy
import xarray
from affine import Affine

from .crs import CRS, CRSError, SomeCRS, norm_crs_or_error
from .geobox import Coordinate, GeoBox
from .math import affine_from_axis

XarrayObject = Union[xarray.DataArray, xarray.Dataset]
XrT = TypeVar("XrT", xarray.DataArray, xarray.Dataset)


@dataclass
class GeoState:
    """
    Geospatial information for xarray object.
    """

    spatial_dims: Optional[Tuple[str, str]] = None
    transform: Optional[Affine] = None
    crs: Optional[CRS] = None
    geobox: Optional[GeoBox] = None


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
    """Find spatial dimensions of `xx`.

    Checks for presence of dimensions named:
      y, x | latitude, longitude | lat, lon

    Returns
    =======
    None -- if no dimensions with expected names are found
    ('y', 'x') | ('latitude', 'longitude') | ('lat', 'lon')

    If *relaxed* is True and none of the above dimension names are found,
    assume that last two dimensions are spatial dimensions.
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


def _mk_crs_coord(crs: CRS, name: str = "spatial_ref") -> xarray.DataArray:
    # pylint: disable=protected-access

    cf = crs.proj.to_cf()
    cf.setdefault("grid_mapping_name", "??")
    epsg = 0 if crs.epsg is None else crs.epsg
    crs_wkt = cf.get("crs_wkt", None) or crs.wkt

    return xarray.DataArray(
        numpy.asarray(epsg, "int32"),
        name=name,
        dims=(),
        attrs={"spatial_ref": crs_wkt, **cf},
    )


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
    crs_coord_name: str = "spatial_ref",
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
    gbox: GeoBox, crs_coord_name: Optional[str] = "spatial_ref"
) -> Dict[Hashable, xarray.DataArray]:
    """
    Dictionary of Coordinates in xarray format.

    :param crs_coord_name:
       Use custom name for CRS coordinate, default is "spatial_ref".
       Set to ``None`` to not generate CRS coordinate at all.

    Returns
    =======

    Dictionary name:str -> xr.DataArray

    where names are either ``y,x`` for projected or ``latitude, longitude`` for geographic.

    """
    attrs = {}
    crs = gbox.crs
    if crs is not None:
        attrs["crs"] = str(crs)

    coords: Dict[Hashable, xarray.DataArray] = {
        name: _coord_to_xr(name, coord, **attrs)
        for name, coord in gbox.coordinates.items()
    }

    if crs_coord_name is not None and crs is not None:
        coords[crs_coord_name] = _mk_crs_coord(crs, crs_coord_name)

    return coords


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


def _locate_geo_info(src: XarrayObject) -> GeoState:
    sdims = spatial_dims(src, relaxed=True)
    if sdims is None:
        return GeoState()

    transform: Optional[Affine] = None
    crs: Optional[CRS] = None
    geobox: Optional[GeoBox] = None

    _yy, _xx = (src[dim] for dim in sdims)
    fallback_res = (coord.attrs.get("resolution", None) for coord in (_xx, _yy))

    try:
        transform = affine_from_axis(_xx.values, _yy.values, fallback_res)
    except ValueError:
        # this can fail when any dimension is shorter than 2 elements
        pass

    _crs_coords = _locate_crs_coords(src)
    num_candiates = len(_crs_coords)
    if num_candiates > 0:
        if num_candiates > 1:
            warnings.warn("Multiple CRS coordinates are present")
        crs = _extract_crs(_crs_coords[0])
    else:
        # try looking in attributes
        crs = _get_crs_from_attrs(src, sdims)

    if transform is not None:
        width = _xx.shape[0]
        height = _yy.shape[0]
        geobox = GeoBox(width, height, transform, crs)

    return GeoState(spatial_dims=sdims, transform=transform, crs=crs, geobox=geobox)


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
        return self._state.crs

    @property
    def geobox(self) -> Optional[GeoBox]:
        return self._state.geobox


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

    def assign_crs(
        self, crs: SomeCRS, crs_coord_name: str = "spatial_ref"
    ) -> xarray.DataArray:
        return assign_crs(self._xx, crs=crs, crs_coord_name=crs_coord_name)


@xarray.register_dataset_accessor("odc")
class ODCExtensionDs(ODCExtension):
    """
    ODC extension for :py:class:`xarray.Dataset`.
    """

    def __init__(self, ds: xarray.Dataset):
        ODCExtension.__init__(self, _locate_geo_info(ds))
        self._ds = ds

    @property
    def uncached(self) -> "ODCExtensionDs":
        return ODCExtensionDs(self._ds)

    def assign_crs(
        self, crs: SomeCRS, crs_coord_name: str = "spatial_ref"
    ) -> xarray.Dataset:
        return assign_crs(self._ds, crs=crs, crs_coord_name=crs_coord_name)


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
