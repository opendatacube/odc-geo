# This file is part of the Open Data Cube, see https://opendatacube.org for more information
#
# Copyright (c) 2015-2020 ODC Contributors
# SPDX-License-Identifier: Apache-2.0
"""
Add geometric extensions to :class:`xarray.Dataset` and :class:`xarray.DataArray` for use
with Data Cube by Monkey Patching those classes.
"""
import warnings
from typing import Optional, Tuple, Union

import xarray

from ._crs import CRS, CRSError
from ._geobox import GeoBox
from .math import affine_from_axis

# from datacube.utils import geometry, spatial_dims
# from datacube.utils.math import affine_from_axis


def _norm_crs(crs):
    if crs is None or isinstance(crs, CRS):
        return crs
    if isinstance(crs, str):
        return CRS(crs)

    raise ValueError(f"Can not interpret {type(crs)} as CRS")


def _get_crs_from_attrs(obj, sdims):
    """Looks for attribute named `crs` containing CRS string
    - Checks spatials coords attrs
    - Checks data variable attrs
    - Checks dataset attrs

    Returns
    =======
    Content for `.attrs[crs]` usually it's a string
    None if not present in any of the places listed above
    """
    crs_set = set()

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


def _get_crs_from_coord(obj, mode="strict"):
    """Looks for dimensionless coordinate with `spatial_ref` attribute.

     obj: Dataset | DataArray
     mode: strict|any|all
        strict -- raise Error if multiple candidates
        any    -- return first one
        all    -- return a list of all found CRSs

    Returns
    =======
    None     - if none found
    crs:str  - if found one
    crs:str  - if found several but mode is any

    (crs: str, crs: str) - if found several and mode=all
    """
    grid_mapping = obj.attrs.get("grid_mapping", None)

    # First check CF convention "pointer"
    if grid_mapping is not None and grid_mapping in obj.coords:
        coord = obj.coords[grid_mapping]
        spatial_ref = coord.attrs.get("spatial_ref", None)
        if spatial_ref is not None:
            return spatial_ref
        raise ValueError(f"Coordinate '{grid_mapping}' has no `spatial_ref` attribute")

    # No explicit `grid_mapping` find some "CRS" coordinate
    candidates = tuple(
        coord.attrs["spatial_ref"]
        for coord in obj.coords.values()
        if coord.ndim == 0 and "spatial_ref" in coord.attrs
    )

    if len(candidates) == 0:
        return None
    if len(candidates) == 1:
        return candidates[0]

    if mode == "strict":
        raise ValueError("Too many candidates when looking for CRS")
    if mode == "all":
        return candidates
    if mode == "any":
        return candidates[0]

    raise ValueError(f"Mode needs to be: strict|any|all got {mode}")


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


def _xarray_affine_impl(obj):
    sdims = spatial_dims(obj, relaxed=True)
    if sdims is None:
        return None, None

    yy, xx = (obj[dim] for dim in sdims)
    fallback_res = (coord.attrs.get("resolution", None) for coord in (xx, yy))

    return affine_from_axis(xx.values, yy.values, fallback_res), sdims


def _xarray_affine(obj):
    transform, _ = _xarray_affine_impl(obj)
    return transform


def _xarray_extent(obj):
    geobox = obj.geobox
    return None if geobox is None else geobox.extent


def _xarray_geobox(obj):
    transform, sdims = _xarray_affine_impl(obj)
    if sdims is None:
        return None

    crs = None
    try:
        crs = _get_crs_from_coord(obj)
    except ValueError:
        pass

    if crs is None:
        crs = _get_crs_from_attrs(obj, sdims)

    if crs is None:
        return None

    try:
        crs = _norm_crs(crs)
    except (ValueError, CRSError):
        warnings.warn(f"Encountered malformed CRS: {crs}")
        return None

    h, w = (obj.coords[dim].size for dim in sdims)

    return GeoBox(w, h, transform, crs)


xarray.Dataset.geobox = property(_xarray_geobox)  # type: ignore
xarray.Dataset.affine = property(_xarray_affine)  # type: ignore
xarray.Dataset.extent = property(_xarray_extent)  # type: ignore
xarray.DataArray.geobox = property(_xarray_geobox)  # type: ignore
xarray.DataArray.affine = property(_xarray_affine)  # type: ignore
xarray.DataArray.extent = property(_xarray_extent)  # type: ignore
