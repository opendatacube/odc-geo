# This file is part of the Open Data Cube, see https://opendatacube.org for more information
#
# Copyright (c) 2015-2020 ODC Contributors
# SPDX-License-Identifier: Apache-2.0
import math
from collections import OrderedDict, namedtuple
from typing import Dict, Hashable, List, Optional, Tuple, Union

import numpy
import xarray as xr
from affine import Affine

from ._crs import CRS, MaybeCRS, norm_crs_or_error
from ._geom import (
    BoundingBox,
    Geometry,
    bbox_intersection,
    bbox_union,
    polygon_from_transform,
)
from ._overlap import is_affine_st
from .math import is_almost_int

from ._roi import roi_normalise, roi_shape

Coordinate = namedtuple("Coordinate", ("values", "units", "resolution"))


def _align_pix(left: float, right: float, res: float, off: float) -> Tuple[float, int]:
    if res < 0:
        res = -res
        val = math.ceil((right - off) / res) * res + off
        width = max(1, int(math.ceil((val - left - 0.1 * res) / res)))
    else:
        val = math.floor((left - off) / res) * res + off
        width = max(1, int(math.ceil((right - val - 0.1 * res) / res)))
    return val, width


class GeoBox:
    """
    Defines the location and resolution of a rectangular grid of data,
    including it's :py:class:`CRS`.

    :param crs: Coordinate Reference System
    :param affine: Affine transformation defining the location of the geobox
    """

    def __init__(self, width: int, height: int, affine: Affine, crs: MaybeCRS):
        assert is_affine_st(
            affine
        ), "Only axis-aligned geoboxes are currently supported"
        self.width = width
        self.height = height
        self.affine = affine
        self.extent = polygon_from_transform(width, height, affine, crs=crs)

    @classmethod
    def from_geopolygon(
        cls,
        geopolygon: Geometry,
        resolution: Tuple[float, float],
        crs: MaybeCRS = None,
        align: Optional[Tuple[float, float]] = None,
    ) -> "GeoBox":
        """
        :param resolution: (y_resolution, x_resolution)
        :param crs: CRS to use, if different from the geopolygon
        :param align: Align geobox such that point 'align' lies on the pixel boundary.
        """
        align = align or (0.0, 0.0)
        assert (
            0.0 <= align[1] <= abs(resolution[1])
        ), "X align must be in [0, abs(x_resolution)] range"
        assert (
            0.0 <= align[0] <= abs(resolution[0])
        ), "Y align must be in [0, abs(y_resolution)] range"

        if crs is None:
            crs = geopolygon.crs
        else:
            geopolygon = geopolygon.to_crs(crs)

        bounding_box = geopolygon.boundingbox
        offx, width = _align_pix(
            bounding_box.left, bounding_box.right, resolution[1], align[1]
        )
        offy, height = _align_pix(
            bounding_box.bottom, bounding_box.top, resolution[0], align[0]
        )
        affine = Affine.translation(offx, offy) * Affine.scale(
            resolution[1], resolution[0]
        )
        return GeoBox(crs=crs, affine=affine, width=width, height=height)

    def buffered(self, ybuff, xbuff) -> "GeoBox":
        """
        Produce a tile buffered by ybuff, xbuff (in CRS units)
        """
        by, bx = (
            _round_to_res(buf, res) for buf, res in zip((ybuff, xbuff), self.resolution)
        )
        affine = self.affine * Affine.translation(-bx, -by)

        return GeoBox(
            width=self.width + 2 * bx,
            height=self.height + 2 * by,
            affine=affine,
            crs=self.crs,
        )

    def __getitem__(self, roi) -> "GeoBox":
        if isinstance(roi, int):
            roi = (slice(roi, roi + 1), slice(None, None))

        if isinstance(roi, slice):
            roi = (roi, slice(None, None))

        if len(roi) > 2:
            raise ValueError("Expect 2d slice")

        if not all(s.step is None or s.step == 1 for s in roi):
            raise NotImplementedError("scaling not implemented, yet")

        roi = roi_normalise(roi, self.shape)
        ty, tx = (s.start for s in roi)
        h, w = roi_shape(roi)

        affine = self.affine * Affine.translation(tx, ty)

        return GeoBox(width=w, height=h, affine=affine, crs=self.crs)

    def __or__(self, other) -> "GeoBox":
        """A geobox that encompasses both self and other."""
        return geobox_union_conservative([self, other])

    def __and__(self, other) -> "GeoBox":
        """A geobox that is contained in both self and other."""
        return geobox_intersection_conservative([self, other])

    def is_empty(self) -> bool:
        return self.width == 0 or self.height == 0

    def __bool__(self) -> bool:
        return not self.is_empty()

    def __hash__(self):
        return hash((*self.shape, self.crs, self.affine))

    @property
    def transform(self) -> Affine:
        return self.affine

    @property
    def shape(self) -> Tuple[int, int]:
        return self.height, self.width

    @property
    def crs(self) -> Optional[CRS]:
        return self.extent.crs

    @property
    def dimensions(self) -> Tuple[str, str]:
        """
        List of dimension names of the GeoBox
        """
        crs = self.crs
        if crs is None:
            return ("y", "x")
        return crs.dimensions

    @property
    def resolution(self) -> Tuple[float, float]:
        """
        Resolution in Y,X dimensions
        """
        return self.affine.e, self.affine.a

    @property
    def alignment(self) -> Tuple[float, float]:
        """
        Alignment of pixel boundaries in Y,X dimensions
        """
        return self.affine.yoff % abs(self.affine.e), self.affine.xoff % abs(
            self.affine.a
        )

    @property
    def coordinates(self) -> Dict[str, Coordinate]:
        """
        dict of coordinate labels
        """
        yres, xres = self.resolution
        yoff, xoff = self.affine.yoff, self.affine.xoff

        xs = numpy.arange(self.width) * xres + (xoff + xres / 2)
        ys = numpy.arange(self.height) * yres + (yoff + yres / 2)

        units = self.crs.units if self.crs is not None else ("1", "1")

        return OrderedDict(
            (dim, Coordinate(labels, units, res))
            for dim, labels, units, res in zip(
                self.dimensions, (ys, xs), units, (yres, xres)
            )
        )

    def xr_coords(
        self, with_crs: Union[bool, str] = False
    ) -> Dict[Hashable, xr.DataArray]:
        """Dictionary of Coordinates in xarray format

        :param with_crs: If True include netcdf/cf style CRS Coordinate
        with default name 'spatial_ref', if with_crs is a string then treat
        the string as a name of the coordinate.

        Returns
        =======

        OrderedDict name:str -> xr.DataArray

        where names are either `y,x` for projected or `latitude, longitude` for geographic.

        """
        spatial_ref = "spatial_ref"
        if isinstance(with_crs, str):
            spatial_ref = with_crs
            with_crs = True

        attrs = {}
        coords = self.coordinates
        crs = self.crs
        if crs is not None:
            attrs["crs"] = str(crs)

        coords: Dict[Hashable, xr.DataArray] = {
            n: _coord_to_xr(n, c, **attrs) for n, c in coords.items()
        }

        if with_crs and crs is not None:
            coords[spatial_ref] = _mk_crs_coord(crs, spatial_ref)

        return coords

    @property
    def geographic_extent(self) -> Geometry:
        """GeoBox extent in EPSG:4326"""
        if self.crs is None or self.crs.geographic:
            return self.extent
        return self.extent.to_crs(CRS("EPSG:4326"))

    coords = coordinates
    dims = dimensions

    def __str__(self):
        return f"GeoBox({self.geographic_extent})"

    def __repr__(self):
        return f"GeoBox({self.width}, {self.height}, {self.affine!r}, {self.crs})"

    def __eq__(self, other):
        if not isinstance(other, GeoBox):
            return False

        return (
            self.shape == other.shape
            and self.transform == other.transform
            and self.crs == other.crs
        )


def bounding_box_in_pixel_domain(geobox: GeoBox, reference: GeoBox) -> BoundingBox:
    """
    Returns the bounding box of `geobox` with respect to the pixel grid
    defined by `reference` when their coordinate grids are compatible,
    that is, have the same CRS, same pixel size and orientation, and
    are related by whole pixel translation,
    otherwise raises `ValueError`.
    """
    tol = 1.0e-8

    if reference.crs != geobox.crs:
        raise ValueError("Cannot combine geoboxes in different CRSs")

    a, b, c, d, e, f, *_ = ~reference.affine * geobox.affine

    if not (
        numpy.isclose(a, 1)
        and numpy.isclose(b, 0)
        and is_almost_int(c, tol)
        and numpy.isclose(d, 0)
        and numpy.isclose(e, 1)
        and is_almost_int(f, tol)
    ):
        raise ValueError("Incompatible grids")

    tx, ty = round(c), round(f)
    return BoundingBox(tx, ty, tx + geobox.width, ty + geobox.height)


def geobox_union_conservative(geoboxes: List[GeoBox]) -> GeoBox:
    """Union of geoboxes. Fails whenever incompatible grids are encountered."""
    if len(geoboxes) == 0:
        raise ValueError("No geoboxes supplied")

    reference, *_ = geoboxes

    bbox = bbox_union(
        bounding_box_in_pixel_domain(geobox, reference=reference) for geobox in geoboxes
    )

    affine = reference.affine * Affine.translation(*bbox[:2])

    return GeoBox(
        width=bbox.width, height=bbox.height, affine=affine, crs=reference.crs
    )


def geobox_intersection_conservative(geoboxes: List[GeoBox]) -> GeoBox:
    """
    Intersection of geoboxes. Fails whenever incompatible grids are encountered.
    """
    if len(geoboxes) == 0:
        raise ValueError("No geoboxes supplied")

    reference, *_ = geoboxes

    bbox = bbox_intersection(
        bounding_box_in_pixel_domain(geobox, reference=reference) for geobox in geoboxes
    )

    # standardise empty geobox representation
    if bbox.left > bbox.right:
        bbox = BoundingBox(
            left=bbox.left, bottom=bbox.bottom, right=bbox.left, top=bbox.top
        )
    if bbox.bottom > bbox.top:
        bbox = BoundingBox(
            left=bbox.left, bottom=bbox.bottom, right=bbox.right, top=bbox.bottom
        )

    affine = reference.affine * Affine.translation(*bbox[:2])

    return GeoBox(
        width=bbox.width, height=bbox.height, affine=affine, crs=reference.crs
    )


def scaled_down_geobox(src_geobox: GeoBox, scaler: int) -> GeoBox:
    """Given a source geobox and integer scaler compute geobox of a scaled down image.

    Output geobox will be padded when shape is not a multiple of scaler.
    Example: 5x4, scaler=2 -> 3x2

    NOTE: here we assume that pixel coordinates are 0,0 at the top-left
          corner of a top-left pixel.

    """
    assert scaler > 1

    H, W = (X // scaler + (1 if X % scaler else 0) for X in src_geobox.shape)

    # Since 0,0 is at the corner of a pixel, not center, there is no
    # translation between pixel plane coords due to scaling
    A = src_geobox.transform * Affine.scale(scaler, scaler)

    return GeoBox(W, H, A, src_geobox.crs)


def _round_to_res(value: float, res: float) -> int:
    res = abs(res)
    return int(math.ceil((value - 0.1 * res) / res))


def _mk_crs_coord(crs: CRS, name: str = "spatial_ref") -> xr.DataArray:
    # pylint: disable=protected-access

    if crs.projected:
        grid_mapping_name = crs._crs.to_cf().get("grid_mapping_name")
        if grid_mapping_name is None:
            grid_mapping_name = "??"
        grid_mapping_name = grid_mapping_name.lower()
    else:
        grid_mapping_name = "latitude_longitude"

    epsg = 0 if crs.epsg is None else crs.epsg

    return xr.DataArray(
        numpy.asarray(epsg, "int32"),
        name=name,
        dims=(),
        attrs={"spatial_ref": crs.wkt, "grid_mapping_name": grid_mapping_name},
    )


def _coord_to_xr(name: str, c: Coordinate, **attrs) -> xr.DataArray:
    """Construct xr.DataArray from named Coordinate object, this can then be used
    to define coordinates for xr.Dataset|xr.DataArray
    """
    attrs = dict(units=c.units, resolution=c.resolution, **attrs)
    return xr.DataArray(c.values, coords={name: c.values}, dims=(name,), attrs=attrs)


def assign_crs(
    xx: Union[xr.DataArray, xr.Dataset],
    crs: MaybeCRS = None,
    crs_coord_name: str = "spatial_ref",
) -> Union[xr.Dataset, xr.DataArray]:
    """
    Assign CRS for a non-georegistered array or dataset.

    Returns a new object with CRS information populated.

    Can also be called without ``crs`` argument on data that already has CRS
    information but not in the format used by datacube, in this case CRS
    metadata will be restructured into a shape used by datacube. This format
    allows for better propagation of CRS information through various
    computations.

    .. code-block:: python

        xx = odc.geo.assign_crs(xr.open_rasterio("some-file.tif"))
        print(xx.geobox)
        print(xx.astype("float32").geobox)


    :param xx: :py:class:`xarray.Dataset` or :py:class:`~xarray.DataArray`
    :param crs: CRS to assign, if omitted try to guess from attributes
    :param crs_coord_name: how to name crs corodinate (defaults to ``spatial_ref``)
    """
    if crs is None:
        geobox = getattr(xx, "geobox", None)
        if geobox is None:
            raise ValueError("Failed to guess CRS for this object")
        crs = geobox.crs

    crs = norm_crs_or_error(crs)
    crs_coord = _mk_crs_coord(crs, name=crs_coord_name)
    xx = xx.assign_coords({crs_coord.name: crs_coord})

    xx.attrs.update(grid_mapping=crs_coord_name)

    if isinstance(xx, xr.Dataset):
        for band in xx.data_vars.values():
            band.attrs.update(grid_mapping=crs_coord_name)

    return xx
