"""
Interop with other geometry libraries.
"""

import re
from typing import Any, List, Tuple, Union

from .crs import CRS, MaybeCRS, Optional, norm_crs
from .gcp import GCPGeoBox
from .geobox import GeoBox
from .geom import Geometry, point
from .types import XY, xy_


def from_geopandas(series) -> List[Geometry]:
    """
    Convert Geopandas data into list of :py:class:`~odc.geo.geom.Geometry`.
    """
    crs = getattr(series, "crs", None)
    gg = getattr(series, "geometry", None)

    if crs is None or gg is None:
        return []

    return [Geometry(g, crs) for g in gg]


def extract_gcps_raw(src) -> Tuple[List[XY[float]], List[XY[float]], CRS]:
    """
    Extract Ground Control points from :py:class:`rasterio.DatasetReader`.

    :returns: ``[pixel coors], [world coords], CRS``
    """
    pts, gcp_crs = src.gcps
    if len(pts) == 0:
        raise ValueError(f"Could not access GCP on {src}")

    pix_pts = [xy_(pt.col, pt.row) for pt in pts]
    wld_pts = [xy_(pt.x, pt.y) for pt in pts]

    return pix_pts, wld_pts, norm_crs(gcp_crs)


def extract_gcps(
    src,
    output_crs: MaybeCRS = None,
) -> Tuple[List[XY[float]], List[Geometry]]:
    """
    Extract Ground Control points from :py:class:`rasterio.DatasetReader`.

    :returns: ``[pixel coords], [world coords]``
    """
    pix, wld, gcp_crs = extract_gcps_raw(src)
    output_crs = norm_crs(output_crs)

    wld_pts = (point(pt.x, pt.y, gcp_crs) for pt in wld)
    if output_crs is not None and gcp_crs != output_crs:
        wld_pts = (pt.to_crs(output_crs) for pt in wld_pts)

    return pix, list(wld_pts)


def map_crs(m: Any, /) -> Optional[CRS]:
    def _from_name(srs: str) -> Optional[CRS]:
        if match := re.match(r"epsg:?(?P<code>\d+)", srs.lower()):
            return CRS(f"epsg:{match['code']}")
        return None

    _crs = getattr(m, "crs", None)
    if isinstance(_crs, str):
        # probably folium map
        return _from_name(_crs)

    if isinstance(_crs, dict):
        # ipylealflet uses dict
        if (name := _crs.get("name", None)) is not None:
            if (crs := _from_name(name)) is not None:
                return crs
        if (proj4def := _crs.get("proj4def", None)) is not None:
            return CRS(proj4def)

    return None


def rio_geobox(rdr: Any) -> Union[GeoBox, GCPGeoBox]:
    """
    Construct GeoBox from rasterio.

    :param rdr: Opened :py:class:`rasterio.DatasetReader`
    :returns:
       :py:class:`~odc.geo.geobox.GeoBox` or :py:class:`~odc.geo.gcp.GCPGeoBox`.
    """
    pts, _ = rdr.gcps
    if len(pts) > 0:
        return GCPGeoBox.from_rio(rdr)

    return GeoBox.from_rio(rdr)
