"""
Interop with other geometry libraries.
"""

from typing import List, Tuple

from .crs import CRS, MaybeCRS, norm_crs
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

    :returns: ``[pixel coors], [world coords]``
    """
    pix, wld, gcp_crs = extract_gcps_raw(src)
    output_crs = norm_crs(output_crs)

    wld_pts = (point(pt.x, pt.y, gcp_crs) for pt in wld)
    if output_crs is not None and gcp_crs != output_crs:
        wld_pts = (pt.to_crs(output_crs) for pt in wld_pts)

    return pix, list(wld_pts)
