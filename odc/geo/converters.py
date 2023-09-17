"""
Interop with other geometry libraries.
"""

import re
from io import BytesIO
from typing import Any, Dict, List, Tuple, Union

from ._interop import have
from .crs import CRS, MaybeCRS, Optional, norm_crs
from .gcp import GCPGeoBox
from .geobox import GeoBox
from .geom import Geometry, point
from .types import XY, MaybeNodata, xy_

GEOTIFF_TAGS = {
    34264,  # ModelTransformation
    34735,  # GeoKeyDirectory
    34736,  # GeoDoubleParams
    34737,  # GeoAsciiParams
    33550,  # ModelPixelScale
    33922,  # ModelTiePoint
    #
    42112,  # GDAL_METADATA
    42113,  # GDAL_NODATA
    #
    # probably never used in the wild
    33920,  # IrasB Transformation Matrix
    50844,  # RPCCoefficientTag
}


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


def geotiff_metadata(
    geobox: GeoBox,
    nodata: MaybeNodata = None,
    gdal_metadata: Optional[str] = None,
) -> Tuple[List[Tuple[int, int, int, Any]], Dict[str, Any]]:
    """
    Convert GeoBox to geotiff tags and metadata for :py:mod:`tifffile`.

    .. note::

       Requires :py:mod:`rasterio`, :py:mod:`tifffile` and :py:mod:`xarray`.


    :returns:
       List of TIFF tag tuples suitable for passing to :py:mod:`tifffile` as
       ``extratags=``, and dictionary representation of GEOTIFF tags.

    """
    # pylint: disable=import-outside-toplevel

    if not (have.tifffile and have.rasterio):
        raise RuntimeError(
            "Please install `tifffile` and `rasterio` to use this method"
        )

    from tifffile import TiffFile

    from ._cog import to_cog
    from .xr import xr_zeros

    buf = to_cog(
        xr_zeros(geobox[:2, :2]), nodata=nodata, compress=None, overview_levels=[]
    )
    tf = TiffFile(BytesIO(buf), mode="r")
    assert tf.geotiff_metadata is not None
    geo_tags: List[Tuple[int, int, int, Any]] = [
        (t.code, t.dtype.value, t.count, t.value)
        for t in tf.pages.first.tags.values()
        if t.code in GEOTIFF_TAGS
    ]

    if gdal_metadata is not None:
        geo_tags.append((42112, 2, len(gdal_metadata) + 1, gdal_metadata))

    return geo_tags, tf.geotiff_metadata
