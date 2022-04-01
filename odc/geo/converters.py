"""
Interop with other geometry libraries.
"""

from typing import List

from .geom import Geometry


def from_geopandas(series) -> List[Geometry]:
    """
    Convert Geopandas data into list of :py:class:`~odc.geo.geom.Geometry`.
    """
    crs = getattr(series, "crs", None)
    gg = getattr(series, "geometry", None)

    if crs is None or gg is None:
        return []

    return [Geometry(g, crs) for g in gg]
