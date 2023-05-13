import json
import lzma
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from warnings import catch_warnings, filterwarnings

from ..crs import MaybeCRS, norm_crs
from ..geom import Geometry, box, multigeom


def data_path(fname: Optional[str] = None) -> Path:
    """Location of data folder or file within."""
    prefix = Path(__file__).parent
    if fname is None:
        return prefix
    return prefix / fname


@lru_cache
def ocean_geojson() -> Dict[str, Any]:
    with lzma.open(data_path("ocean.geojson.xz"), "rt") as xz:
        return json.load(xz)


def ocean_geom(
    crs: MaybeCRS = None, bbox: Optional[Tuple[float, float, float, float]] = None
) -> Geometry:
    """Return world oceans geometry."""
    gjson = ocean_geojson()
    gg = multigeom([Geometry(f["geometry"], "epsg:4326") for f in gjson["features"]])
    if bbox is not None:
        with catch_warnings():
            filterwarnings("ignore")
            gg = gg & box(*bbox, "epsg:4326")
    crs = norm_crs(crs)
    if crs is not None:
        gg = gg.to_crs(crs)

    return gg


@lru_cache
def gbox_css() -> str:
    with open(data_path("gbox.css"), "rt", encoding="utf8") as src:
        return src.read()


def country_geom(iso3: str, crs: MaybeCRS = None) -> Geometry:
    """
    Extract geometry for a country from geopandas sample data.
    """
    # pylint: disable=import-outside-toplevel
    import geopandas as gpd

    from ..converters import from_geopandas

    with catch_warnings():
        filterwarnings("ignore", category=FutureWarning)
        df = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    (gg,) = from_geopandas(df[df.iso_a3 == iso3])
    crs = norm_crs(crs)
    if crs is not None:
        gg = gg.to_crs(crs)
    return gg
