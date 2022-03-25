import json
import lzma
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

from ..geom import Geometry, multigeom


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


def ocean_geom() -> Geometry:
    """Return world oceans geometry."""
    gjson = ocean_geojson()
    return multigeom([Geometry(f["geometry"], "epsg:4326") for f in gjson["features"]])


@lru_cache
def gbox_css() -> str:
    with open(data_path("gbox.css"), "rt", encoding="utf8") as src:
        return src.read()
