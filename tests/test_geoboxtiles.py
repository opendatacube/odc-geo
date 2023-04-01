import numpy as np
import pytest

from odc.geo import MaybeCRS, geom
from odc.geo.geobox import GeoBox, GeoboxTiles


@pytest.mark.parametrize(
    "iso3, crs, resolution",
    [
        ("AUS", "epsg:4326", 0.1),
        ("AUS", "epsg:3577", 10_000),
        ("AUS", "epsg:3857", 10_000),
        ("NZL", "epsg:3857", 1_000),
    ],
)
def test_geoboxtiles_intersect(
    country: geom.Geometry, resolution: float, crs: MaybeCRS
):
    assert isinstance(country, geom.Geometry)
    assert country.crs == crs

    geobox = GeoBox.from_geopolygon(country, resolution=resolution, tight=True)
    assert geobox.crs == crs

    gbt = GeoboxTiles(geobox, (11, 13))
    assert gbt.base is geobox
    mm = gbt.grid_intersect(gbt)

    assert len(gbt.chunks) == 2
    assert (sum(gbt.chunks[0]), sum(gbt.chunks[1])) == gbt.base.shape

    for iy, ix in np.ndindex(gbt.shape.yx):
        idx = (iy, ix)
        assert idx in mm
        assert len(mm[idx]) >= 1
        assert idx in mm[idx]
