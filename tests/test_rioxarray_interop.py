from pathlib import Path

import pytest
from affine import Affine

from odc.geo.gcp import GCPGeoBox
from odc.geo.geobox import GeoBox

rioxarray = pytest.importorskip("rioxarray")


@pytest.mark.parametrize("fname", ["au-gcp.tif", "au-3577.tif", "au-3577-rotated.tif"])
@pytest.mark.parametrize("parse_coordinates", [True, False])
def test_rioxarray_interop(data_dir: Path, fname: str, parse_coordinates: bool):
    xx = rioxarray.open_rasterio(
        data_dir / fname, chunks=64, parse_coordinates=parse_coordinates
    )
    assert xx.rio.crs is not None
    assert xx.rio.crs == xx.odc.crs

    if "-gcp" in fname:
        assert isinstance(xx.odc.geobox, GCPGeoBox)
    else:
        assert isinstance(xx.rio.transform(), Affine)
        assert xx.rio.transform() == xx.odc.transform
        assert isinstance(xx.odc.geobox, GeoBox)
        assert xx[0, 1:2, -1:].odc.geobox is not None
