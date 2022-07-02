from pathlib import Path

import pytest
import xarray as xr

from odc.geo.data import ocean_geom
from odc.geo.geobox import GeoBox
from odc.geo.xr import rasterize


@pytest.fixture(scope="session")
def data_dir():
    return Path(__file__).parent.joinpath("data")


@pytest.fixture()
def ocean_raster() -> xr.DataArray:
    gbox = GeoBox.from_bbox(bbox=(-180, -90, 180, 90), shape=(128, 256))
    return rasterize(ocean_geom(), gbox)


@pytest.fixture()
def ocean_raster_ds(ocean_raster: xr.DataArray) -> xr.Dataset:
    xx = ocean_raster.astype("int16") * 3_000
    xx.attrs["nodata"] = -1

    return xr.Dataset(
        dict(
            red=xx,
            green=xx,
            blue=xx,
        )
    )
