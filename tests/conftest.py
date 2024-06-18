from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from odc.geo.data import country_geom, ocean_geom
from odc.geo.geobox import GeoBox
from odc.geo.xr import rasterize

# pylint: disable=protected-access,import-outside-toplevel,redefined-outer-name


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


@pytest.fixture()
def iso3():
    return "AUS"


@pytest.fixture()
def crs():
    return "epsg:3857"


@pytest.fixture()
def country(iso3, crs):
    yield country_geom(iso3, crs=crs)


@pytest.fixture()
def country_raster(country, resolution):
    geobox = GeoBox.from_geopolygon(country, resolution=resolution, tight=True)
    yield rasterize(country, geobox)


@pytest.fixture()
def country_raster_f32(country, resolution):
    geobox = GeoBox.from_geopolygon(country, resolution=resolution, tight=True)
    xx = rasterize(country, geobox)
    xx = xr.where(xx, np.random.uniform(0, 100, xx.shape).astype("float32"), 0)
    yield xx
