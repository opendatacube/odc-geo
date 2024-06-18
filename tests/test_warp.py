import numpy as np
import numpy.testing as npt
import pytest
import xarray as xr
from affine import Affine

from odc.geo import MaybeCRS
from odc.geo.warp import resampling_s2rio, rio_reproject, rio_warp_affine

NaN = float("nan")


@pytest.mark.parametrize(
    "iso3, crs, resolution",
    [
        ("AUS", "epsg:4326", 0.1),
        ("AUS", "epsg:3577", 10_000),
        ("AUS", "epsg:3857", 10_000),
        ("NZL", "epsg:3857", 5_000),
    ],
)
@pytest.mark.parametrize("resampling", ["nearest", "bilinear", "average", "sum"])
def test_warp_nan(country_raster_f32: xr.DataArray, crs: MaybeCRS, resampling: str):
    xx = country_raster_f32
    assert isinstance(xx, xr.DataArray)
    assert xx.odc.crs == crs
    assert xx.odc.nodata is None
    assert xx.dtype == "float32"

    mid = xx.shape[0] // 2
    xx.data[mid, :] = NaN
    xx.data[:, -10] = NaN

    assert resampling_s2rio(resampling) is not None
    assert np.isnan(xx.data).sum() > 0

    src_gbox = xx.odc.geobox
    dst_gbox = src_gbox.zoom_to(shape=100).pad(10)

    yy1 = np.full(dst_gbox.shape, -333, dtype=xx.dtype)
    yy2 = np.full(dst_gbox.shape, -333, dtype=xx.dtype)

    assert rio_reproject(xx.data, yy1, src_gbox, dst_gbox, resampling=resampling) is yy1
    assert (
        rio_reproject(
            xx.data,
            yy2,
            src_gbox,
            dst_gbox,
            resampling=resampling,
            src_nodata=NaN,
            dst_nodata=NaN,
        )
        is yy2
    )

    npt.assert_array_equal(yy1, yy2)

    # make sure all pixels were replaced
    assert (yy1 == -333).sum() == 0

    # expect to see NaNs in the output
    assert np.isnan(yy2).sum() > 0

    A = Affine.identity()
    xx_ = xx.data.copy() * 0
    assert rio_warp_affine(xx.data, xx_, A, resampling) is xx_
    npt.assert_array_equal(xx.data, xx_)
