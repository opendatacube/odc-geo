import numpy as np
import pytest
import xarray as xr

from odc.geo._interop import is_dask_collection
from odc.geo._rgba import is_rgb
from odc.geo.testutils import daskify
from odc.geo.xr import ODCExtensionDa, ODCExtensionDs


def test_colorize(ocean_raster: xr.DataArray):
    xx = ocean_raster

    assert isinstance(xx.odc, ODCExtensionDa)
    cmap = np.asarray(
        [
            [153, 153, 102, 255],
            [51, 102, 153, 255],
        ],
        dtype="uint8",
    )

    cc = xx.odc.colorize(cmap)
    assert isinstance(cc.odc, ODCExtensionDa)
    assert cc.odc.geobox == xx.odc.geobox
    assert cc.dtype == cmap.dtype
    assert cc.shape == (*xx.shape, 4)

    cc = xx.astype("int16").odc.colorize(cmap, clip=True)
    assert isinstance(cc.odc, ODCExtensionDa)
    assert cc.odc.geobox == xx.odc.geobox
    assert cc.dtype == cmap.dtype
    assert cc.shape == (*xx.shape, 4)

    _xx = daskify(xx, chunks=(16, 32))
    assert is_dask_collection(_xx) is True

    _cc = _xx.odc.colorize(cmap, clip=True)
    assert is_dask_collection(_cc) is True

    assert isinstance(_cc.odc, ODCExtensionDa)
    assert _cc.odc.geobox == _xx.odc.geobox
    assert _cc.dtype == cmap.dtype
    assert _cc.shape == (*_xx.shape, 4)

    assert bool((_cc.compute() == cc).all()) is True


def test_rgba(ocean_raster_ds: xr.Dataset):
    xx = ocean_raster_ds
    assert xx.red.dtype == "int16"
    assert isinstance(xx.odc, ODCExtensionDs)
    assert xx.red.nodata == -1

    cc = xx.odc.to_rgba((0, 3_000))
    assert isinstance(cc.odc, ODCExtensionDa)
    assert cc.odc.geobox == xx.odc.geobox
    assert cc.dtype == "uint8"
    assert cc.shape == (*xx.red.shape, 4)
    assert is_rgb(cc) is True
    assert is_rgb(xx.red) is False

    cc = xx.astype("float32").odc.to_rgba()
    assert isinstance(cc.odc, ODCExtensionDa)
    assert cc.odc.geobox == xx.odc.geobox
    assert cc.dtype == "uint8"
    assert cc.shape == (*xx.red.shape, 4)

    _xx = xx.rename(dict(red="b1", green="b2", blue="b3"))
    for dv in _xx.data_vars.values():
        dv.attrs.pop("nodata", None)

    cc = _xx.odc.to_rgba(3_000, bands=["b1", "b2", "b3"])
    assert isinstance(cc.odc, ODCExtensionDa)
    assert cc.odc.geobox == xx.odc.geobox
    assert cc.dtype == "uint8"
    assert cc.shape == (*xx.red.shape, 4)
    assert is_rgb(cc[..., 0]) is False
    assert is_rgb(cc[..., :2]) is False
    assert is_rgb(cc[..., :3]) is True

    # can't guess band names
    with pytest.raises(ValueError):
        _ = _xx.odc.to_rgba(3_000)

    # too many red band candidates
    _xx = xx.rename({"red": "red1"})
    _xx["red2"] = xx.red
    with pytest.raises(ValueError):
        _ = _xx.odc.to_rgba(3_000)

    _xx = daskify(xx, chunks=(51, 39))
    assert is_dask_collection(_xx) is True
    _cc = _xx.odc.to_rgba(clamp=3000)
    assert isinstance(_cc.odc, ODCExtensionDa)
    assert is_dask_collection(_cc) is True
    assert _cc.odc.geobox == xx.odc.geobox
    assert _cc.dtype == "uint8"
    assert _cc.shape == (*xx.red.shape, 4)
    assert bool((cc == _cc.compute()).all()) is True

    # missing clamp for dask inputs
    with pytest.raises(ValueError):
        _ = _xx.odc.to_rgba()

    # smoke-test compression
    assert isinstance(cc.odc.compress("jpeg", 60, transparent=(0, 0, 0)), bytes)
    assert isinstance(_cc.odc.compress("png", 9), bytes)
    assert isinstance(cc.odc.compress(as_data_url=True), str)

    assert isinstance(xx.red.astype("uint8").odc.compress(), bytes)

    with pytest.raises(ValueError):
        _ = xx.red.odc.compress("no-such-format")

    with pytest.raises(ValueError):
        # expect 2d/3d inputs
        _ = xx.red[0].odc.compress()
