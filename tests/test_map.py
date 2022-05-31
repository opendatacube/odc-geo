import numpy as np
import pytest
import xarray as xr

from odc.geo._interop import have
from odc.geo.xr import ODCExtensionDa

cmap = np.asarray(
    [
        [153, 153, 102, 255],
        [51, 102, 153, 255],
    ],
    dtype="uint8",
)


def test_add_to(ocean_raster: xr.DataArray):
    # Check map safe range first
    xx = ocean_raster.copy()[20:-20]
    assert isinstance(xx.odc, ODCExtensionDa)

    url, bounds = xx.odc.add_to(None, cmap=cmap, max_size=32, zlevel=1)
    assert isinstance(url, str)
    assert url.startswith("data:")
    assert "image/png" in url
    assert isinstance(bounds, tuple)
    assert isinstance(bounds[0], tuple)
    assert isinstance(bounds[0], tuple)

    url, bounds = xx.odc.add_to(None, cmap=cmap, max_size=32, fmt="jpeg", quality=50)
    assert isinstance(url, str)
    assert url.startswith("data:")
    assert "image/jpeg" in url
    assert isinstance(bounds, tuple)
    assert isinstance(bounds[0], tuple)
    assert isinstance(bounds[0], tuple)

    for not_a_map in [xx.data, {}, []]:
        with pytest.raises(ValueError):
            _ = xx.odc.add_to(not_a_map)


@pytest.mark.skipif(have.folium is False, reason="No folium installed")
def test_add_to_folium(ocean_raster: xr.DataArray):
    import folium

    xx = ocean_raster.copy()[20:-20]
    assert isinstance(xx.odc, ODCExtensionDa)

    m = folium.Map()
    img_overlay = xx.odc.add_to(m, cmap=cmap, max_size=32, zlevel=1)
    assert img_overlay is not None


@pytest.mark.skipif(have.ipyleaflet is False, reason="No ipyleaflet installed")
def test_add_to_ipyleaflet(ocean_raster: xr.DataArray):
    import ipyleaflet

    xx = ocean_raster.copy()[20:-20]
    assert isinstance(xx.odc, ODCExtensionDa)

    m = ipyleaflet.Map()
    img_overlay = xx.odc.add_to(m, cmap=cmap, max_size=32, name="xx")
    assert img_overlay is not None
