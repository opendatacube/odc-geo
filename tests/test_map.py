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


@pytest.mark.skipif(have.folium is False, reason="No folium installed")
def test_explore(ocean_raster: xr.DataArray, ocean_raster_ds: xr.Dataset):
    import folium
    from folium.raster_layers import ImageOverlay

    # Test explore on dataset input and verify that output is a folium map
    # that contains an ImageOverlay layer
    m = ocean_raster_ds.odc.explore()
    assert isinstance(m, folium.folium.Map)
    assert any(isinstance(child, ImageOverlay) for child in m._children.values())

    # Verify that explore fails if bands cannot be guessed
    with pytest.raises(ValueError):
        ocean_raster_ds.rename({"red": "band1"}).odc.explore()

    # Verify that error can be avoided using `bands` param
    ocean_raster_ds.rename({"red": "band1"}).odc.explore(
        bands=("band1", "green", "blue")
    )

    # Test explore on data array input and verify that output is a folium
    # map that contains an ImageOverlay layer
    m = ocean_raster.odc.explore()
    assert isinstance(m, folium.folium.Map)
    assert any(isinstance(child, ImageOverlay) for child in m._children.values())

    # Verify that error is raised if a timeseries is passed
    with pytest.raises(ValueError):
        xr.concat([ocean_raster, ocean_raster], dim="time").odc.explore()
