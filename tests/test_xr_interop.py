# This file is part of the Open Data Cube, see https://opendatacube.org for more information
#
# Copyright (c) 2015-2020 ODC Contributors
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import MagicMock


from odc.geo._xr_interop import _mk_crs_coord, xr_coords
from odc.geo.geobox import (
    GeoBox,
)
from odc.geo.testutils.geom import (
    epsg3577,
    mkA,
)


def test_geobox_xr_coords():
    A = mkA(0, scale=(10, -10), translation=(-48800, -2983006))

    w, h = 512, 256
    gbox = GeoBox(w, h, A, epsg3577)

    cc = xr_coords(gbox, with_crs=False)
    assert list(cc) == ["y", "x"]
    assert cc["y"].shape == (gbox.shape[0],)
    assert cc["x"].shape == (gbox.shape[1],)
    assert "crs" in cc["y"].attrs
    assert "crs" in cc["x"].attrs

    cc = xr_coords(gbox, with_crs=True)
    assert list(cc) == ["y", "x", "spatial_ref"]
    assert cc["spatial_ref"].shape == ()
    assert cc["spatial_ref"].attrs["spatial_ref"] == gbox.crs.wkt
    assert isinstance(cc["spatial_ref"].attrs["grid_mapping_name"], str)

    # with_crs should be default True
    assert list(xr_coords(gbox)) == list(xr_coords(gbox, with_crs=True))

    cc = xr_coords(gbox, with_crs="Albers")
    assert list(cc) == ["y", "x", "Albers"]

    # geographic CRS
    A = mkA(0, scale=(0.1, -0.1), translation=(10, 30))
    gbox = GeoBox(w, h, A, "epsg:4326")
    cc = xr_coords(gbox, with_crs=True)
    assert list(cc) == ["latitude", "longitude", "spatial_ref"]
    assert cc["spatial_ref"].shape == ()
    assert cc["spatial_ref"].attrs["spatial_ref"] == gbox.crs.wkt
    assert isinstance(cc["spatial_ref"].attrs["grid_mapping_name"], str)

    # missing CRS for GeoBox
    gbox = GeoBox(w, h, A, None)
    cc = xr_coords(gbox, with_crs=True)
    assert list(cc) == ["y", "x"]

    # check CRS without name
    crs = MagicMock()
    crs.projected = True
    crs.wkt = epsg3577.wkt
    crs.epsg = epsg3577.epsg
    crs._crs = MagicMock()
    crs._crs.to_cf.return_value = {}
    assert _mk_crs_coord(crs).attrs["grid_mapping_name"] == "??"
