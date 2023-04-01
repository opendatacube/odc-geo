# pylint: disable=wrong-import-position
import pytest

pytest.importorskip("dask")

import numpy as np
from dask.base import tokenize

from odc.geo.gcp import GCPGeoBox, GCPMapping
from odc.geo.geobox import CRS, GeoBox, GeoboxTiles


def test_tokenize_geobox():
    gbox = GeoBox.from_bbox([0, 0, 1, 1], shape=(100, 100))
    tk = tokenize(gbox)
    print(tk)
    assert tokenize(gbox) == tokenize(gbox)
    assert tokenize(gbox) != tokenize(gbox.pad(1))

    gbt = GeoboxTiles(gbox, (1, 3))
    assert tokenize(gbt) == tokenize(gbt)
    assert tokenize(GeoboxTiles(gbox, (1, 2))) != tokenize(GeoboxTiles(gbox, (2, 1)))
    assert tokenize(GeoboxTiles(gbox, (1, 2))) != tokenize(
        GeoboxTiles(gbox.pad(1), (1, 2))
    )

    gbt = GeoboxTiles(gbox, ((30, 40, 30), (45, 55)))
    assert tokenize(gbt) == tokenize(GeoboxTiles(gbox, gbt.chunks))
    assert tokenize(gbt) != tokenize(GeoboxTiles(gbox, gbt.chunks[::-1]))

    crs = CRS("epsg:4326")
    assert tokenize(crs) == tokenize(crs)
    assert tokenize(crs) == tokenize(CRS(crs))
    assert tokenize(CRS("epsg:4326")) == tokenize(CRS("EPSG:4326"))
    assert tokenize(CRS("epsg:4326")) != tokenize(CRS("EPSG:3857"))


def test_tokenize_gcpgeobox():
    pts = np.vstack([c.ravel() for c in np.meshgrid([0, 2, 3], [0, 1, 2, 5])]).T
    assert pts.shape[0] > 9
    assert pts.shape[1] == 2

    mapping = GCPMapping(pts, pts + 10, "epsg:3857")
    mapping2 = GCPMapping(pts, pts + 10, "epsg:4326")
    mapping_ = GCPMapping(mapping._pix, mapping._wld, mapping.crs)

    assert tokenize(mapping) == tokenize(mapping_)

    gbox = GCPGeoBox((10, 20), mapping)
    gbox_ = GCPGeoBox((10, 20), mapping_)

    gbox2 = GCPGeoBox((10, 20), mapping2)
    assert tokenize(gbox) == tokenize(gbox_)
    assert tokenize(gbox[:3, :2]) == tokenize(gbox[0:3, 0:2])

    assert tokenize(gbox) != tokenize(gbox2)
