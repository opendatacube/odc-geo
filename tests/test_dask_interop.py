# pylint: disable=wrong-import-position
import pytest

pytest.importorskip("dask")

from dask.base import tokenize

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

    crs = CRS("epsg:4326")
    assert tokenize(crs) == tokenize(crs)
    assert tokenize(crs) == tokenize(CRS(crs))
    assert tokenize(CRS("epsg:4326")) == tokenize(CRS("EPSG:4326"))
    assert tokenize(CRS("epsg:4326")) != tokenize(CRS("EPSG:3857"))
