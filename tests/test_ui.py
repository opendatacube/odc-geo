from odc.geo.geobox import GeoBox
from odc.geo.ui import svg_base_map


def test_svg_base_smoke_only():
    gbox = GeoBox.from_bbox((110, -45, 160, -10), resolution=0.05)
    assert "<svg" in svg_base_map()

    assert "</svg>" in svg_base_map(gbox, gbox.extent, "<path/>")

    # long
    assert "</svg>" in svg_base_map("<path/>", bbox=(-10, 0, 10, 1))

    # tall
    assert "</svg>" in svg_base_map("<path/>", bbox=(0, 0, 1, 10))

    # cross-hair
    assert "</svg>" in svg_base_map(target=(0, 10))
