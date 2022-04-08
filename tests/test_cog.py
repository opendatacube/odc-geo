import odc.geo.xr  # pylint: disable=unused-import
from odc.geo.gridspec import GridSpec
from odc.geo.testutils import xr_zeros


def test_write_cog():
    gs = GridSpec.web_tiles(2)
    gbox = gs[2, 1]

    img = xr_zeros(gbox)
    assert img.odc.geobox == gbox

    img_bytes = img.odc.to_cog(blocksize=32)
    assert isinstance(img_bytes, bytes)

    img_bytes2 = img.odc.write_cog(":mem:", blocksize=32)
    assert len(img_bytes) == len(img_bytes2)
