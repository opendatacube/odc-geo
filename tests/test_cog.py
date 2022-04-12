from odc.geo.gridspec import GridSpec
from odc.geo.xr import xr_zeros


def test_write_cog():
    gs = GridSpec.web_tiles(2)
    gbox = gs[2, 1]

    img = xr_zeros(gbox, dtype="uint16")
    assert img.odc.geobox == gbox

    img_bytes = img.odc.to_cog(blocksize=32)
    assert isinstance(img_bytes, bytes)

    img_bytes2 = img.odc.write_cog(":mem:", blocksize=32)
    assert len(img_bytes) == len(img_bytes2)

    img_bytes = img.odc.to_cog(blocksize=32, overview_levels=[2])
    assert isinstance(img_bytes, bytes)

    img_bytes2 = img.odc.write_cog(
        ":mem:", blocksize=32, overview_levels=[2], use_windowed_writes=True
    )
    assert len(img_bytes) == len(img_bytes2)


def test_write_cog_ovr():
    gs = GridSpec.web_tiles(2)
    gbox = gs[2, 1]

    img = xr_zeros(gbox, dtype="uint16")
    assert img.odc.geobox == gbox
    ovrs = [img[::2, ::2], img[::4, ::4]]

    img_bytes = img.odc.to_cog(blocksize=32, overviews=ovrs)
    assert isinstance(img_bytes, bytes)

    img_bytes2 = img.odc.write_cog(":mem:", blocksize=32, overviews=ovrs)
    assert len(img_bytes) == len(img_bytes2)
