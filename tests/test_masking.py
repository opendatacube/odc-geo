import numpy as np
from odc.geo.masking import (
    bits_to_bool,
    enum_to_bool,
    scale_and_offset,
    mask_invalid_data,
)


from xarray import DataArray, Dataset

# Top left is cloud, top right is cloud shadow
# Bottom left is both cloud and cloud shadow, bottom right is neither
xx_bits = DataArray(
    [[0b00010000, 0b00001000], [0b00011000, 0b00000000]],
    dims=("y", "x"),
    attrs={"nodata": 0},
)

# Test some values, so 3 is cloud, 9 is cloud shadow
xx_values = DataArray([[3, 9], [3, 0]], dims=("y", "x"), attrs={"nodata": 0})

# Array with at least one zero to test nodata
xx_with_nodata = DataArray([[0, 1], [2, 3]], dims=("y", "x"), attrs={"nodata": 0})


# Test bits_to_bool
def test_bits_to_bool():
    # Test with bits
    mask = bits_to_bool(xx_bits, bits=[4, 3], bitflags=None)
    assert mask.equals(DataArray([[True, True], [True, False]], dims=("y", "x")))

    # Test with bitflags
    mask = bits_to_bool(xx_bits, bits=None, bitflags=0b00011000)
    assert mask.equals(DataArray([[True, True], [True, False]], dims=("y", "x")))

    # Test with invert
    mask = bits_to_bool(xx_bits, bits=[4, 3], bitflags=None, invert=True)
    assert mask.equals(DataArray([[False, False], [False, True]], dims=("y", "x")))

    mask = bits_to_bool(xx_bits, bits=None, bitflags=0b00010000, invert=True)
    assert mask.equals(DataArray([[False, True], [False, True]], dims=("y", "x")))


# Test enum_to_bool
def test_enum_to_bool():
    mask = enum_to_bool(xx_values, values=[3, 9])
    assert mask.equals(DataArray([[True, True], [True, False]], dims=("y", "x")))

    mask = enum_to_bool(xx_values, values=[3, 9], invert=True)
    assert mask.equals(DataArray([[False, False], [False, True]], dims=("y", "x")))


# Test apply_scale_and_offset
def test_scale_and_offset():
    mask = scale_and_offset(xx_values, scale=1.0, offset=0.0)
    assert mask.equals(DataArray([[3, 9], [3, 0]], dims=("y", "x")))

    mask = scale_and_offset(xx_values)
    assert mask.equals(DataArray([[3, 9], [3, 0]], dims=("y", "x")))

    mask = scale_and_offset(xx_values, scale=2.0, offset=1.0)
    assert mask.equals(DataArray([[7, 19], [7, 0]], dims=("y", "x")))


# Test mask_invalid
def test_mask_invalid_data():
    mask = mask_invalid_data(xx_with_nodata)
    assert mask.equals(DataArray([[np.nan, 1.0], [2.0, 3.0]], dims=("y", "x")))

    mask = mask_invalid_data(xx_with_nodata, nodata=1)
    assert mask.equals(DataArray([[0, np.nan], [2, 3]], dims=("y", "x")))


# Test landsat masking
def test_mask_landsat():
    xx = Dataset(
        {"pixel_qa": xx_bits, "red": scale_and_offset(xx_with_nodata, offset=20000)}
    )
    print(xx)

    xx = xx.odc.mask_ls()

    assert xx["red"].equals(
        DataArray([[np.nan, np.nan], [np.nan, 0.3500825]], dims=("y", "x"))
    )


def test_mask_sentinel2():
    xx = Dataset(
        {"scl": xx_values, "red": scale_and_offset(xx_with_nodata, offset=8000)}
    )

    xx = xx.odc.mask_s2()

    assert xx["red"].equals(
        DataArray([[np.nan, np.nan], [np.nan, 0.7003]], dims=("y", "x"))
    )

    assert xx.red.odc.nodata is not None
