# This file is part of the Open Data Cube, see https://opendatacube.org for more information
#
# Copyright (c) 2015-2020 ODC Contributors
# SPDX-License-Identifier: Apache-2.0
"""
Functions around supporting cloud masking.
"""

from typing import Annotated, Any, Callable, Sequence
import numpy as np
from xarray import DataArray, Dataset

from enum import Enum


class SENTINEL2_L2A_SCL(Enum):
    """
    Sentinel-2 Scene Classification Layer (SCL) values.
    """

    NO_DATA = 0
    SATURATED_OR_DEFECTIVE = 1
    DARK_AREA_PIXELS = 2
    CLOUD_SHADOWS = 3
    VEGETATION = 4
    NOT_VEGETATED = 5
    WATER = 6
    UNCLASSIFIED = 7
    CLOUD_MEDIUM_PROBABILITY = 8
    CLOUD_HIGH_PROBABILITY = 9
    THIN_CIRRUS = 10
    SNOW = 11


SENTINEL2_L2A_SCALE = 0.0001
SENTINEL2_L2A_OFFSET = -0.1


class LANDSAT_C2L2_PIXEL_QA(Enum):
    """
    Landsat Collection 2 Surface Reflectance Pixel Quality values.
    """

    NO_DATA = 0
    DILATED_CLOUD = 1
    CIRRUS = 2
    CLOUD = 3
    CLOUD_SHADOW = 4
    SNOW = 5
    CLEAR = 6
    WATER = 7
    # Not sure how to implement these yet...
    # CLOUD_CONFIDENCE = [8, 9]
    # CLOUD_SHADOW_CONFIDENCE = [10, 11]
    # SNOW_ICE_CONFIDENCE = [12, 13]
    # CIRRUS_CONFIDENCE = [14, 15]


LANDSAT_C2L2_SCALE = 0.0000275
LANDSAT_C2L2_OFFSET = -0.2

# TODO: QA_RADSAT and QA_AEROSOL for Landsat Collection 2 Surface Reflectance


def bits_to_bool(
    xx: DataArray,
    bits: Sequence[int] | None = None,
    bitflags: int | None = None,
    invert: bool = False,
) -> DataArray:
    """
    Convert integer array into boolean array using bitmasks.

    :param xx: DataArray with integer values
    :param bits: List of bit positions to convert to a bitflag mask (e.g. [0, 1, 2] -> 0b111)
    :param bitflags: Integer value with bits set that will be used to extract the boolean mask (e.g. 0b00011000)
    :param invert: Invert the mask
    :return: DataArray with boolean values
    """
    assert not (
        bits is None and bitflags is None
    ), "Either bits or bitflags must be provided"
    assert not (
        bits is not None and bitflags is not None
    ), "Only one of bits or bitflags can be provided"

    if bitflags is None:
        bitflags = 0

    if bits is not None:
        for b in bits:
            bitflags |= 1 << b

    mask = (xx & bitflags) != 0

    if invert:
        mask = ~mask

    return mask


def enum_to_bool(
    xx: DataArray, values: Sequence[Any], invert: bool = False
) -> DataArray:
    """
    Convert array into boolean array using a list of invalid values.

    :param xx: DataArray with integer values
    :param values: List of valid values to convert to a boolean mask
    :param invert: Invert the mask
    :return: DataArray with boolean values
    """

    mask = xx.isin(values)

    if invert:
        mask = ~mask

    return mask


def scale_and_offset(
    xx: DataArray | Dataset,
    scale: float | None = None,
    offset: float | None = None,
    clip: Annotated[Sequence[int | float], 2] | None = None,
) -> DataArray | Dataset:
    """
    Apply scale and offset to the DataArray. Leave scale and offset blank to use
    the values from the DataArray's attrs.

    :param xx: DataArray with integer values
    :param scale: Scale factor
    :param offset: Offset
    :return: DataArray with scaled and offset values
    """

    # For the Dataset case, we do this recursively for all variables.
    if isinstance(xx, Dataset):
        for var in xx.data_vars:
            xx[var] = scale_and_offset(xx[var], scale, offset, clip=clip)

        return xx

    # "Scales" and "offsets" is used by GDAL.
    if scale is None:
        scale = xx.attrs.get("scales")

    if offset is None:
        offset = xx.attrs.get("offsets")

    # Catch the case where one is provided and not the other...
    if scale is None and offset is not None:
        scale = 1.0

    if offset is None and scale is not None:
        offset = 0.0

    # Store the nodata values to apply to the result
    nodata = xx.odc.nodata

    # Stash the attributes
    attrs = dict(xx.attrs.items())

    if nodata is not None:
        nodata_mask = xx == nodata

    # If both are missing, we can just return the original array.
    if scale is not None and offset is not None:
        xx = (xx * scale) + offset

    if clip is not None:
        assert len(clip) == 2, "Clip must be a list of two values"
        xx = xx.clip(clip[0], clip[1])

    # Re-attach nodata
    if nodata is not None:
        xx = xx.where(~nodata_mask, other=nodata)

    xx.attrs = attrs  # Not sure if this is required

    return xx


# pylint: disable-next=dangerous-default-value
def mask_invalid_data(
    xx: DataArray | Dataset,
    nodata: int | float | None = None,
    skip_bands: Sequence[str] = [],
) -> DataArray | Dataset:
    """
    Mask out invalid data values.

    :param xx: DataArray
    :return: DataArray with invalid data values converted to np.nan. Note this will change the dtype to float.
    """
    if isinstance(xx, Dataset):
        for var in xx.data_vars:
            if var not in skip_bands:
                xx[var] = mask_invalid_data(xx[var], nodata)
        return xx

    if nodata is None:
        nodata = xx.odc.nodata

    assert nodata is not None, "Nodata value must be provided or available in attrs"

    xx = xx.where(xx != nodata)
    xx.odc.nodata = np.nan

    return xx


# pylint: disable-next=too-many-arguments, dangerous-default-value
def mask_clouds(
    xx: Dataset,
    qa_name: str,
    scale: float,
    offset: float,
    clip: tuple,
    mask_func: Callable = enum_to_bool,  # Pass the function for enum-based masks (bits_to_bool or enum_to_bool)
    mask_func_args: dict = {},
    apply_mask: bool = True,
    keep_qa: bool = False,
    return_mask: bool = False,
) -> Dataset:
    """
    General cloud masking function for both Landsat and Sentinel-2 products.

    :param xx: Dataset or DataArray
    :param qa_name: QA band to use for masking
    :param mask_classes: List of mask class values (e.g., cloud, cloud shadow)
    :param scale: Scale value for the dataset
    :param offset: Offset value for the dataset
    :param clip: Clip range for the data
    :param includ_cirrus: Whether to include cirrus in the mask
    :param apply_mask: Apply the cloud mask to the data, erasing data where clouds are present
    :param keep_qa: Keep the QA band in the output
    :param return_mask: Return the mask as a variable called "mask"
    :param enum_to_bool_func: Function to convert bit values to boolean mask (either bits_to_bool or enum_to_bool)
    :return: Dataset or DataArray with invalid data values converted to np.nan. This will change the dtype to float.
    """
    attrs = dict(xx.attrs.items())

    # Retrieve the QA band
    qa = xx[qa_name]

    # Drop the QA band and apply other preprocessing steps
    xx = xx.drop_vars(qa_name)
    xx = mask_invalid_data(xx)
    xx = scale_and_offset(xx, scale=scale, offset=offset, clip=clip)

    # Generate the mask
    mask = mask_func(qa, **mask_func_args)

    # Apply the mask if required
    if apply_mask:
        xx = xx.where(~mask)

    # Set 'nodata' to np.nan for all variables
    for var in xx.data_vars:
        xx[var].odc.nodata = np.nan

    # Optionally keep the QA band
    if keep_qa:
        xx[qa_name] = qa

    # Optionally return the mask
    if return_mask:
        xx["mask"] = mask

    xx.attrs = attrs

    return xx  # type: ignore


def mask_ls(
    xx: Dataset,
    qa_name: str = "pixel_qa",
    include_cirrus: bool = False,
    apply_mask: bool = True,
    keep_qa: bool = False,
    return_mask: bool = False,
) -> Dataset:
    """
    Perform cloud masking for Landsat Collection 2 products.
    """
    mask_bits = [
        LANDSAT_C2L2_PIXEL_QA.CLOUD.value,
        LANDSAT_C2L2_PIXEL_QA.CLOUD_SHADOW.value,
    ]
    if include_cirrus:
        mask_bits.append(LANDSAT_C2L2_PIXEL_QA.CIRRUS.value)

    return mask_clouds(
        xx=xx,
        qa_name=qa_name,
        scale=LANDSAT_C2L2_SCALE,
        offset=LANDSAT_C2L2_OFFSET,
        clip=(0.0, 1.0),
        mask_func=bits_to_bool,
        mask_func_args={"bits": mask_bits},
        apply_mask=apply_mask,
        keep_qa=keep_qa,
        return_mask=return_mask,
    )


def mask_s2(
    xx: Dataset,
    qa_name: str = "scl",
    include_cirrus: bool = False,
    apply_mask: bool = True,
    keep_qa: bool = False,
    return_mask: bool = False,
) -> Dataset:
    """
    Perform cloud masking for Sentinel-2 L2A products.
    """
    mask_values = [
        SENTINEL2_L2A_SCL.SATURATED_OR_DEFECTIVE.value,
        SENTINEL2_L2A_SCL.CLOUD_MEDIUM_PROBABILITY.value,
        SENTINEL2_L2A_SCL.CLOUD_HIGH_PROBABILITY.value,
        SENTINEL2_L2A_SCL.CLOUD_SHADOWS.value,
    ]
    if include_cirrus:
        mask_values.append(SENTINEL2_L2A_SCL.THIN_CIRRUS.value)

    return mask_clouds(
        xx=xx,
        qa_name=qa_name,
        scale=SENTINEL2_L2A_SCALE,
        offset=SENTINEL2_L2A_OFFSET,
        mask_func=enum_to_bool,
        mask_func_args={"values": mask_values},
        clip=(0.0, 1.0),
        apply_mask=apply_mask,
        keep_qa=keep_qa,
        return_mask=return_mask,
    )
