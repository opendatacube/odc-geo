"""
Read point data from LAS/COPC files
"""

from __future__ import annotations

import warnings
from typing import Literal, Sequence, Union

import laspy  # pylint: disable=import-error
import laspy.copc  # pylint: disable=import-error
import numpy as np
import xarray as xr

from odc.geo.geom import CRS
from odc.geo.xr import xr_crs_coord

DriverMode = Union[Literal["copc"], Literal["laspy"], Literal["auto"]]


def _extract(data: laspy.ScaleAwarePointRecord | laspy.LasData, var: str) -> np.ndarray:
    xx = data[var]
    if isinstance(xx, np.ndarray):
        return xx
    return xx.copy()


def xr_from_laspy(
    src: laspy.copc.CopcReader | laspy.LasReader,
    channels: Sequence[str] | None = None,
    **query,
) -> xr.Dataset:
    crs = CRS(src.header.parse_crs())
    crs_coord = xr_crs_coord(crs)

    if isinstance(src, laspy.copc.CopcReader):
        data = src.query(**query)
    else:
        if len(query) > 0:
            warnings.warn("Query params are only supported for COPC files")
        data = src.read()

    all_vars: list[str] = list(data.point_format.dimension_names)
    if channels is not None:
        keeps = set(channels)
        all_vars = [v for v in all_vars if v in keeps]

    X, Y, Z = (_extract(data, n) for n in ["x", "y", "z"])
    T = (data["gps_time"] + 10**9).astype("datetime64[s]")
    channel_names = [n for n in all_vars if n not in ("X", "Y", "Z", "gps_time")]

    return xr.Dataset(
        {n: xr.DataArray(_extract(data, n), dims=["index"]) for n in channel_names},
        coords={
            "x": xr.DataArray(X, dims=["index"]),
            "y": xr.DataArray(Y, dims=["index"]),
            "z": xr.DataArray(Z, dims=["index"]),
            "time": xr.DataArray(T, dims=["index"]),
            "index": xr.DataArray(np.arange(len(data), dtype="uint32"), dims=["index"]),
            crs_coord.name: crs_coord,
        },
    )


def load_las(
    src,
    channels: Sequence[str] | None = None,
    *,
    driver: DriverMode = "auto",
    level: int | range | None = None,
    resolution: int | float | None = None,
    bounds: laspy.copc.Bounds | None = None,
) -> xr.Dataset:
    """Load LAS file as :py:class:`xarray.Dataset`.

    :param driver: One of ``auto`` (default), ``copc`` or ``laspy``.

    COPC specific options

    :param level: Load "lower res" sample (``0`` fewest points)
    :param resolution: alternative way to specify level.
    :param bounds: Spatially crop
    """
    if driver == "auto":
        try:
            rdr = laspy.copc.CopcReader.open(src)
        except:  # pylint: disable=bare-except
            rdr = laspy.open(src)
    elif driver == "copc":
        rdr = laspy.copc.CopcReader.open(src)
    elif driver == "laspy":
        rdr = laspy.open(src)
    else:
        raise ValueError(f"Bad driver supplied: {driver}")

    return xr_from_laspy(
        rdr,
        channels=channels,
        level=level,
        resolution=resolution,
        bounds=bounds,
    )
