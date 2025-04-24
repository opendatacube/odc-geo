"""
Read point data from LAS/COPC files
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal, Sequence, Union

import numpy as np
import xarray as xr

from ..crs import MaybeCRS, norm_crs
from ..xr import xr_crs_coord

DriverMode = Union[Literal["copc"], Literal["laspy"], Literal["auto"]]
LasData = Union["laspy.ScaleAwarePointRecord", "laspy.LasData"]
LasSource = Union["laspy.copc.CopcReader", "laspy.LasReader"]


def _extract(data: LasData, var: str) -> np.ndarray:
    xx = data[var]
    if isinstance(xx, np.ndarray):
        return xx
    return xx.copy()


def _las_source_name(src: LasSource) -> str:
    # pylint: disable=import-error,import-outside-toplevel,protected-access
    import laspy.copc

    if isinstance(src, laspy.copc.CopcReader):
        return src.source.name
    if (src := getattr(src, "_source", None)) is not None:
        return src.name

    return "<unknown>"  # pragma: no cover


def _is_copc(src: LasSource) -> bool:
    # pylint: disable=import-error,import-outside-toplevel
    import laspy.copc

    return isinstance(src, laspy.copc.CopcReader)


def xr_from_laspy(
    src: LasSource,
    channels: Sequence[str] | None = None,
    force_crs: MaybeCRS = None,
    **query,
) -> xr.Dataset:
    maybe_crs_coord: dict[str, xr.DataArray] = {}
    force_crs = norm_crs(force_crs)

    if force_crs is not None:
        crs_coord = xr_crs_coord(force_crs)
        maybe_crs_coord[str(crs_coord.name)] = crs_coord
    else:
        if (wkt := src.header.parse_crs()) is None:
            warnings.warn(f"No CRS found in LAS: {_las_source_name(src)}")
        else:
            crs_coord = xr_crs_coord(wkt)
            maybe_crs_coord[str(crs_coord.name)] = crs_coord

    if _is_copc(src):
        data = src.query(**query)
    else:
        query = {k: v for k, v in query.items() if v is not None}
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
            **maybe_crs_coord,
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
    force_crs: MaybeCRS = None,
) -> xr.Dataset:
    """Load LAS file as :py:class:`xarray.Dataset`.

    :param driver: One of ``auto`` (default), ``copc`` or ``laspy``.

    COPC specific options

    :param level: Load "lower res" sample (``0`` fewest points)
    :param resolution: alternative way to specify level.
    :param bounds: Spatially crop
    """
    # pylint: disable=import-error,import-outside-toplevel
    import laspy
    import laspy.copc

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
        force_crs=force_crs,
    )


if TYPE_CHECKING:
    # pylint: disable=import-error,import-outside-toplevel
    import laspy
    import laspy.copc
