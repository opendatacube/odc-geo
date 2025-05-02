"""
Read point data from LAS/COPC files
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from math import ceil, log2
from typing import TYPE_CHECKING, Literal, Protocol, Type, TypeAlias, Union

import numpy as np
import xarray as xr

from ..crs import MaybeCRS, norm_crs
from ..xr import xr_crs_coord

DriverMode: TypeAlias = Union[Literal["copc"], Literal["laspy"], Literal["auto"]]
ScaleAwarePointRecord = Type["laspy.ScaleAwarePointRecord"]
LasData = Type["laspy.LasData"]
LasReader = Type["laspy.LasReader"]
CopcReader = Type["laspy.copc.CopcReader"]
CopcInfoVlr = Type["laspy.copc.CopcInfoVlr"]
CopcEntry = Type["laspy.copc.Entry"]
CopcBounds = Type["laspy.copc.Bounds"]
LasSource: TypeAlias = Union[CopcReader, LasReader]


class CopcEntryLike(Protocol):
    """
    A protocol for COPC entry-like objects.

    :ivar point_count: Number of points in the chunk
    :ivar byte_size: Size of the chunk in bytes
    :ivar offset: Offset of the chunk in the file
    """

    point_count: int
    byte_size: int
    offset: int


class ChunkMD:
    """
    Minimal metadata for a chunk of a COPC file.

    :ivar point_count: Number of points in the chunk
    :ivar byte_size: Size of the chunk in bytes
    :ivar offset: Offset of the chunk in the file
    """

    __slots__ = ("point_count", "byte_size", "offset")

    def __init__(self, point_count: int, byte_size: int, offset: int):
        self.point_count = point_count
        self.byte_size = byte_size
        self.offset = offset

    def __repr__(self) -> str:
        return f"Chunk(point_count={self.point_count}, byte_size={self.byte_size}, offset={self.offset})"

    def __str__(self) -> str:
        return self.__repr__()


def _extract(data: LasData | ScaleAwarePointRecord, var: str) -> np.ndarray:
    xx = data[var]
    if isinstance(xx, np.ndarray):
        return xx
    return xx.copy()


def _extract_all(
    data: LasData | ScaleAwarePointRecord,
    channels: Sequence[str] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    time_var_name = "gps_time"
    if channels is None:
        all_vars: list[str] = list(data.point_format.dimension_names)
    else:
        keeps = set(channels)
        all_vars = [v for v in data.point_format.dimension_names if v in keeps]

    channel_names = [n for n in all_vars if n not in ("X", "Y", "Z", time_var_name)]

    X, Y, Z, T = (_extract(data, n) for n in ["x", "y", "z", time_var_name])
    coords = {"x": X, "y": Y, "z": Z, "time": (T + 10**9).astype("datetime64[s]")}

    data_vars = {n: _extract(data, n) for n in channel_names}
    return coords, data_vars


def _las_source_name(src: LasSource) -> str:
    # pylint: disable=protected-access
    stream = getattr(src, "source", getattr(src, "_source", None))
    if stream is None:
        return "<unknown>"
    return getattr(stream, "name", "<unknown>")


def _is_copc(src: LasSource) -> bool:
    # pylint: disable=import-error,import-outside-toplevel
    import laspy.copc

    return isinstance(src, laspy.copc.CopcReader)


def _norm_query(
    copc_info: CopcInfoVlr,
    resolution: int | float | None = None,
    level: int | range | None = None,
    bounds: CopcBounds | None = None,
) -> tuple[range | None, CopcBounds | None]:
    if resolution is not None:
        level_max = max(1, ceil(log2(copc_info.spacing / resolution)) + 1)
        level = range(0, level_max)

    if isinstance(level, int):
        level = range(level, level + 1)

    return level, bounds


def _extract_crs_coord(
    src: LasSource, force_crs: MaybeCRS | None
) -> dict[str, xr.DataArray]:
    maybe_crs_coord: dict[str, xr.DataArray] = {}
    force_crs = norm_crs(force_crs)

    if force_crs is not None:
        crs_coord = xr_crs_coord(force_crs)
        maybe_crs_coord[str(crs_coord.name)] = crs_coord
    else:
        if (wkt := src.header.parse_crs()) is not None:
            crs_coord = xr_crs_coord(wkt)
            maybe_crs_coord[str(crs_coord.name)] = crs_coord
        else:
            warnings.warn(f"No CRS found in LAS: {_las_source_name(src)}")

    return maybe_crs_coord


def xr_from_laspy(
    src: LasSource,
    channels: Sequence[str] | None = None,
    force_crs: MaybeCRS = None,
    **query,
) -> xr.Dataset:
    maybe_crs_coord = _extract_crs_coord(src, force_crs)

    if _is_copc(src):
        data = src.query(**query)
    else:
        query = {k: v for k, v in query.items() if v is not None}
        if len(query) > 0:
            warnings.warn("Query params are only supported for COPC files")
        data = src.read()

    dims = ("index",)
    coords, data_vars = _extract_all(data, channels)
    coords = {k: xr.DataArray(v, dims=dims) for k, v in coords.items()}
    coords.update(
        index=xr.DataArray(np.arange(len(data), dtype="uint32"), dims=dims),
        **maybe_crs_coord,
    )

    data_vars = {n: xr.DataArray(v, dims=dims) for n, v in data_vars.items()}

    return xr.Dataset(data_vars, coords=coords)


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


class ChunkExtractor:
    """
    Decompress a chunk of a COPC file.

    :param header: Header of the COPC file
    :param record_data: Bytes of LASZIP VLR record data
    """

    __slots__ = ("_header", "_record_data")

    def __init__(
        self,
        header: "laspy.LasHeader",
        record_data: bytes,
    ):
        self._header = header
        self._record_data = record_data

    @classmethod
    def from_copc_reader(cls, src: CopcReader) -> "ChunkExtractor":
        return cls(src.header, src.laszip_vlr.record_data)

    @staticmethod
    def load_chunk_metadata(
        src: CopcReader,
        *,
        resolution: int | float | None = None,
        level: int | range | None = None,
        bounds: CopcBounds | None = None,
    ) -> list[ChunkMD]:
        # pylint: disable=import-error,import-outside-toplevel
        from laspy.copc import load_octree_for_query

        level_range, query_bounds = _norm_query(
            src.copc_info,
            resolution=resolution,
            level=level,
            bounds=bounds,
        )

        root_page = src.root_page
        nodes = load_octree_for_query(
            src.source,
            src.copc_info,
            root_page,
            query_bounds=query_bounds,
            level_range=level_range,
        )
        return [ChunkMD(e.point_count, e.byte_size, e.offset) for e in nodes]

    def __call__(
        self,
        compressed_bytes: bytes,
        what: CopcEntryLike | Sequence[CopcEntryLike] | int,
    ) -> "laspy.ScaleAwarePointRecord":
        """
        Decompress a chunk of a COPC file.

        :param compressed_bytes: Bytes of compressed point data
        :param what: What are we decompressing?

            - Expected number of points present in the compressed data of a single node
            - Sequence of nodes present in the compressed data in specified order
            - Or a single node
        """
        # pylint: disable=import-error,import-outside-toplevel,no-name-in-module,redefined-outer-name
        from laspy import PackedPointRecord, ScaleAwarePointRecord
        from lazrs import decompress_points_with_chunk_table

        if isinstance(what, Sequence):
            chunk_table = [(n.point_count, n.byte_size) for n in what]
            num_points = sum(n.point_count for n in what)
        else:
            num_points = what if isinstance(what, int) else what.point_count
            chunk_table = [(num_points, len(compressed_bytes))]

        hdr = self._header
        _data_u8 = np.zeros(num_points * hdr.point_format.size, dtype=np.uint8)

        decompress_points_with_chunk_table(
            compressed_bytes,
            self._record_data,
            _data_u8,
            chunk_table,
        )

        ppr = PackedPointRecord.from_buffer(_data_u8, hdr.point_format)
        return ScaleAwarePointRecord(
            ppr.array, ppr.point_format, hdr.scales, hdr.offsets
        )


if TYPE_CHECKING:
    # pylint: disable=import-error,import-outside-toplevel
    import laspy
    import laspy.copc
