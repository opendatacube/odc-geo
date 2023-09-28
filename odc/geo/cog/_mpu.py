"""
Multi-part upload as a graph
"""
from functools import partial
from typing import Any, Dict, Iterable, Iterator, List, Optional, Protocol, Tuple, Union

from dask import bag as dask_bag

SomeData = Union[bytes, bytearray]


class PartsWriter(Protocol):
    """Protocol for labeled parts data writer."""

    def __call__(self, part: int, data: SomeData) -> Dict[str, Any]:
        ...

    def finalise(self, parts: List[Dict[str, Any]]) -> Any:
        ...

    @property
    def min_write_sz(self) -> int:
        ...

    @property
    def max_write_sz(self) -> int:
        ...

    @property
    def min_part(self) -> int:
        ...

    @property
    def max_part(self) -> int:
        ...


class MPUChunk:
    """
    chunk cache and writer
    """

    # pylint: disable=too-many-arguments

    __slots__ = (
        "nextPartId",
        "write_credits",
        "data",
        "left_data",
        "parts",
        "observed",
        "is_final",
    )

    def __init__(
        self,
        partId: int,
        write_credits: int,
        data: Optional[bytearray] = None,
        left_data: Optional[bytearray] = None,
        parts: Optional[List[Dict[str, Any]]] = None,
        observed: Optional[List[Tuple[int, Any]]] = None,
        is_final: bool = False,
    ) -> None:
        self.nextPartId = partId
        self.write_credits = write_credits
        self.data = bytearray() if data is None else data
        self.left_data = left_data
        self.parts: List[Dict[str, Any]] = [] if parts is None else parts
        self.observed: List[Tuple[int, Any]] = [] if observed is None else observed
        self.is_final = is_final

    def __dask_tokenize__(self):
        return (
            "MPUChunk",
            self.nextPartId,
            self.write_credits,
            self.data,
            self.left_data,
            self.parts,
            self.observed,
            self.is_final,
        )

    def __repr__(self) -> str:
        s = f"MPUChunk: {self.nextPartId}#{self.write_credits} cache: {len(self.data)}"
        if self.observed:
            s = f"{s} observed[{len(self.observed)}]"
        if self.parts:
            s = f"{s} parts: [{len(self.parts)}]"
        if self.is_final:
            s = f"{s} final"
        return s

    def append(self, data: SomeData, chunk_id: Any = None):
        sz = len(data)
        self.observed.append((sz, chunk_id))
        self.data += data

    @property
    def started_write(self) -> bool:
        return len(self.parts) > 0

    @staticmethod
    def merge(
        lhs: "MPUChunk",
        rhs: "MPUChunk",
        write: Optional[PartsWriter] = None,
    ) -> "MPUChunk":
        """
        If ``write=`` is not provided but flush is needed, RuntimeError will be raised.
        """
        if not rhs.started_write:
            # no writes on the right
            # Just append
            assert rhs.left_data is None
            assert len(rhs.parts) == 0

            return MPUChunk(
                lhs.nextPartId,
                lhs.write_credits + rhs.write_credits,
                lhs.data + rhs.data,
                lhs.left_data,
                lhs.parts,
                lhs.observed + rhs.observed,
                rhs.is_final,
            )

        # Flush `lhs.data + rhs.left_data` if we can
        #  or else move it into .left_data
        lhs.final_flush(write, rhs.left_data)

        return MPUChunk(
            rhs.nextPartId,
            rhs.write_credits,
            rhs.data,
            lhs.left_data,
            lhs.parts + rhs.parts,
            lhs.observed + rhs.observed,
            rhs.is_final,
        )

    def final_flush(
        self, write: Optional[PartsWriter], extra_data: Optional[bytearray] = None
    ) -> int:
        data = self.data
        if extra_data is not None:
            data += extra_data

        def _flush_data(pw: PartsWriter):
            assert pw.min_part <= self.nextPartId <= pw.max_part

            part = pw(self.nextPartId, data)
            self.parts.append(part)
            self.data = bytearray()
            self.nextPartId += 1
            self.write_credits -= 1
            return len(data)

        def can_flush(pw: PartsWriter):
            return self.write_credits > 0 and (
                self.is_final or len(data) >= pw.min_write_sz
            )

        if self.started_write:
            # When starting to write we ensure that there is always enough
            # data and write credits left to flush the remainder
            #
            # User must have provided `write` function

            if write is None:
                raise RuntimeError("Flush required but no writer provided")

            assert can_flush(write)
            return _flush_data(write)

        # Haven't started writing yet
        # - Flush if possible and writer is provided
        # - OR just move all the data to .left_data section
        if write is not None and can_flush(write):
            return _flush_data(write)

        if self.left_data is None:
            self.left_data, self.data = data, bytearray()
        else:
            self.left_data, self.data = self.left_data + data, bytearray()

        return 0

    def maybe_write(self, write: PartsWriter, spill_sz: int) -> int:
        # if not last section keep 'min_write_sz' and 1 partId around after flush
        bytes_to_keep, parts_to_keep = (
            (0, 0) if self.is_final else (write.min_write_sz, 1)
        )

        if self.write_credits - 1 < parts_to_keep:
            return 0

        bytes_to_write = len(self.data) - bytes_to_keep
        if bytes_to_write < spill_sz:
            return 0

        part = write(self.nextPartId, self.data[:bytes_to_write])

        self.parts.append(part)
        self.data = self.data[bytes_to_write:]
        self.nextPartId += 1
        self.write_credits -= 1

        return bytes_to_write

    @staticmethod
    def gen_bunch(
        partId: int,
        n: int,
        *,
        writes_per_chunk: int = 1,
        mark_final: bool = False,
    ) -> Iterator["MPUChunk"]:
        for idx in range(n):
            is_final = mark_final and idx == (n - 1)
            yield MPUChunk(
                partId + idx * writes_per_chunk, writes_per_chunk, is_final=is_final
            )

    @staticmethod
    def from_dask_bag(
        partId: int,
        chunks: dask_bag.Bag,
        *,
        writes_per_chunk: int = 1,
        mark_final: bool = False,
        write: Optional[PartsWriter] = None,
        spill_sz: int = 0,
    ) -> dask_bag.Item:
        mpus = dask_bag.from_sequence(
            MPUChunk.gen_bunch(
                partId,
                chunks.npartitions,
                writes_per_chunk=writes_per_chunk,
                mark_final=mark_final,
            ),
            npartitions=chunks.npartitions,
        )

        mpus = dask_bag.map_partitions(
            _mpu_append_chunks_op,
            mpus,
            chunks,
            token="mpu.append",
        )

        return mpus.fold(partial(_merge_and_spill_op, write=write, spill_sz=spill_sz))


def _mpu_append_chunks_op(
    mpus: Iterable[MPUChunk], chunks: Iterable[Tuple[bytes, Any]]
):
    # expect 1 MPUChunk per partition
    (mpu,) = mpus
    for chunk in chunks:
        data, chunk_id = chunk
        mpu.append(data, chunk_id)
    return [mpu]


def _merge_and_spill_op(
    lhs: MPUChunk,
    rhs: MPUChunk,
    write: Optional[PartsWriter] = None,
    spill_sz: int = 0,
) -> MPUChunk:
    mm = MPUChunk.merge(lhs, rhs, write)
    if write is None or spill_sz == 0:
        return mm

    mm.maybe_write(write, spill_sz)
    return mm
