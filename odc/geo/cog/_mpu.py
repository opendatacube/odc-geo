"""
Multi-part upload as a graph
"""
from __future__ import annotations

from functools import partial
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Protocol,
    Tuple,
    Union,
)

if TYPE_CHECKING:
    import dask.bag
    from dask.delayed import Delayed

__all__ = [
    "SomeData",
    "PartsWriter",
    "MPUChunk",
    "mpu_write",
]

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

    # pylint: disable=too-many-arguments,too-many-instance-attributes

    __slots__ = (
        "nextPartId",
        "write_credits",
        "data",
        "left_data",
        "parts",
        "observed",
        "is_final",
        "lhs_keep",
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
        lhs_keep: int = 0,
    ) -> None:
        self.nextPartId = partId
        self.write_credits = write_credits
        self.data = bytearray() if data is None else data
        self.left_data = bytearray() if left_data is None else left_data
        self.parts: List[Dict[str, Any]] = [] if parts is None else parts
        self.observed: List[Tuple[int, Any]] = [] if observed is None else observed
        self.is_final = is_final
        self.lhs_keep = lhs_keep
        # if supplying data must also supply observed
        assert data is None or (observed is not None and len(observed) > 0)

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
            assert len(rhs.left_data) == 0
            assert len(rhs.parts) == 0

            return MPUChunk(
                lhs.nextPartId,
                lhs.write_credits + rhs.write_credits,
                lhs.data + rhs.data,
                lhs.left_data,
                lhs.parts,
                lhs.observed + rhs.observed,
                rhs.is_final,
                lhs.lhs_keep,
            )

        # Flush `lhs.data + rhs.left_data` if we can
        #  or else move it into .left_data
        lhs.flush_rhs(write, rhs.left_data)

        return MPUChunk(
            rhs.nextPartId,
            rhs.write_credits,
            rhs.data,
            lhs.left_data,
            lhs.parts + rhs.parts,
            lhs.observed + rhs.observed,
            rhs.is_final,
            lhs.lhs_keep,
        )

    def flush_rhs(
        self, write: Optional[PartsWriter], extra_data: Optional[bytearray] = None
    ) -> int:
        data = self.data
        if extra_data is not None and len(extra_data):
            data += extra_data

        def _flush_data(pw: PartsWriter):
            assert pw.min_part <= self.nextPartId <= pw.max_part

            _data = data
            if not self.started_write and self.lhs_keep > 0:
                self.left_data = bytearray(_data[: self.lhs_keep])
                _data = data[self.lhs_keep :]

            part = pw(self.nextPartId, _data)

            self.parts.append(part)
            self.data = bytearray()
            self.nextPartId += 1
            self.write_credits -= 1
            return len(_data)

        def can_flush(pw: PartsWriter):
            if self.write_credits < 1:
                return False
            if self.started_write:
                return self.is_final or len(data) >= pw.min_write_sz
            if self.is_final:
                return len(data) > self.lhs_keep
            return len(data) - self.lhs_keep >= pw.min_write_sz

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

        self.left_data, self.data = self.left_data + data, bytearray()
        return 0

    def flush(
        self,
        write: PartsWriter,
        leftPartId: Optional[int] = None,
        finalise: bool = True,
    ) -> Tuple[int, Any]:
        rr = None
        if not self.started_write:
            assert not self.left_data
            # TODO: special case this code path
            partId = self.nextPartId if leftPartId is None else leftPartId
            spill_data = self.data
            self.parts.append(write(partId, spill_data))
            self.data = bytearray()

            if finalise:
                rr = write.finalise(self.parts)

            return len(spill_data), rr

        bytes_written = 0
        if self.data:
            self.is_final = True
            bytes_written = self.flush_rhs(write)

        if self.left_data:
            assert len(self.left_data) >= write.min_write_sz
            partId = 1 if leftPartId is None else leftPartId
            self.parts.insert(0, write(partId, self.left_data))
            bytes_written += len(self.left_data)
            self.left_data = bytearray()

        if finalise:
            rr = write.finalise(self.parts)

        return bytes_written, rr

    def maybe_write(self, write: PartsWriter, spill_sz: int) -> int:
        # if not last section keep 'min_write_sz' and 1 partId around after flush
        rhs_keep, parts_to_keep = (0, 0) if self.is_final else (write.min_write_sz, 1)
        lhs_keep = 0 if self.started_write else self.lhs_keep

        if self.write_credits - 1 < parts_to_keep:
            return 0

        bytes_to_write = len(self.data) - rhs_keep - lhs_keep
        if bytes_to_write < spill_sz:
            return 0

        if lhs_keep == 0:
            spill_data = self.data[:bytes_to_write]
            self.data = bytearray(self.data[bytes_to_write:])
        else:
            spill_data = self.data[lhs_keep : lhs_keep + bytes_to_write]
            assert not self.left_data
            self.left_data = bytearray(self.data[:lhs_keep])
            self.data = bytearray(self.data[bytes_to_write + lhs_keep :])

        assert len(spill_data) == bytes_to_write
        assert len(spill_data) >= spill_sz

        self.parts.append(write(self.nextPartId, spill_data))
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
        lhs_keep: int = 0,
    ) -> Iterator["MPUChunk"]:
        for idx in range(n):
            is_final = mark_final and idx == (n - 1)
            yield MPUChunk(
                partId + idx * writes_per_chunk,
                writes_per_chunk,
                is_final=is_final,
                lhs_keep=lhs_keep,
            )

    @staticmethod
    def from_dask_bag(
        partId: int,
        chunks: "dask.bag.Bag",
        *,
        writes_per_chunk: int = 1,
        mark_final: bool = False,
        lhs_keep: int = 0,
        write: Optional[PartsWriter] = None,
        spill_sz: int = 0,
        split_every: int = 4,  # when applying fold take 4 at a time
    ) -> "dask.bag.Item":
        # pylint: disable=import-outside-toplevel
        import dask.bag

        mpus = dask.bag.from_sequence(
            MPUChunk.gen_bunch(
                partId,
                chunks.npartitions,
                writes_per_chunk=writes_per_chunk,
                mark_final=mark_final,
                lhs_keep=lhs_keep,
            ),
            npartitions=chunks.npartitions,
        )

        mpus = dask.bag.map_partitions(
            _mpu_append_chunks_op,
            mpus,
            chunks,
            write=write,
            spill_sz=spill_sz,
            token="mpu.append",
        )

        return mpus.fold(
            partial(_merge_and_spill_op, write=write, spill_sz=spill_sz),
            split_every=split_every,
        )

    @staticmethod
    def collate_substreams(
        substreams: List["dask.bag.Item"],
        *,
        write: Optional[PartsWriter] = None,
        spill_sz: int = 0,
    ) -> "dask.bag.Item":
        # pylint: disable=import-outside-toplevel
        import dask.bag
        from dask import delayed

        assert len(substreams) > 0

        return dask.bag.Item.from_delayed(
            delayed(_mpu_collate_op)(
                substreams, pure=False, write=write, spill_sz=spill_sz
            )
        )


def mpu_write(
    chunks: "dask.bag.Bag" | list["dask.bag.Bag"],
    write: PartsWriter | None = None,
    *,
    mk_header: Any = None,
    mk_footer: Any = None,
    user_kw: dict[str, Any] | None = None,
    writes_per_chunk: int = 1,
    spill_sz: int = 20 * (1 << 20),
    dask_name_prefix="mpufinalise",
) -> "Delayed":
    # pylint: disable=import-outside-toplevel,too-many-locals,too-many-arguments
    from dask.base import tokenize
    from dask.delayed import delayed

    if not isinstance(chunks, list):
        chunks = [chunks]
    if write is None:
        min_part = 1
        lhs_keep = 0
    else:
        min_part = write.min_part
        lhs_keep = write.min_write_sz

    partId = min_part + 1
    dss: list["dask.bag.Item"] = []
    for idx, ch in enumerate(chunks):
        sub = MPUChunk.from_dask_bag(
            partId,
            ch,
            writes_per_chunk=writes_per_chunk,
            lhs_keep=lhs_keep,
            spill_sz=spill_sz,
            mark_final=mk_footer is None and (idx == len(chunks) - 1),
            write=write,
        )
        dss.append(sub)
        partId = partId + ch.npartitions * writes_per_chunk

    if len(dss) == 1:
        data_substream = dss[0]
    else:
        data_substream = MPUChunk.collate_substreams(
            dss,
            write=write,
            spill_sz=spill_sz,
        )

    tk = tokenize(write, mk_header, mk_footer, user_kw, spill_sz)
    name = f"{dask_name_prefix}-{tk}"

    return delayed(_finalizer_dask_op, name=name, pure=True)(
        data_substream,
        write=write,
        mk_header=mk_header,
        mk_footer=mk_footer,
        user_kw=user_kw,
        dask_key_name=name,
    )


def _mpu_collate_op(
    substreams: List[MPUChunk],
    *,
    write: Optional[PartsWriter] = None,
    spill_sz: int = 0,
) -> MPUChunk:
    assert len(substreams) > 0
    root, *rest = substreams
    for rhs in rest:
        root = MPUChunk.merge(root, rhs, write=write)
        if write and spill_sz:
            root.maybe_write(write, spill_sz)
    return root


def _mpu_append_chunks_op(
    mpus: Iterable[MPUChunk],
    chunks: Iterable[Tuple[bytes, Any]],
    write: Optional[PartsWriter] = None,
    spill_sz: int = 0,
):
    # expect 1 MPUChunk per partition
    (mpu,) = mpus
    for chunk in chunks:
        data, chunk_id = chunk
        mpu.append(data, chunk_id)
        if write is not None and spill_sz > 0:
            mpu.maybe_write(write, spill_sz)

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


def _finalizer_dask_op(
    data_substream: MPUChunk,
    *,
    write: PartsWriter | None = None,
    mk_header: Any = None,
    mk_footer: Any = None,
    user_kw: dict[str, Any] | None = None,
):
    if user_kw is None:
        user_kw = {}

    _root = data_substream
    hdr_bytes, footer_bytes = [
        None if op is None else op(data_substream.observed, **user_kw)
        for op in [mk_header, mk_footer]
    ]

    if footer_bytes:
        _root.append(footer_bytes)

    if hdr_bytes:
        hdr = MPUChunk(1, 1)
        hdr.append(hdr_bytes)
        _root = MPUChunk.merge(hdr, _root)

    if write is None:
        return _root

    _, rr = _root.flush(write, leftPartId=1, finalise=True)
    return rr
