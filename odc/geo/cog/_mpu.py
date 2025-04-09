"""
Multi-part upload as a graph
"""

from __future__ import annotations

from concurrent.futures import TimeoutError as FuturesTimeoutError
from functools import partial
from typing import (
    TYPE_CHECKING,
    Any,
    Iterable,
    Iterator,
    Optional,
    Protocol,
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

    def __call__(self, part: int, data: SomeData) -> dict[str, Any]: ...

    def finalise(self, parts: list[dict[str, Any]]) -> Any: ...

    @property
    def min_write_sz(self) -> int: ...

    @property
    def max_write_sz(self) -> int: ...

    @property
    def min_part(self) -> int: ...

    @property
    def max_part(self) -> int: ...


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
        "__dict__",
    )

    def __init__(
        self,
        partId: int,
        write_credits: int,
        data: Optional[bytearray] = None,
        left_data: Optional[bytearray] = None,
        parts: Optional[list[dict[str, Any]]] = None,
        observed: Optional[list[tuple[int, Any]]] = None,
        is_final: bool = False,
        lhs_keep: int = 0,
    ) -> None:
        self.nextPartId = partId
        self.write_credits = write_credits
        self.data = bytearray() if data is None else data
        self.left_data = bytearray() if left_data is None else left_data
        self.parts: list[dict[str, Any]] = [] if parts is None else parts
        self.observed: list[tuple[int, Any]] = [] if observed is None else observed
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
            self.lhs_keep,
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
        data_to_flush = bytearray(self.data)
        if extra_data:
            data_to_flush += extra_data
        self.data = bytearray()
        writeable_len = len(data_to_flush)
        if not self.started_write and self.lhs_keep > 0:
            writeable_len = max(0, writeable_len - self.lhs_keep)
        should_flush = (self.is_final and writeable_len > 0) or (
            not self.is_final and writeable_len >= write.min_write_sz
        )
        if not self.is_final and self.write_credits < 1:
            should_flush = False
        if not should_flush:
            self.data = data_to_flush
            return 0
        bytes_flushed_total = 0
        max_chunk_size = write.max_write_sz
        current_data = data_to_flush
        if not self.started_write and self.lhs_keep > 0:
            if len(current_data) < self.lhs_keep:
                self.left_data = (
                    self.left_data + current_data if self.left_data else current_data
                )
                return 0
            self.left_data = bytearray(current_data[: self.lhs_keep])
            current_data = current_data[self.lhs_keep :]
        offset = 0
        total_size = len(current_data)
        while offset < total_size:
            if self.write_credits < 1:
                self.data = bytearray(current_data[offset:])
                raise RuntimeError(
                    f"Insufficient write credits at part {self.nextPartId}"
                )
            if not write.min_part <= self.nextPartId <= write.max_part:
                self.data = bytearray(current_data[offset:])
                raise ValueError(
                    f"Next Part ID {self.nextPartId} out of range [{write.min_part}, {write.max_part}]"
                )
            chunk_size = min(total_size - offset, max_chunk_size)
            chunk_data = current_data[offset : offset + chunk_size]
            try:
                part = write(self.nextPartId, bytes(chunk_data))
            except Exception as e:
                self.data = bytearray(current_data[offset:])
                raise RuntimeError(f"Writer failed for part {self.nextPartId}") from e
            if part.get("PartNumber") != self.nextPartId:
                pass
            self.parts.append(part)
            bytes_flushed_total += len(chunk_data)
            self.nextPartId += 1
            self.write_credits -= 1
            offset += chunk_size
        return bytes_flushed_total

    def flush(
        self,
        write: PartsWriter,
        leftPartId: Optional[int] = None,
        finalise: bool = True,
    ) -> tuple[int, Any]:
        total_bytes = 0
        result = None
        if not self.started_write:
            partId = self.nextPartId if leftPartId is None else leftPartId
            if self.data:
                part_info = write(partId, self.data)
                self.parts.append(part_info)
                total_bytes += len(self.data)
            self.data = bytearray()
        else:
            if self.data:
                orig_final = self.is_final
                self.is_final = True
                total_bytes += self.flush_rhs(write)
                self.is_final = orig_final
            if self.left_data:
                partId = write.min_part if leftPartId is None else leftPartId
                self.parts.insert(0, write(partId, self.left_data))
                total_bytes += len(self.left_data)
                self.left_data = bytearray()
        if finalise:
            result = write.finalise(self.parts)
        return total_bytes, result

    def maybe_write(self, write: PartsWriter, spill_sz: int) -> int:
        rhs_keep = 0 if self.is_final else write.min_write_sz
        lhs_keep = 0 if self.started_write else self.lhs_keep
        parts_to_keep = 0 if self.is_final else 1
        if self.write_credits - parts_to_keep < 1:
            return 0
        bytes_available = len(self.data) - rhs_keep - lhs_keep
        if bytes_available < spill_sz:
            return 0
        if lhs_keep == 0:
            spill_data = self.data[:bytes_available]
            self.data = self.data[bytes_available:]
        else:
            spill_data = self.data[lhs_keep : lhs_keep + bytes_available]
            self.left_data = self.data[:lhs_keep]
            self.data = self.data[lhs_keep + bytes_available :]
        try:
            part_info = write(self.nextPartId, bytes(spill_data))
        except Exception as e:
            if lhs_keep == 0:
                self.data = spill_data + self.data
            else:
                self.data = self.left_data + spill_data + self.data
                self.left_data = bytearray()
            raise RuntimeError(
                f"Writer failed during maybe_write for part {self.nextPartId}"
            ) from e
        self.parts.append(part_info)
        self.nextPartId += 1
        self.write_credits -= 1
        return len(spill_data)

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
                lhs_keep=lhs_keep if idx == 0 else 0,
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
        substreams: list["dask.bag.Item"],
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
            lhs_keep=lhs_keep if idx == 0 else 0,
            spill_sz=spill_sz,
            mark_final=mk_footer is None and (idx == len(chunks) - 1),
            write=write,
        )
        dss.append(sub)
        partId += ch.npartitions * writes_per_chunk
    data_substream = (
        dss[0]
        if len(dss) == 1
        else MPUChunk.collate_substreams(dss, write=write, spill_sz=spill_sz)
    )
    # pylint: disable=import-outside-toplevel
    from dask import delayed, tokenize

    tk = tokenize(write, mk_header, mk_footer, user_kw, spill_sz, data_substream)
    name = f"{dask_name_prefix}-{tk}"
    return delayed(_finalizer_dask_op, name=name, pure=False)(
        data_substream,
        write=write,
        mk_header=mk_header,
        mk_footer=mk_footer,
        user_kw=user_kw,
    )


def _mpu_collate_op(
    substreams: list[MPUChunk],
    *,
    write: Optional[PartsWriter] = None,
    spill_sz: int = 0,
) -> MPUChunk:
    if not substreams:
        raise ValueError("Received empty list of substreams to collate.")
    root = substreams[0]
    for rhs in substreams[1:]:
        root = MPUChunk.merge(root, rhs, write=write)
        if write and spill_sz > 0:
            root.maybe_write(write, spill_sz)
    return root


def _mpu_append_chunks_op(
    mpus: Iterable[MPUChunk],
    chunks: Iterable[tuple[bytes, Any]],
    write: Optional[PartsWriter] = None,
    spill_sz: int = 0,
) -> list[MPUChunk]:
    (mpu,) = mpus
    for data, chunk_id in chunks:
        mpu.append(data, chunk_id)
        if write and spill_sz > 0:
            mpu.maybe_write(write, spill_sz)
    return [mpu]


def _merge_and_spill_op(
    lhs: MPUChunk,
    rhs: MPUChunk,
    write: Optional[PartsWriter] = None,
    spill_sz: int = 0,
) -> MPUChunk:
    merged = MPUChunk.merge(lhs, rhs, write)
    if write and spill_sz:
        merged.maybe_write(write, spill_sz)
    return merged


def _finalizer_dask_op(
    data_substream: MPUChunk,
    *,
    write: PartsWriter | None = None,
    mk_header: Any = None,
    mk_footer: Any = None,
    user_kw: dict[str, Any] | None = None,
    final_task_timeout: int = 600,
):
    user_kw = user_kw or {}
    _root = data_substream
    try:
        hdr_bytes, footer_bytes = [
            None if op is None else op(_root.observed, **user_kw)
            for op in [mk_header, mk_footer]
        ]
        if footer_bytes:
            _root.append(footer_bytes)
        if hdr_bytes:
            hdr_chunk = MPUChunk(write.min_part if write else 1, 1)
            hdr_chunk.append(hdr_bytes)
            _root = MPUChunk.merge(hdr_chunk, _root, write=write)
    except Exception as e:
        raise RuntimeError(f"Failed during header/footer processing: {e}") from e

    if write is None:
        return _root

    try:
        # pylint: disable=import-outside-toplevel
        from dask.distributed import get_client

        client = get_client()
    except Exception as e:
        raise RuntimeError(f"Dask client not found in _finalizer_dask_op: {e}") from e

    try:
        future = client.submit(_remote_final_flush, _root, write, pure=False)
    except Exception as e:
        raise RuntimeError(f"Dask task submission failed: {e}") from e

    try:
        final_upload_result = future.result(timeout=final_task_timeout)
    except FuturesTimeoutError:
        try:
            future.cancel(asynchronous=True)
        except Exception as cancel_e:
            raise RuntimeError(
                f"Final task timed out after {final_task_timeout} seconds: {cancel_e}"
            ) from cancel_e
        raise
    except Exception as e:
        raise RuntimeError(f"Remote final task failed: {e}") from e
    finally:
        future.release()
    return final_upload_result


def _remote_final_flush(chunk_state: MPUChunk, writer_obj: PartsWriter):
    try:
        _, result = chunk_state.flush(writer_obj, leftPartId=None, finalise=True)
        return result
    except Exception as e:
        raise RuntimeError(f"Error during remote final flush/commit task: {e}") from e


def get_mpu_kwargs(
    mk_header=None,
    mk_footer=None,
    user_kw=None,
    writes_per_chunk=1,
    spill_sz=20 * (1 << 20),
    client=None,
) -> dict:
    """
    Construct shared keyword arguments for multipart uploads.
    """
    return {
        "mk_header": mk_header,
        "mk_footer": mk_footer,
        "user_kw": user_kw,
        "writes_per_chunk": writes_per_chunk,
        "spill_sz": spill_sz,
        "client": client,
    }


def mpu_upload(
    chunks: Union[dask.bag.Bag, list[dask.bag.Bag]],
    *,
    writer: Any,
    dask_name_prefix: str,
    **kw,
) -> "Delayed":
    """Shared logic for multipart uploads to storage services."""
    client = kw.pop("client", None)
    writer_kw = dict(kw)
    if client is not None:
        writer_kw["client"] = client
    spill_sz = kw.get("spill_sz", 20 * (1 << 20))
    write_instance: PartsWriter = None
    if spill_sz > 0:
        try:
            write_instance = writer(**writer_kw)
        except Exception as e:
            raise RuntimeError(f"Writer {writer} instantiation failed: {e}.") from e
    return mpu_write(chunks, write_instance, dask_name_prefix=dask_name_prefix, **kw)
