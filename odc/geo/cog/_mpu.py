"""
Multi-part upload as a graph
"""

from __future__ import annotations
import logging

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
    from dask import delayed
    from dask.base import tokenize
    from dask.delayed import Delayed
    from dask.distributed import Client, Future, get_client, Lock


__all__ = [
    "SomeData",
    "PartsWriter",
    "MPUChunk",
    "mpu_write",
]

mpu_logger = logging.getLogger(__name__)
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

    __slots__ = (
        "nextPartId",
        "write_credits",
        "data",
        "left_data",
        "parts",
        "observed",
        "is_final",
        "lhs_keep",
        "_global_counter",
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
        self._global_counter = 0
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
            s = f"{s} observed: [{len(self.observed)}]"
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
            # no writes on the right – just append
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
        lhs.flush_rhs(write, rhs.left_data)
        # Use a Dask distributed lock to ensure the nextPartId update is atomic
        lock = Lock("mpu_merge_lock")
        with lock:
            all_parts = lhs.parts + rhs.parts
            all_parts.sort(key=lambda p: int(p["PartNumber"]))
            new_nextPartId = (
                all_parts[-1]["PartNumber"] + 1
                if all_parts
                else max(lhs.nextPartId, rhs.nextPartId)
            )
        return MPUChunk(
            new_nextPartId,
            rhs.write_credits,
            rhs.data,
            lhs.left_data,
            all_parts,
            lhs.observed + rhs.observed,
            rhs.is_final,
            lhs.lhs_keep,
        )

    def flush_rhs(
        self,
        write: Optional["PartsWriter"],
        extra_data: Optional[bytearray] = None,
    ) -> int:
        """
        Flushes self.data (plus optional extra_data from RHS during merge),
        respecting max_write_sz by potentially writing multiple parts.
        Returns total bytes flushed.
        """
        if write is None:
            combined_len = len(self.data) + (len(extra_data) if extra_data else 0)
            if combined_len > 0:
                mpu_logger.error("flush_rhs: Flush required but no writer provided.")
                raise RuntimeError("Flush required but no writer provided")
            return 0

        data_to_flush = bytearray(self.data)
        if extra_data is not None and len(extra_data):
            data_to_flush += extra_data
            mpu_logger.debug(
                "flush_rhs: Combined internal buffer (%s bytes) with extra data (%s bytes). Total: %s bytes.",
                len(self.data),
                len(extra_data),
                len(data_to_flush),
            )
        else:
            mpu_logger.debug(
                "flush_rhs: Preparing to flush internal buffer (%s bytes).",
                len(data_to_flush),
            )

        self.data = bytearray()

        bytes_flushed_total = 0
        max_chunk_size = write.max_write_sz

        def _flush_looping(pw: PartsWriter):
            nonlocal bytes_flushed_total
            current_data = data_to_flush

            if not self.started_write and self.lhs_keep > 0:
                if len(current_data) < self.lhs_keep:
                    mpu_logger.warning(
                        "flush_rhs: Data (%s) less than lhs_keep (%s). Keeping all in left_data.",
                        len(current_data),
                        self.lhs_keep,
                    )
                    if self.left_data:
                        self.left_data += current_data
                    else:
                        self.left_data = current_data
                    return 0
                else:
                    if self.left_data:
                        mpu_logger.warning(
                            "flush_rhs: Overwriting non-empty left_data while handling lhs_keep."
                        )
                    self.left_data = bytearray(current_data[: self.lhs_keep])
                    current_data = current_data[self.lhs_keep :]
                    mpu_logger.debug(
                        "flush_rhs: Separated %s bytes into left_data.",
                        len(self.left_data),
                    )

            offset = 0
            total_size_to_write = len(current_data)
            mpu_logger.debug(
                "flush_rhs: Starting loop to flush %s bytes.", total_size_to_write
            )

            while offset < total_size_to_write:
                if self.write_credits < 1:
                    mpu_logger.error(
                        "flush_rhs: Ran out of write credits at part %s while flushing large chunk.",
                        self.nextPartId,
                    )
                    self.data = bytearray(current_data[offset:])
                    mpu_logger.warning(
                        "flush_rhs: Stored %s remaining bytes back in buffer due to lack of credits.",
                        len(self.data),
                    )
                    raise RuntimeError(
                        f"Insufficient write credits ({self.write_credits}) during"
                        f"multi-part flush_rhs at part {self.nextPartId}"
                    )

                if not pw.min_part <= self.nextPartId <= pw.max_part:
                    self.data = bytearray(current_data[offset:])
                    raise ValueError(
                        f"flush_rhs: Next Part ID {self.nextPartId} out of writer range [{pw.min_part}, {pw.max_part}]"
                    )

                chunk_size = min(total_size_to_write - offset, max_chunk_size)
                chunk_data = current_data[offset : offset + chunk_size]

                mpu_logger.info(
                    "flush_rhs: Writing part %s, size %s (max: %s)",
                    self.nextPartId,
                    len(chunk_data),
                    max_chunk_size,
                )
                try:
                    part = pw(self.nextPartId, bytes(chunk_data))
                except Exception as e:
                    mpu_logger.error(
                        "flush_rhs: Writer failed for part %s: %s",
                        self.nextPartId,
                        e,
                        exc_info=True,
                    )
                    self.data = bytearray(current_data[offset:])
                    raise RuntimeError(
                        f"Writer failed during flush_rhs for part {self.nextPartId}"
                    ) from e

                if "PartNumber" not in part or "BlockId" not in part:
                    mpu_logger.warning(
                        "flush_rhs: Writer returned malformed part metadata for part %s: %s",
                        self.nextPartId,
                        part,
                    )
                    self.data = bytearray(current_data[offset:])
                    raise ValueError(
                        f"Writer for part {self.nextPartId} returned malformed metadata: {part}"
                    )

                if part.get("PartNumber") != self.nextPartId:
                    mpu_logger.warning(
                        "PartNumber mismatch for part %s: Expected %s, got %s",
                        self.nextPartId,
                        self.nextPartId,
                        part.get("PartNumber"),
                    )

                self.parts.append(part)
                bytes_flushed_total += len(chunk_data)

                self.nextPartId += 1
                self.write_credits -= 1
                offset += chunk_size
                mpu_logger.debug(
                    "flush_rhs: Part written. Next ID: %s, Credits left: %s",
                    self.nextPartId,
                    self.write_credits,
                )

            mpu_logger.debug(
                "flush_rhs: Finished flushing loop. Total bytes written: %s",
                bytes_flushed_total,
            )
            return bytes_flushed_total

        def can_flush_now(pw: PartsWriter):
            writeable_len = len(data_to_flush)
            if not self.started_write and self.lhs_keep > 0:
                writeable_len = max(0, writeable_len - self.lhs_keep)

            should_flush = (self.is_final and writeable_len > 0) or (
                not self.is_final and writeable_len >= pw.min_write_sz
            )

            if not self.is_final and self.write_credits < 1:
                return False

            return should_flush

        if can_flush_now(write):
            mpu_logger.debug("flush_rhs: Conditions met, proceeding with flush.")
            try:
                return _flush_looping(write)
            except Exception:
                raise
        else:
            self.data = data_to_flush
            mpu_logger.debug(
                "flush_rhs: Conditions not met (is_final=%s, available=%s, min_write=%s), deferring flush.",
                self.is_final,
                len(self.data),
                write.min_write_sz,
            )
            return 0

    def flush(
        self,
        write: PartsWriter,
        leftPartId: Optional[int] = None,
        finalise: bool = True,
    ) -> tuple[int, Any]:
        """
        Flush the current chunk.
        Prior to finalisation, if data has been accumulated but not flushed and only one write credit remains,
        do not flush left_data separately. Instead, leave it in memory to be merged with subsequent data.
        """
        rr = None
        if not self.started_write:
            assert (
                not self.left_data
            ), "Leftover data should be empty if no flush occurred"
            partId = self.nextPartId if leftPartId is None else leftPartId
            spill_data = self.data
            if spill_data:
                part_info = write(partId, spill_data)
                part_info["global_index"] = self._global_counter
                self._global_counter += 1
                self.parts.append(part_info)
            self.data = bytearray()

            if finalise:
                for part in self.parts:
                    if "PartNumber" not in part or "BlockId" not in part:
                        raise ValueError(f"Malformed part metadata: {part}")
                rr = write.finalise(self.parts)
            return len(spill_data), rr

        bytes_written = 0
        if self.data:
            self.is_final = True
            bytes_written = self.flush_rhs(write)

        if self.left_data:
            if len(self.left_data) >= write.min_write_sz:
                partId = self.nextPartId if leftPartId is None else leftPartId
                self.parts.insert(0, write(partId, self.left_data))
                bytes_written += len(self.left_data)
                self.left_data = bytearray()
            else:
                logging.getLogger(__name__).debug(
                    "Leftover data size (%s bytes) is less than min_write_sz; deferring flush.",
                    len(self.left_data),
                )
        if finalise:
            for part in self.parts:
                if "PartNumber" not in part or "BlockId" not in part:
                    raise ValueError(f"Malformed part metadata: {part}")
            rr = write.finalise(self.parts)
        return bytes_written, rr

    def maybe_write(self, write: "PartsWriter", spill_sz: int) -> int:
        """
        Spill data to the writer if enough data is available, respecting max_write_sz.
        Writes at most one chunk per call. Returns number of bytes written.
        """
        max_chunk_size = write.max_write_sz
        rhs_keep = 0 if self.is_final else write.min_write_sz
        lhs_keep = 0 if self.started_write else self.lhs_keep

        if self.write_credits < 1:
            mpu_logger.debug("maybe_write: Insufficient write credits, skipping write.")
            return 0

        bytes_available_total = len(self.data) - rhs_keep - lhs_keep

        if bytes_available_total < spill_sz:
            mpu_logger.debug(
                "maybe_write: Available bytes (%s) < spill_sz (%s), skipping write.",
                bytes_available_total,
                spill_sz,
            )
            return 0

        spill_chunk_size = min(bytes_available_total, max_chunk_size)

        if spill_chunk_size <= 0:
            mpu_logger.warning(
                "maybe_write: Calculated spill_chunk_size <= 0, skipping write."
            )
            return 0

        spill_data: bytearray
        if lhs_keep == 0:
            spill_data = self.data[:spill_chunk_size]
            self.data = self.data[spill_chunk_size:]
            mpu_logger.debug(
                "maybe_write: Sliced %s bytes from start of buffer.", len(spill_data)
            )
        else:
            spill_data = self.data[lhs_keep : lhs_keep + spill_chunk_size]
            if self.left_data:
                mpu_logger.warning(
                    "maybe_write: Overwriting non-empty left_data while handling lhs_keep."
                )
            self.left_data = self.data[:lhs_keep]
            self.data = self.data[lhs_keep + spill_chunk_size :]
            mpu_logger.debug(
                "maybe_write: Separated %s lhs bytes, sliced %s bytes.",
                len(self.left_data),
                len(spill_data),
            )

        if any(p["PartNumber"] == self.nextPartId for p in self.parts):
            mpu_logger.warning(
                "maybe_write: nextPartId %s collision detected, incrementing.",
                self.nextPartId,
            )
            self.nextPartId += 1

        if not write.min_part <= self.nextPartId <= write.max_part:
            if lhs_keep == 0:
                self.data = spill_data + self.data
            else:
                self.data = self.left_data + spill_data + self.data
                self.left_data = bytearray()
            raise ValueError(
                f"maybe_write: Next Part ID {self.nextPartId} out of writer range [{write.min_part}, {write.max_part}]"
            )

        global_index = self._global_counter
        self._global_counter += 1

        mpu_logger.info(
            "maybe_write: Writing part %s, size %s (max: %s)",
            self.nextPartId,
            len(spill_data),
            max_chunk_size,
        )
        try:
            part_info = write(self.nextPartId, bytes(spill_data))
        except Exception as e:
            mpu_logger.error(
                "maybe_write: Writer failed for part %s: %s",
                self.nextPartId,
                e,
                exc_info=True,
            )
            if lhs_keep == 0:
                self.data = spill_data + self.data
            else:
                self.data = self.left_data + spill_data + self.data
                self.left_data = bytearray()
            raise RuntimeError(
                f"Writer failed during maybe_write for part {self.nextPartId}"
            ) from e

        part_info["global_index"] = global_index
        self.parts.append(part_info)

        self.nextPartId += 1
        self.write_credits -= 1
        mpu_logger.debug(
            "maybe_write: Part written. Next ID: %s, Credits left: %s",
            self.nextPartId,
            self.write_credits,
        )

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
        split_every: int = 4,
    ) -> "dask.bag.Item":

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
    substreams: list[MPUChunk],
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
    chunks: Iterable[tuple[bytes, Any]],
    write: Optional[PartsWriter] = None,
    spill_sz: int = 0,
):
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


def flush_chunk(chunk, writer):
    return chunk.flush(writer, leftPartId=None, finalise=True)


def _finalizer_dask_op(
    data_substream: "MPUChunk",
    *,
    write: Optional["PartsWriter"] = None,
    mk_header: Any = None,
    mk_footer: Any = None,
    user_kw: Optional[dict[str, Any]] = None,
    final_task_timeout: int = 600,
):
    if user_kw is None:
        user_kw = {}

    _root = data_substream
    mpu_logger.debug("_finalizer_dask_op: Processing headers/footers...")

    try:
        hdr_bytes, footer_bytes = [
            None if op is None else op(data_substream.observed, **user_kw)
            for op in [mk_header, mk_footer]
        ]
        if footer_bytes:
            if not isinstance(footer_bytes, (bytes, bytearray)):
                raise TypeError(
                    f"mk_footer must return bytes or bytearray, got {type(footer_bytes)}"
                )
            _root.append(footer_bytes)
            mpu_logger.debug(
                "_finalizer_dask_op: Appended %s footer bytes.", len(footer_bytes)
            )
        if hdr_bytes:
            if not isinstance(hdr_bytes, (bytes, bytearray)):
                raise TypeError(
                    f"mk_header must return bytes or bytearray, got {type(hdr_bytes)}"
                )
            header_part_id = 1
            if write and header_part_id < write.min_part:
                mpu_logger.warning(
                    "Header part ID %s < writer min %s. Adjusting.",
                    header_part_id,
                    write.min_part,
                )
                header_part_id = write.min_part
            hdr = MPUChunk(header_part_id, 1)
            hdr.append(hdr_bytes)
            _root = MPUChunk.merge(hdr, _root, write=write)
            mpu_logger.debug(
                "_finalizer_dask_op: Merged %s header bytes.", len(hdr_bytes)
            )
    except Exception as e:
        mpu_logger.exception(
            "Error processing header/footer in _finalizer_dask_op: %s", e
        )
        raise RuntimeError(f"Failed during header/footer processing: {e}") from e

    if write is None:
        mpu_logger.debug(
            "_finalizer_dask_op: No writer provided, returning final MPUChunk structure."
        )
        return _root

    client: Optional[Client] = None
    try:
        client = get_client()
        mpu_logger.debug("_finalizer_dask_op: Obtained Dask client: %s", client)
    except ValueError as e:
        mpu_logger.error(
            "_finalizer_dask_op: Could not get Dask client! Cannot submit final task."
        )
        raise RuntimeError(f"Dask client not found in _finalizer_dask_op {e}") from e
    except Exception as e:
        mpu_logger.exception("Unexpected error getting Dask client: %s", e)
        raise

    def flush_final_chunk(chunk: MPUChunk, writer_obj: PartsWriter):
        worker_logger = logging.getLogger(__name__)
        worker_logger.info(
            "Executing final flush/finalise task on worker (Chunk PartId: %s)...",
            chunk.nextPartId,
        )
        try:
            worker_logger.info(
                "Calling final chunk.flush(..., finalise=True) for writer %s...",
                type(writer_obj),
            )
            _, final_result = chunk.flush(writer_obj, leftPartId=None, finalise=True)
            worker_logger.info("Final chunk.flush(..., finalise=True) completed.")
            worker_logger.info(
                "Final flush/finalise task COMPLETED on worker. Result: %s",
                final_result,
            )
            return final_result
        except Exception as worker_e:
            worker_logger.exception(
                "Error during final flush/finalise task execution on worker: %s",
                worker_e,
            )
            raise

    future: Optional[Future] = None
    try:
        mpu_logger.info(
            "_finalizer_dask_op: Submitting final flush/finalise task to Dask scheduler..."
        )
        future = client.submit(flush_final_chunk, _root, write, pure=False)
        mpu_logger.info("_finalizer_dask_op: Final task %s submitted.", future.key)
    except Exception as submit_e:
        mpu_logger.error("Failed to submit final Dask task: %s", submit_e)
        raise RuntimeError(f"Dask task submission failed: {submit_e}") from submit_e

    mpu_logger.info(
        "_finalizer_dask_op: Waiting for final task future %s to complete (timeout=%s s)...",
        future.key,
        final_task_timeout,
    )
    final_commit_result = None
    try:
        final_commit_result = future.result(timeout=final_task_timeout)
        mpu_logger.info(
            "_finalizer_dask_op: Final task future %s COMPLETED. Result received.",
            future.key,
        )

    except FuturesTimeoutError:
        mpu_logger.error(
            "_finalizer_dask_op: Timed out waiting for final task future %s after %s s.",
            future.key,
            final_task_timeout,
        )
        try:
            mpu_logger.warning("Attempting to cancel timed-out future: %s", future.key)
            future.cancel(asynchronous=True)
            mpu_logger.warning("Cancellation requested for future: %s", future.key)
        except Exception as cancel_e:
            mpu_logger.error(
                "Error requesting cancellation for future %s: %s", future.key, cancel_e
            )
        raise
    except Exception as e:
        mpu_logger.error(
            "_finalizer_dask_op: Final task future %s failed or error retrieving result: %s",
            future.key,
            e,
            exc_info=True,
        )
        try:
            if future and not future.cancelled() and future.status != "error":
                mpu_logger.warning(
                    "Attempting to cancel failed/errored future: %s", future.key
                )
                future.cancel(asynchronous=True)
        except Exception:
            pass
        raise
    finally:
        if future:
            pass

    mpu_logger.info(
        "_finalizer_dask_op: Returning final result: %s", final_commit_result
    )
    return final_commit_result


def get_mpu_kwargs(
    mk_header=None,
    mk_footer=None,
    user_kw=None,
    writes_per_chunk=1,
    spill_sz=20 * (1 << 20),
    client=None,
) -> dict:
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
    client = kw.pop("client", None)
    writer_kw = dict(kw)
    if client is not None:
        writer_kw["client"] = client
    spill_sz = kw.get("spill_sz", 20 * (1 << 20))
    if spill_sz:
        write = writer(writer_kw)
    else:
        write = None
    return mpu_write(
        chunks,
        write,
        dask_name_prefix=dask_name_prefix,
        **kw,
    )
