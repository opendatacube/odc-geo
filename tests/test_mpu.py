from hashlib import md5
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest

from odc.geo.cog._mpu import MPUChunk, SomeData, mpu_write

FakeWriteResult = Tuple[int, SomeData, Dict[str, Any]]
# pylint: disable=unbalanced-tuple-unpacking,redefined-outer-name,import-outside-toplevel


class FakeWriter:
    """
    Fake multi-part upload writer.
    """

    def __init__(self, **limits) -> None:
        self._dst: List[FakeWriteResult] = []
        self._limits = limits

    def __call__(self, part: int, data: SomeData) -> Dict[str, Any]:
        assert 1 <= part <= 10_000
        ww = {"PartNumber": part, "ETag": f'"{etag(data)}"'}
        self._dst.append((part, data, ww))
        return ww

    def finalise(self, parts: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {"Parts": parts, "Writer": self}

    @property
    def min_write_sz(self) -> int:
        return self._limits.get("min_write_sz", 5 * (1 << 20))

    @property
    def max_write_sz(self) -> int:
        return self._limits.get("max_write_sz", 5 * (1 << 30))

    @property
    def min_part(self) -> int:
        return self._limits.get("min_part", 1)

    @property
    def max_part(self) -> int:
        return self._limits.get("max_part", 10_000)

    @property
    def raw_parts(self) -> List[FakeWriteResult]:
        return self._dst

    @property
    def data(self) -> bytes:
        return b"".join(data for _, data, _ in sorted(self._dst, key=lambda x: x[0]))

    @property
    def parts(self) -> List[Dict[str, Any]]:
        return [part for _, _, part in sorted(self._dst, key=lambda x: x[0])]

    def reset(self) -> "FakeWriter":
        self._dst = []
        return self


def etag(data):
    return f'"{md5(data).hexdigest()}"'


def _mb(x: float) -> int:
    return int(x * (1 << 20))


def _mk_fake_data(sz: int) -> bytes:
    return np.random.bytes(sz)


def _split(data: bytes, offsets=List[int]) -> Tuple[bytes, ...]:
    parts = []

    for sz in offsets:
        parts.append(data[:sz])
        data = data[sz:]
    if data:
        parts.append(data)

    return tuple(parts)


@pytest.fixture(scope="function")
def write(request):
    limits = getattr(request, "param", {})
    yield FakeWriter(**limits)


def test_s3_mpu_merge_small(write: FakeWriter) -> None:
    # Test situation where parts get joined and get written eventually in one chunk
    data = _mk_fake_data(100)
    da, db, dc = _split(data, [10, 20])

    a = MPUChunk(1, 10)
    b = MPUChunk(12, 2)
    c = MPUChunk(14, 3, is_final=True)
    assert a.is_final is False
    assert b.is_final is False
    assert c.is_final is True
    assert a.started_write is False

    assert a.maybe_write(write, _mb(5)) == 0
    assert a.started_write is False

    b.append(db, "b1")
    c.append(dc, "c1")
    a.append(da, "a1")

    # b + c
    bc = MPUChunk.merge(b, c, write)
    assert bc.is_final is True
    assert bc.nextPartId == b.nextPartId
    assert bc.write_credits == (b.write_credits + c.write_credits)
    assert bc.started_write is False
    assert bc.left_data == b""
    assert bc.data == (db + dc)

    # a + (b + c)
    abc = MPUChunk.merge(a, bc, write)

    assert abc.is_final is True
    assert abc.started_write is False
    assert abc.data == data
    assert abc.left_data == b""
    assert abc.nextPartId == a.nextPartId
    assert abc.write_credits == (a.write_credits + b.write_credits + c.write_credits)
    assert len(abc.observed) == 3
    assert abc.observed == [(len(da), "a1"), (len(db), "b1"), (len(dc), "c1")]

    assert len(write.parts) == 0

    assert abc.flush_rhs(write) == len(data)
    assert len(abc.parts) == 1
    assert len(write.parts) == 1
    pid, _data, part = write.raw_parts[0]
    assert _data == data
    assert pid == a.nextPartId
    assert part["PartNumber"] == pid
    assert abc.parts[0] == part


def test_mpu_multi_writes(write: FakeWriter) -> None:
    data = _mk_fake_data(_mb(20))
    da1, db1, db2, dc1 = _split(
        data,
        [
            _mb(5.2),
            _mb(7),
            _mb(6),
        ],
    )
    a = MPUChunk(1, 10)
    b = MPUChunk(10, 2)
    c = MPUChunk(12, 3, is_final=True)
    assert (a.is_final, b.is_final, c.is_final) == (False, False, True)

    b.append(db1, "b1")
    # have enough data to write, but not enough to have left-over
    assert b.maybe_write(write, _mb(6)) == 0
    b.append(db2, "b2")
    assert b.observed == [(len(db1), "b1"), (len(db2), "b2")]
    # have enough data to write and have left-over still
    #  (7 + 6) - 6 > 5
    assert b.maybe_write(write, _mb(6)) > 0
    assert b.started_write is True
    assert len(b.data) == _mb(5)
    assert b.write_credits == 1
    assert b.nextPartId == 11

    a.append(da1, "a1")
    assert a.maybe_write(write, _mb(6)) == 0

    c.append(dc1, "c1")
    assert c.maybe_write(write, _mb(6)) == 0

    # Should flush a
    assert len(write.parts) == 1
    ab = MPUChunk.merge(a, b, write)
    assert len(write.parts) == 2
    assert len(ab.parts) == 2
    assert ab.started_write is True
    assert ab.is_final is False
    assert ab.data == (db1 + db2)[-_mb(5) :]
    assert ab.left_data == b""

    abc = MPUChunk.merge(ab, c, write)
    assert len(abc.parts) == 2
    assert abc.is_final is True
    assert abc.observed == [
        (len(da1), "a1"),
        (len(db1), "b1"),
        (len(db2), "b2"),
        (len(dc1), "c1"),
    ]

    assert abc.flush_rhs(write, None) > 0
    assert len(abc.parts) == 3
    assert write.parts == abc.parts
    assert write.data == data


def test_mpu_left_data(write: FakeWriter) -> None:
    data = _mk_fake_data(_mb(3 + 2 + (6 + 5.2)))
    da1, db1, dc1, dc2 = _split(
        data,
        [
            _mb(3),
            _mb(2),
            _mb(6),
        ],
    )
    a = MPUChunk(1, 100)
    b = MPUChunk(100, 100)
    c = MPUChunk(200, 100)

    c.append(dc1, "c1")
    c.append(dc2, "c2")
    assert c.maybe_write(write, _mb(6)) > 0
    assert c.started_write is True
    assert len(c.parts) == 1
    assert c.left_data == b""

    b.append(db1, "b1")
    a.append(da1, "a1")

    bc = MPUChunk.merge(b, c, write)
    assert bc.started_write is True
    assert len(bc.left_data) > 0
    assert bc.nextPartId == 201
    assert bc.write_credits == 99

    # Expect (a.data + bc.left_data) to be written to PartId=1
    abc = MPUChunk.merge(a, bc, write)
    assert len(abc.parts) == 2
    assert abc.parts[0]["PartNumber"] == 1
    assert abc.nextPartId == 201
    assert abc.write_credits == 99
    assert abc.is_final is False

    assert abc.observed == [
        (len(da1), "a1"),
        (len(db1), "b1"),
        (len(dc1), "c1"),
        (len(dc2), "c2"),
    ]

    assert abc.flush_rhs(write) > 0
    assert abc.nextPartId == 202
    assert abc.write_credits == 98
    assert len(abc.parts) == 3
    assert write.data == data
    assert write.parts == abc.parts


def test_mpu_misc(write: FakeWriter) -> None:
    a = MPUChunk(1, 10)
    b = MPUChunk(10, 1)

    data = _mk_fake_data(_mb(3 + (6 + 7)))
    da1, db1, db2 = _split(
        data,
        [_mb(3), _mb(6), _mb(7)],
    )
    b.append(db1, "b1")
    b.append(db2, "b2")

    # not enough credits to write
    assert b.maybe_write(write, _mb(5)) == 0

    a.append(da1, "a1")

    ab = MPUChunk.merge(a, b, write)
    assert ab.started_write is False
    assert len(write.parts) == 0
    assert ab.nextPartId == 1
    assert ab.write_credits == 11

    assert ab.flush_rhs(write) > 0
    assert len(ab.parts) == 1
    assert write.data == data
    assert write.parts == ab.parts

    assert " final" in repr(
        MPUChunk(
            1,
            1,
            bytearray(),
            parts=[{}],
            observed=[(0, None)],
            is_final=True,
        )
    )


def test_lhs_keep(write: FakeWriter) -> None:
    def mpu(lhs_keep: int, extra_bytes: int, is_final: bool) -> Tuple[MPUChunk, bytes]:
        a = MPUChunk(10, 100, lhs_keep=lhs_keep, is_final=is_final)
        a.append(data := _mk_fake_data(lhs_keep + extra_bytes))
        return a, data

    lhs_keep = write.min_write_sz

    a, data = mpu(lhs_keep, write.min_write_sz // 2, is_final=False)
    assert a.left_data == b""
    assert a.data == data
    assert not a.started_write

    # should move all data .left_data
    assert a.flush_rhs(write) == 0
    assert a.data == b""
    assert a.left_data == data

    a, data = mpu(lhs_keep, write.min_write_sz // 2, is_final=True)
    assert a.left_data == b""
    assert a.data == data
    assert not a.started_write

    # should not spill: not enough due to lhs
    assert a.maybe_write(write, write.min_write_sz) == 0
    assert not a.started_write

    assert a.flush_rhs(write) == len(data) - lhs_keep
    assert a.started_write
    assert a.data == b""
    assert a.left_data == data[:lhs_keep]


@pytest.mark.parametrize("spill_sz", [10, 11, 10_000, 0])
def test_dask_parts(spill_sz: int):
    pytest.importorskip("dask")
    from dask import bag
    from dask.delayed import Delayed

    write = FakeWriter(min_write_sz=10)
    assert write.min_write_sz == 10
    assert spill_sz == 0 or spill_sz >= write.min_write_sz

    data = _mk_fake_data(103)
    parts = _split(data, [30, 50])
    parts = [(bb, idx) for idx, bb in enumerate(parts)]

    chunks = bag.from_sequence(parts, npartitions=len(parts))

    FOOTER = b"\n------------\n"
    HEADER = b"Header\n\n"

    def mk_footer(observed, expected_sz: int = 0):
        # total sum of observed elements should match data length
        assert sum(sz for sz, _ in observed) == expected_sz
        return FOOTER

    def mk_header(observed, expected_sz: int = 0):
        # total sum of observed elements should match data length
        assert sum(sz for sz, _ in observed) == expected_sz
        return HEADER

    # check single bag case with writer
    rr = mpu_write(chunks, write, spill_sz=spill_sz)
    assert isinstance(rr, Delayed)
    rr = rr.compute()
    assert isinstance(rr, dict)
    assert rr["Writer"] is write
    assert rr["Parts"] == write.parts
    assert write.data == data

    # check substreams + footer + header
    rr = mpu_write(
        [chunks] * 3,
        write=write.reset(),
        spill_sz=spill_sz,
        mk_footer=mk_footer,
        mk_header=mk_header,
        user_kw={"expected_sz": len(data) * 3},
    )
    assert isinstance(rr, Delayed)
    rr = rr.compute()
    assert isinstance(rr, dict)
    assert rr["Writer"] is write
    assert rr["Parts"] == write.parts
    assert write.data == b"".join([HEADER, data * 3, FOOTER])

    # check no writer case
    rr = mpu_write(chunks)
    assert isinstance(rr, Delayed)
    rr = rr.compute()
    assert isinstance(rr, MPUChunk)
    assert rr.is_final
    assert rr.lhs_keep == 0
    assert rr.left_data == bytearray()
    assert rr.data == data
    assert rr.started_write is False
    assert rr.parts == []
