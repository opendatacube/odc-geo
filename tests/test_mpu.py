from hashlib import md5
from typing import Any, Dict, List, Tuple

import numpy as np

from odc.geo.cog._mpu import MPUChunk, PartsWriter

FakeWriteResult = Tuple[int, bytearray, Dict[str, Any]]
# pylint: disable=unbalanced-tuple-unpacking


def etag(data):
    return f'"{md5(data).hexdigest()}"'


def _mb(x: float) -> int:
    return int(x * (1 << 20))


def mk_fake_parts_writer(dst: List[FakeWriteResult]) -> PartsWriter:
    def write(part: int, data: bytearray) -> Dict[str, Any]:
        assert 1 <= part <= 10_000
        ww = {"PartNumber": part, "ETag": f'"{etag(data)}"'}
        dst.append((part, data, ww))
        return ww

    return write


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


def _data(parts_written: List[FakeWriteResult]) -> bytes:
    return b"".join(data for _, data, _ in sorted(parts_written, key=lambda x: x[0]))


def _parts(parts_written: List[FakeWriteResult]) -> List[Dict[str, Any]]:
    return [part for _, _, part in sorted(parts_written, key=lambda x: x[0])]


def test_s3_mpu_merge_small() -> None:
    # Test situation where parts get joined and get written eventually in one chunk
    parts_written: List[FakeWriteResult] = []
    write = mk_fake_parts_writer(parts_written)

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
    assert bc.left_data is None
    assert bc.data == (db + dc)

    # a + (b + c)
    abc = MPUChunk.merge(a, bc, write)

    assert abc.is_final is True
    assert abc.started_write is False
    assert abc.data == data
    assert abc.left_data is None
    assert abc.nextPartId == a.nextPartId
    assert abc.write_credits == (a.write_credits + b.write_credits + c.write_credits)
    assert len(abc.observed) == 3
    assert abc.observed == [(len(da), "a1"), (len(db), "b1"), (len(dc), "c1")]

    assert len(parts_written) == 0

    assert abc.final_flush(write) == len(data)
    assert len(abc.parts) == 1
    assert len(parts_written) == 1
    pid, _data, part = parts_written[0]
    assert _data == data
    assert pid == a.nextPartId
    assert part["PartNumber"] == pid
    assert abc.parts[0] == part


def test_mpu_multi_writes() -> None:
    parts_written: List[FakeWriteResult] = []
    write = mk_fake_parts_writer(parts_written)

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
    assert len(parts_written) == 1
    ab = MPUChunk.merge(a, b, write)
    assert len(parts_written) == 2
    assert len(ab.parts) == 2
    assert ab.started_write is True
    assert ab.is_final is False
    assert ab.data == (db1 + db2)[-_mb(5) :]
    assert ab.left_data is None

    abc = MPUChunk.merge(ab, c, write)
    assert len(abc.parts) == 2
    assert abc.is_final is True
    assert abc.observed == [
        (len(da1), "a1"),
        (len(db1), "b1"),
        (len(db2), "b2"),
        (len(dc1), "c1"),
    ]

    assert abc.final_flush(write, None) > 0
    assert len(abc.parts) == 3
    assert _parts(parts_written) == abc.parts
    assert _data(parts_written) == data


def test_mpu_left_data() -> None:
    parts_written: List[FakeWriteResult] = []
    write = mk_fake_parts_writer(parts_written)

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
    assert c.left_data is None

    b.append(db1, "b1")
    a.append(da1, "a1")

    bc = MPUChunk.merge(b, c, write)
    assert bc.started_write is True
    assert bc.left_data is not None
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

    assert abc.final_flush(write) > 0
    assert abc.nextPartId == 202
    assert abc.write_credits == 98
    assert len(abc.parts) == 3
    assert _data(parts_written) == data
    assert _parts(parts_written) == abc.parts


def test_mpu_misc() -> None:
    parts_written: List[FakeWriteResult] = []
    write = mk_fake_parts_writer(parts_written)

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
    assert len(parts_written) == 0
    assert ab.nextPartId == 1
    assert ab.write_credits == 11

    assert ab.final_flush(write) > 0
    assert len(ab.parts) == 1
    assert _data(parts_written) == data
    assert _parts(parts_written) == ab.parts
