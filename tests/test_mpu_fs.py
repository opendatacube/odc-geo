from __future__ import annotations

import pickle
from pathlib import Path
from uuid import uuid4

import pytest

from odc.geo.cog._mpu_fs import MPUFileSink

# pylint: disable=protected-access


def slurp(path: Path) -> bytes:
    with path.open("rb") as f:
        return f.read()


@pytest.mark.parametrize("parts_base", [f"parts-{uuid4().hex[:8]}", None])
@pytest.mark.parametrize("num_parts", [1, 2, 3, 10])
def test_filesink(tmp_path: Path, parts_base: str | None, num_parts: int):
    dst = Path(tmp_path / f"{uuid4().hex}.bin")
    assert dst.exists() is False
    if parts_base is not None:
        parts_base = str(tmp_path / parts_base)

    sink = MPUFileSink(dst, parts_base)
    assert sink.__dask_tokenize__() == sink.__dask_tokenize__()
    assert sink.max_part > sink.min_part
    assert sink.max_write_sz > sink.min_write_sz

    # simulate distributed writing
    sink2 = pickle.loads(pickle.dumps(sink))

    chunks = [
        (idx, f"part: {idx}\n...{idx}\n".encode("utf8"))
        for idx in range(1, num_parts + 1)
    ]
    expected_content = b"".join(data for _, data in chunks)

    parts = []
    for idx, data in chunks:
        _sink = sink if idx % 2 == 0 else sink2
        parts.append(_sink(idx, data))

    for (idx, data), part in zip(chunks, parts):
        assert part["PartNumber"] == idx
        assert part["Size"] == len(data)
        part_path = Path(part["Path"])
        assert part_path.exists()
        assert slurp(part_path) == data

    assert sink._parts_dir.exists()
    assert dst.exists() is False

    assert sink.finalise(parts) == dst

    assert dst.exists()
    assert sink._parts_dir.exists() is False
    assert any(Path(p["Path"]).exists() for p in parts) is False

    assert slurp(dst) == expected_content
