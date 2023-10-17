"""
MPU file sink
"""
from __future__ import annotations

import mmap
from pathlib import Path
from typing import Any

from ._mpu import SomeData


class MPUFileSink:
    """
    Writes chunks to disk, then gathers them into final destination.

    Assumes shared filesystem across Dask cluster.
    """

    def __init__(
        self,
        dst: str | Path,
        parts_base: str | Path | None = None,
        **limits,
    ) -> None:
        dst = Path(dst)
        if parts_base is None:
            parts_dir = dst.parent / f".{dst.name}.parts"
        else:
            parts_dir = Path(parts_base) / f".{dst.name}.parts"

        self._dst = dst
        self._parts_dir = parts_dir
        self._limits = limits

    @property
    def min_write_sz(self) -> int:
        return self._limits.get("min_write_sz", 4096)

    @property
    def max_write_sz(self) -> int:
        return self._limits.get("min_write_sz", 5 * (1 << 30))

    @property
    def min_part(self) -> int:
        return self._limits.get("min_part", 1)

    @property
    def max_part(self) -> int:
        return self._limits.get("min_part", 10_000)

    def _ensure_dst_file(self, part: int) -> Path:
        parts_dir = self._parts_dir
        if not parts_dir.exists():
            try:
                parts_dir.mkdir(parents=True)
            except FileExistsError:
                pass
        return parts_dir / f"p{part:04d}.bin"

    def __call__(self, part: int, data: SomeData) -> dict[str, Any]:
        dst = self._ensure_dst_file(part)
        with open(dst, "wb") as f:
            nb = f.write(data)
            assert nb == len(data)

        return {"PartNumber": part, "Path": str(dst), "Size": len(data)}

    def finalise(self, parts: list[dict[str, Any]], keep_parts: bool = False) -> Any:
        assert len(parts) > 0
        dst = self._dst
        first, *rest = parts
        p1 = Path(first["Path"])
        p1.rename(dst)

        with open(dst, "ab") as f:
            for part in rest:
                src_path = Path(part["Path"])
                with src_path.open("rb") as src:
                    with mmap.mmap(
                        src.fileno(), 0, access=mmap.ACCESS_READ
                    ) as src_bytes:
                        f.write(src_bytes)

                if not keep_parts:
                    src_path.unlink()

        if not keep_parts:
            self._parts_dir.rmdir()

        return dst

    def __dask_tokenize__(self):
        return (self._dst, self._parts_dir)
