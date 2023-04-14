# This file is part of the Open Data Cube, see https://opendatacube.org for more information
#
# Copyright (c) 2015-2023 ODC Contributors
# SPDX-License-Identifier: Apache-2.0
"""
Working with 2d+ chunks.
"""
from typing import Any, Mapping, Optional, Tuple

import numpy as np

from .roi import VariableSizedTiles
from .types import Chunks2d


class BlockAssembler:
    """
    Construct contigous 2+ dim array from incomplete set of YX blocks.
    """

    def __init__(
        self,
        blocks: Mapping[Tuple[int, int], np.ndarray],
        chunks: Chunks2d,
        axis: int = 0,
    ) -> None:
        self._shape = BlockAssembler._verify_shape(blocks, chunks, axis=axis)
        self._dtype = (
            np.dtype("float32")
            if len(blocks) == 0
            else np.find_common_type((b.dtype for b in blocks.values()), [])
        )
        self._axis = axis
        self._blocks = blocks
        self._tiles = VariableSizedTiles(chunks)
        assert self._tiles.base.shape == self._shape[axis : axis + 2]

    @staticmethod
    def _verify_shape(
        blocks: Mapping[Tuple[int, int], np.ndarray],
        chunks: Chunks2d,
        axis: int = 0,
    ) -> Tuple[int, ...]:
        """
        compute total shape including prefix and postfix dimensions.
        """
        state: Optional[Tuple[int, Tuple[int, ...], Tuple[int, ...]]] = None
        chy, chx = chunks
        for (iy, ix), b in blocks.items():
            if state is None:
                if b.ndim < axis + 2:
                    raise ValueError(
                        f"Too few dimensions for `axis={axis}` ({b.ndim} < {axis+2})"
                    )
                state = (b.ndim, b.shape[:axis], b.shape[axis + 2 :])

            if state[0] != b.ndim or state[1:] != (b.shape[:axis], b.shape[axis + 2 :]):
                raise ValueError("Extra dimensions must be the same across all blocks")

            yx_shape = b.shape[axis : axis + 2]
            if yx_shape != (chy[iy], chx[ix]):
                raise ValueError(f"Mismatched block YxX shape at [{iy}, {ix}]")

        ny, nx = sum(chy), sum(chx)
        if state is None:
            return (ny, nx)
        _, prefix, postfix = state
        return (*prefix, ny, nx, *postfix)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def ndim(self) -> int:
        return len(self._shape)

    def extract(
        self,
        fill_value: Any = None,
        dtype=None,
        casting="same_kind",
    ) -> np.ndarray:
        """
        Paste all blocks together into one array possibly with type coercion.
        """
        if dtype is None:
            dtype = self._dtype
            if fill_value is not None:
                # possibly upgrade to float based on fill_value
                dtype = np.find_common_type([dtype], [np.min_scalar_type(fill_value)])

        if fill_value is None:
            fill_value = dtype.type("nan" if np.issubdtype(dtype, np.floating) else 0)

        prefix = tuple([slice(None)] * self._axis)
        postfix = tuple([slice(None)] * (self.ndim - self._axis - 2))

        xx = np.full(self.shape, fill_value, dtype=dtype)
        for idx, b in self._blocks.items():
            roi = (*prefix, *self._tiles[idx], *postfix)
            np.copyto(xx[roi], b, casting=casting)

        return xx
