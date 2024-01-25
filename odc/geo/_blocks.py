# This file is part of the Open Data Cube, see https://opendatacube.org for more information
#
# Copyright (c) 2015-2023 ODC Contributors
# SPDX-License-Identifier: Apache-2.0
"""
Working with 2d+ chunks.
"""
from typing import Any, Iterator, Mapping, Optional, Tuple

import numpy as np

from .roi import VariableSizedTiles, roi_intersect3, roi_normalise, roi_shape
from .types import Chunks2d


def _find_common_type(array_types, scalar_types):
    # TODO: don't use find_common_type as it's being removed from numpy
    return np.find_common_type(array_types, scalar_types)


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
            else _find_common_type([b.dtype for b in blocks.values()], [])
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

    def _norm_roi(self, roi):
        if roi is None:
            roi = tuple(slice(0, n) for n in self._shape)
        if not isinstance(roi, tuple):
            roi = (roi,)
        ndim = len(roi)
        if ndim == 2:
            prefix = tuple(slice(0, n) for n in self._shape[: self._axis])
            postfix = tuple(slice(0, n) for n in self._shape[self._axis + 2 :])
            roi = (*prefix, *roi, *postfix)
        elif ndim < self.ndim:
            roi = (*roi, *tuple(slice(0, n) for n in self._shape[ndim:]))
        elif ndim > self.ndim:
            raise IndexError("Too many index dimensions")

        # squeeze out non Y/X axis that were indexed with a single int
        YX = (self._axis, self._axis + 1)
        to_squeze = tuple(
            idx
            for idx, s in enumerate(roi)
            if not isinstance(s, slice) and idx not in YX
        )
        return roi_normalise(roi, self._shape), to_squeze

    def with_yx(self, src, yx):
        return (*src[: self._axis], *yx, *src[self._axis + 2 :])

    def extract(
        self,
        fill_value: Any = None,
        *,
        dtype=None,
        roi=None,
        casting="same_kind",
    ) -> np.ndarray:
        """
        Paste all blocks together into one array possibly with type coercion.
        """
        if dtype is None:
            dtype = self._dtype
            if fill_value is not None:
                # possibly upgrade to float based on fill_value
                dtype = _find_common_type([dtype], [np.min_scalar_type(fill_value)])
        else:
            dtype = np.dtype(dtype)

        if fill_value is None:
            fill_value = dtype.type("nan" if np.issubdtype(dtype, np.floating) else 0)

        roi, squeeze_axis = self._norm_roi(roi)
        assert len(roi) == self.ndim

        yx_roi = roi[self._axis : self._axis + 2]
        everything = tuple(slice(None) for _ in range(self.ndim))

        xx = np.full(roi_shape(roi), fill_value, dtype=dtype)

        for idx, block in self._blocks.items():
            yx_roi_b = self._tiles[idx]  # area covered by this block
            s_roi, d_roi, _ = roi_intersect3(yx_roi_b, yx_roi)
            s_roi = self.with_yx(roi, s_roi)
            d_roi = self.with_yx(everything, d_roi)
            np.copyto(xx[d_roi], block[s_roi], casting=casting)

        if squeeze_axis:
            xx = np.squeeze(xx, axis=squeeze_axis)
        return xx

    def __getitem__(self, roi) -> np.ndarray:
        return self.extract(roi=roi)

    def planes_yx(self, yx_roi=None) -> Iterator[Any]:
        a = self._axis
        if yx_roi is None:
            yx = np.s_[:, :]
        else:
            ry, rx = yx_roi
            yx = (ry, rx)

        for idx in np.ndindex(self.shape[:a] + self.shape[a + 2 :]):
            yield (*idx[:a], *yx, *idx[a:])
