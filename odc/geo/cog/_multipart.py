"""
Multipart upload interface.

Defines the `MultiPartUploadBase` class for implementing multipart upload functionality.
This interface standardises methods for initiating, uploading, and finalising
multipart uploads across storage backends.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Union

from dask.delayed import Delayed

from ._mpu import get_mpu_kwargs, mpu_upload

if TYPE_CHECKING:
    # pylint: disable=import-outside-toplevel,import-error
    import dask.bag


class MultiPartUploadBase(ABC):
    """Abstract base class for multipart upload."""

    @abstractmethod
    def initiate(self, **kwargs) -> str:
        """Initiate a multipart upload and return an identifier."""

    @abstractmethod
    def write_part(self, part: int, data: bytes) -> dict[str, Any]:
        """Upload a single part."""

    @abstractmethod
    def finalise(self, parts: list[dict[str, Any]]) -> str:
        """Finalise the upload with a list of parts."""

    @abstractmethod
    def cancel(self, other: str = ""):
        """Cancel the multipart upload."""

    @property
    @abstractmethod
    def url(self) -> str:
        """Return the URL of the upload target."""

    @property
    @abstractmethod
    def started(self) -> bool:
        """Check if the multipart upload has been initiated."""

    @abstractmethod
    def writer(self, kw: dict[str, Any], *, client: Any = None) -> Any:
        """
        Return a Dask-compatible writer for multipart uploads.

        :param kw: Additional parameters for the writer.
        :param client: Dask client for distributed execution.
        """

    @abstractmethod
    def dask_name_prefix(self) -> str:
        """Return the dask name prefix specific to the backend."""

    def upload(
        self,
        chunks: Union["dask.bag.Bag", list["dask.bag.Bag"]],
        *,
        mk_header: Any = None,
        mk_footer: Any = None,
        user_kw: Union[dict[str, Any], None] = None,
        writes_per_chunk: int = 1,
        spill_sz: int = 20 * (1 << 20),
        client: Any = None,
    ) -> Delayed:
        """High-level upload that calls mpu_upload under the hood."""
        kwargs = get_mpu_kwargs(
            mk_header=mk_header,
            mk_footer=mk_footer,
            user_kw=user_kw,
            writes_per_chunk=writes_per_chunk,
            spill_sz=spill_sz,
            client=client,
        )
        return mpu_upload(
            chunks,
            writer=self.writer,
            dask_name_prefix=self.dask_name_prefix(),
            **kwargs,
        )
