import base64
from typing import Any

from ._multipart import MultiPartUploadBase


class AzureLimits:
    """
    Common Azure writer settings.
    """

    @property
    def min_write_sz(self) -> int:
        # Azure minimum write size for blocks (default is 4 MiB)
        return 4 * (1 << 20)

    @property
    def max_write_sz(self) -> int:
        # Azure maximum write size for blocks (default is 100 MiB)
        return 100 * (1 << 20)

    @property
    def min_part(self) -> int:
        return 1

    @property
    def max_part(self) -> int:
        # Azure supports up to 50,000 blocks per blob
        return 50_000


class AzMultiPartUpload(AzureLimits, MultiPartUploadBase):
    """
    Azure Blob Storage multipart upload.
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        account_url: str,
        container: str,
        blob: str,
        credential: Any = None,
        client: Any = None,
    ):
        """
        Initialise Azure multipart upload.

        :param account_url: URL of the Azure storage account.
        :param container: Name of the container.
        :param blob: Name of the blob.
        :param credential: Authentication credentials (e.g., SAS token or key).
        """
        self.account_url = account_url
        self.container = container
        self.blob = blob
        self.credential = credential

        # Initialise Azure Blob service client
        # pylint: disable=import-outside-toplevel,import-error
        from azure.storage.blob import BlobServiceClient

        self.blob_service_client = client or BlobServiceClient(
            account_url=account_url, credential=credential
        )
        self.container_client = self.blob_service_client.get_container_client(container)
        self.blob_client = self.container_client.get_blob_client(blob)

        self.block_ids: list[str] = []

    def initiate(self, **kwargs) -> str:
        """
        Initialise the upload. No-op for Azure.
        """
        return "azure-block-upload"

    def write_part(self, part: int, data: bytes) -> dict[str, Any]:
        """
        Stage a block in Azure.

        :param part: Part number (unique).
        :param data: Data for this part.
        :return: A dictionary containing part information.
        """
        block_id = base64.b64encode(f"block-{part}".encode()).decode()
        self.blob_client.stage_block(block_id=block_id, data=data)
        self.block_ids.append(block_id)
        return {"PartNumber": part, "BlockId": block_id}

    def finalise(self, parts: list[dict[str, Any]]) -> str:
        """
        Commit the block list to finalise the upload.

        :param parts: List of uploaded parts metadata.
        :return: The ETag of the finalised blob.
        """
        # pylint: disable=import-outside-toplevel,import-error
        from azure.storage.blob import BlobBlock

        block_list = [BlobBlock(block_id=part["BlockId"]) for part in parts]
        self.blob_client.commit_block_list(block_list)
        return self.blob_client.get_blob_properties().etag

    def cancel(self, other: str = ""):
        """
        Cancel the upload by clearing the block list.
        """
        assert other == ""
        self.block_ids.clear()

    @property
    def url(self) -> str:
        """
        Get the Azure blob URL.

        :return: The full URL of the blob.
        """
        return self.blob_client.url

    @property
    def started(self) -> bool:
        """
        Check if any blocks have been staged.

        :return: True if blocks have been staged, False otherwise.
        """
        return bool(self.block_ids)

    def writer(self, kw: dict[str, Any], *, client: Any = None):
        """
        Return a stateless writer compatible with Dask.
        """
        return DelayedAzureWriter(self, kw)

    def dask_name_prefix(self) -> str:
        """Return the Dask name prefix for Azure."""
        return "azure-finalise"


class DelayedAzureWriter(AzureLimits):
    """
    Dask-compatible writer for Azure Blob Storage multipart uploads.
    """

    def __init__(self, mpu: AzMultiPartUpload, kw: dict[str, Any]):
        """
        Initialise the Azure writer.

        :param mpu: AzMultiPartUpload instance.
        :param kw: Additional parameters for the writer.
        """
        self.mpu = mpu
        self.kw = kw  # Additional metadata like ContentType

    def __call__(self, part: int, data: bytes) -> dict[str, Any]:
        """
        Write a single part to Azure Blob Storage.

        :param part: Part number.
        :param data: Chunk data.
        :return: Metadata for the written part.
        """
        return self.mpu.write_part(part, data)

    def finalise(self, parts: list[dict[str, Any]]) -> str:
        """
        Finalise the upload by committing the block list.

        :param parts: List of uploaded parts metadata.
        :return: ETag of the finalised blob.
        """
        return self.mpu.finalise(parts)
