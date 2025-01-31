"""Tests for the Azure AzMultiPartUpload class."""

import base64
from unittest.mock import MagicMock
import pytest

pytest.importorskip("azure.storage.blob")
from odc.geo.cog._az import AzMultiPartUpload  # noqa: E402


@pytest.fixture
def azure_mpu():
    """Fixture for initializing AzMultiPartUpload."""
    account_url = "https://account_name.blob.core.windows.net"
    return AzMultiPartUpload(account_url, "container", "some.blob", None)


def test_mpu_init(azure_mpu):
    """Basic test for AzMultiPartUpload initialization."""
    assert azure_mpu.account_url == "https://account_name.blob.core.windows.net"
    assert azure_mpu.container == "container"
    assert azure_mpu.blob == "some.blob"
    assert azure_mpu.credential is None


def test_azure_multipart_upload():
    """Test the full Azure AzMultiPartUpload functionality."""
    # Mock Azure Blob SDK client structure
    mock_blob_client = MagicMock()
    mock_container_client = MagicMock()
    mock_blob_service_client = MagicMock()
    mock_blob_service_client.return_value.get_container_client.return_value = (
        mock_container_client
    )
    mock_container_client.get_blob_client.return_value = mock_blob_client

    # Simulate return values for Azure Blob SDK methods
    mock_blob_client.get_blob_properties.return_value.etag = "mock-etag"

    # Test parameters
    account_url = "https://mockaccount.blob.core.windows.net"
    container = "mock-container"
    blob = "mock-blob"
    credential = "mock-sas-token"

    # Create an instance of AzMultiPartUpload and call its methods
    azure_upload = AzMultiPartUpload(
        account_url,
        container,
        blob,
        client=mock_blob_service_client(
            account_url=account_url,
            credential=credential,
        ),
    )
    upload_id = azure_upload.initiate()
    part1 = azure_upload.write_part(1, b"first chunk of data")
    part2 = azure_upload.write_part(2, b"second chunk of data")
    etag = azure_upload.finalise([part1, part2])

    # Define block IDs
    block_id1 = base64.b64encode(b"block-1").decode("utf-8")
    block_id2 = base64.b64encode(b"block-2").decode("utf-8")

    # Verify the results
    assert upload_id == "azure-block-upload"
    assert etag == "mock-etag"

    # Verify BlobServiceClient instantiation
    mock_blob_service_client.assert_called_once_with(
        account_url=account_url, credential=credential
    )

    # Verify stage_block calls
    mock_blob_client.stage_block.assert_any_call(
        block_id=block_id1, data=b"first chunk of data"
    )
    mock_blob_client.stage_block.assert_any_call(
        block_id=block_id2, data=b"second chunk of data"
    )

    # Verify commit_block_list was called correctly
    block_list = mock_blob_client.commit_block_list.call_args[0][0]
    assert len(block_list) == 2
    assert block_list[0].id == block_id1
    assert block_list[1].id == block_id2
    mock_blob_client.commit_block_list.assert_called_once()
