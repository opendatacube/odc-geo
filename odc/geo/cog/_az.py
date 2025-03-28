import base64
import sys
import threading
import time
import uuid
from typing import Any
from dask.distributed import get_client, Lock
from dask.base import tokenize

from azure.core.exceptions import AzureError, HttpResponseError
from azure.storage.blob import BlobBlock, BlobServiceClient, ContentSettings
from ._multipart import MultiPartUploadBase

import logging

logger = logging.getLogger("odc.geo.cog._az")
# Set default level (can be overridden by application config)
logger.setLevel(logging.INFO)
# Avoid adding handlers multiple times if this module is reloaded
if not logger.hasHandlers():
    # Configure console handler
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s (%(funcName)s): %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    # Prevent logs from propagating to the root logger if handlers are added here
    logger.propagate = False

# Example: Increase log level for debugging
# logger.setLevel(logging.DEBUG)


class AzureLimits:
    """
    Common Azure writer settings.
    See: https://learn.microsoft.com/en-us/rest/api/storageservices/put-block#remarks
    See: https://learn.microsoft.com/en-us/rest/api/storageservices/put-block-list#remarks
    """

    @property
    def min_write_sz(self) -> int:
        # While Azure allows smaller blocks, larger blocks are more efficient.
        # Let _mpu handle buffering; min practical size here isn't strictly enforced by API.
        # Setting to a reasonable value like 4MiB helps _mpu manage flushes.
        return 4 * (1 << 20)  # 4 MiB

    @property
    def max_write_sz(self) -> int:
        # Max size for Put Block is 100 MiB for most service versions
        # Newer versions support 4000 MiB, but stick to 100 MiB for compatibility
        return 100 * (1 << 20)  # 100 MiB

    @property
    def min_part(self) -> int:
        # Block IDs are user-defined strings, no numeric requirement like S3 parts
        return 1  # Used conceptually by _mpu

    @property
    def max_part(self) -> int:
        # Max number of blocks per blob is 50,000
        return 50_000


class AzMultiPartUpload(AzureLimits, MultiPartUploadBase):
    """
    Azure Blob Storage multipart upload using Put Block and Put Block List.

    Manages staging blocks and committing them. Designed for use with Dask.
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        account_url: str,
        container: str,
        blob: str,
        credential: Any = None,
        client: Any = None,
        overwrite: bool = True,
    ):
        # Input validation
        if not all([account_url, container, blob]):
            raise ValueError("account_url, container, and blob must be provided")

        self.account_url = account_url
        self.container = container
        self.blob = blob
        self.credential = (
            credential  # Allow None for SAS tokens etc. handled by BlobServiceClient
        )
        self.overwrite = overwrite

        # Unique identifier for this specific upload attempt's blocks
        # Ensures block IDs from different upload attempts to the same blob don't clash
        # before the final commit.
        self.upload_prefix = (
            f"odc-cog-{int(time.time() * 1000)}-{uuid.uuid4().hex[:12]}"
        )
        logger.info(
            "Initialized AzMultiPartUpload for %s/%s with prefix %s",
            container,
            blob,
            self.upload_prefix,
        )

        # Azure SDK Clients
        # Lazily initialized in _get_clients_if_needed to be pickle-friendly for Dask
        self._blob_service_client = None
        self._container_client = None
        self._blob_client = None
        self._clients_initialized = False

        # Dask Synchronization
        # Uses Dask client if available, otherwise assumes single-process execution.
        self._dask_client = None
        try:
            self._dask_client = get_client() if client is None else client
        except ValueError:  # No global client found
            logger.warning(
                "No Dask client found, proceeding in single-threaded mode for locks/variables."
            )
            self._dask_client = None  # Explicitly set to None

        # Lock for ensuring blob deletion (if overwrite=True) happens only once.
        # Variable is removed for simplification.
        lock_var_name = f"odc-az-init-{container}-{blob}-{tokenize(account_url)}"
        self._init_lock = None
        if self._dask_client:
            logger.debug("Using Dask Lock: %s-lock", lock_var_name)
            self._init_lock = Lock(f"{lock_var_name}-lock", client=self._dask_client)
        else:
            # Basic threading lock for non-dask environment
            logger.debug("Using fallback threading.Lock for init coordination.")
            self._init_lock = threading.Lock()  # Keep the lock for non-dask case

        self._initialized_on_scheduler = False  # Flag still used for non-dask check

    def _get_clients_if_needed(self):
        """Initialize Azure SDK clients if they haven't been already."""
        if not self._clients_initialized:
            logger.debug("Initializing Azure clients for %s", self.account_url)

            # Allow credential=None if account_url includes SAS token
            self._blob_service_client = BlobServiceClient(
                account_url=self.account_url, credential=self.credential
            )
            self._container_client = self._blob_service_client.get_container_client(
                self.container
            )
            self._blob_client = self._container_client.get_blob_client(self.blob)
            self._clients_initialized = True
            logger.debug(
                "Azure clients initialized for %s/%s", self.container, self.blob
            )

    def _get_block_id(self, part: int) -> str:
        """
        Generates a unique, Base64 encoded block ID for a given part number.
        Azure requires block IDs to be Base64 encoded strings, max 64 bytes.
        Using a prefix ensures uniqueness across different upload attempts.
        """
        # Combine upload prefix and part number for a unique ID for this upload session
        raw_id = f"{self.upload_prefix}-p{part:06d}"
        # Encode to bytes, then Base64 encode
        b64_bytes = base64.b64encode(raw_id.encode("utf-8"))
        b64_string = b64_bytes.decode("utf-8")
        # Basic validation (length check)
        if len(b64_bytes) > 64:
            # This should realistically never happen with our format
            raise ValueError(
                f"Generated Block ID too long: {len(b64_bytes)} > 64 bytes"
            )
        return b64_string

    def initiate(self, **kwargs) -> str:
        """
        Initiates the upload process. If overwrite=True, deletes the blob
        using a Dask lock to ensure it happens only once before staging begins.
        Returns a conceptual upload ID.
        """
        logger.info(
            "Initiating upload for %s/%s. Overwrite=%s",
            self.container,
            self.blob,
            self.overwrite,
        )
        self._ensure_clients_initialized()  # Ensure clients are ready for potential delete

        upload_id_string = f"azure-block-upload-{self.upload_prefix}"

        if not self.overwrite:
            logger.debug("Overwrite is False, skipping delete step in initiate.")
            return upload_id_string  # Return conceptual ID

        # Perform Deletion using Lock
        if self._dask_client and self._init_lock:
            # Dask Environment
            lock_timeout = 60
            logger.debug(
                "Attempting acquire Dask init lock for delete with timeout=%ss...",
                lock_timeout,
            )
            acquired_lock = self._init_lock.acquire(timeout=lock_timeout)
            if not acquired_lock:
                logger.error("Failed acquire Dask init lock for delete.")
                raise TimeoutError(
                    f"Could not acquire Dask lock for Azure blob initial delete within {lock_timeout}s."
                )
            logger.debug("Dask init lock acquired for delete.")
            try:
                # Check flag *after* acquiring lock (safer)
                # We need a way to check if delete has *already* been done *by this specific initiate call*.
                # Dask Variables are problematic. Let's try deleting unconditionally within the lock.
                # delete_blob is idempotent. The lock ensures only one worker does it at a time.
                logger.info(
                    "Acquired Dask init lock. Deleting blob %s/%s (if exists)...",
                    self.container,
                    self.blob,
                )
                try:
                    self._blob_client.delete_blob(delete_snapshots="include")
                    logger.info(
                        "Blob %s/%s deleted successfully via Dask initiate.",
                        self.container,
                        self.blob,
                    )
                except Exception as e:
                    # Log error but potentially continue if deletion isn't fatal
                    logger.error(
                        "Failed delete blob %s/%s in Dask initiate: %s",
                        self.container,
                        self.blob,
                        e,
                    )
                    # Decide: raise error or allow continuation? Raising seems safer for overwrite=True.
                    raise RuntimeError(
                        f"Failed to delete blob for overwrite during initiate: {e}"
                    ) from e
            finally:
                logger.debug("Releasing Dask init lock after delete.")
                self._init_lock.release()

        elif not self._dask_client and self._init_lock:
            # Non-Dask Environment (threading lock)
            logger.debug("Using non-Dask lock for initial delete.")
            with self._init_lock:
                logger.debug("Acquired non-Dask lock for delete.")
                if self._initialized_on_scheduler:  # Check flag for non-dask case
                    logger.debug(
                        "Blob already deleted by this process (non-Dask flag)."
                    )
                else:
                    logger.info(
                        "Acquired non-Dask lock. Deleting blob %s/%s (if exists)...",
                        self.container,
                        self.blob,
                    )
                    try:
                        self._blob_client.delete_blob(delete_snapshots="include")
                        logger.info(
                            "Blob %s/%s deleted successfully via non-Dask initiate.",
                            self.container,
                            self.blob,
                        )
                        self._initialized_on_scheduler = (
                            True  # Set flag after successful delete
                        )
                    except Exception as e:
                        logger.error(
                            "Failed delete blob %s/%s in non-Dask initiate: %s",
                            self.container,
                            self.blob,
                            e,
                        )
                        raise RuntimeError(
                            f"Failed to delete blob for overwrite during initiate: {e}"
                        ) from e
            logger.debug("Released non-Dask lock.")
        else:
            # No lock configured - should not happen with current __init__
            logger.warning(
                "Initiate called for overwrite=True but no lock found. Deleting directly (potential race)."
            )
            try:
                self._blob_client.delete_blob(delete_snapshots="include")
                logger.info(
                    "Blob %s/%s deleted directly (no lock).", self.container, self.blob
                )
            except Exception as e:
                logger.exception(
                    "Failed delete blob %s/%s (no lock): %s",
                    self.container,
                    self.blob,
                    e,
                )
                raise RuntimeError(
                    f"Failed to delete blob for overwrite during initiate (no lock): {e}"
                ) from e

        return upload_id_string  # Return conceptual ID

    def _ensure_clients_initialized(self):
        """Initialize Azure SDK clients if they haven't been already."""
        if not self._clients_initialized:
            logger.debug("Initializing Azure clients for %s", self.account_url)
            self._blob_service_client = BlobServiceClient(
                account_url=self.account_url, credential=self.credential
            )
            self._container_client = self._blob_service_client.get_container_client(
                self.container
            )
            self._blob_client = self._container_client.get_blob_client(self.blob)
            self._clients_initialized = True
            logger.debug(
                "Azure clients initialized for %s/%s", self.container, self.blob
            )

    def write_part(self, part: int, data: bytes) -> dict[str, Any]:
        """
        Stages a single block (part) of data to Azure Blob Storage using Put Block.
        This block is uncommitted until finalise() is called.

        Args:
            part: The part number (used to generate a unique block ID).
            data: The byte data for this part.

        Returns:
            A dictionary containing 'PartNumber', 'BlockId', and 'Size'.
            This dictionary is collected by Dask and passed to finalise.
        """
        if not data:
            # Staging empty blocks is problematic. _mpu.py should handle this.
            logger.error(
                "Received empty data for part %s. This should not happen.", part
            )
            raise ValueError(f"Attempted to write empty data for part {part}")

        data_len = len(data)
        logger.debug("Staging part %s, size: %s bytes", part, data_len)

        if data_len > self.max_write_sz:
            logger.error(
                "Data size %s for part %s exceeds max_write_sz %s",
                data_len,
                part,
                self.max_write_sz,
            )
            raise ValueError(f"Block size too large: {data_len} > {self.max_write_sz}")

        # Ensure Azure clients are ready on this worker/process
        self._get_clients_if_needed()
        # Ensure potential blob deletion has happened (uses lock/variable)
        self._ensure_clients_initialized()

        # Generate Block ID
        block_id = self._get_block_id(part)
        logger.debug("Generated Block ID '%s' for part %s", block_id, part)

        # Stage Block with Retries
        # Retry logic for transient network errors or throttling

        max_retries = 5  # Number of retries for staging
        retry_delay = 1  # Initial delay in seconds
        for attempt in range(max_retries):
            try:
                self._blob_client.stage_block(
                    block_id=block_id, data=data, length=data_len
                )
                logger.info(
                    "Successfully staged part %s (Block ID: %s), size: %s bytes on attempt %s",
                    part,
                    block_id,
                    data_len,
                    attempt + 1,
                )
                return {
                    "PartNumber": part,
                    "BlockId": block_id,
                    "Size": data_len,
                }
            except HttpResponseError as e:
                # Check for retryable errors (e.g., 5xx server errors, 429 throttling)
                # Note: ResourceExistsError (409) for stage_block is unusual unless IDs clash,
                # which _get_block_id should prevent for a single upload instance.
                # If it happens, it might indicate overlapping upload attempts.
                is_retryable = e.status_code >= 500 or e.status_code == 429
                log_level = logging.WARNING if is_retryable else logging.ERROR

                logger.log(
                    log_level,
                    "Azure HTTP error staging part %s (Block ID: %s), attempt %s/%s: %s",
                    part,
                    block_id,
                    attempt + 1,
                    max_retries,
                    e,
                    exc_info=(
                        not is_retryable
                    ),  # Include stack trace for non-retryable errors
                )

                if is_retryable and attempt < max_retries - 1:
                    logger.info(
                        "Retrying stage_block for part %s (Block ID: %s) in %s seconds...",
                        part,
                        block_id,
                        retry_delay,
                    )
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                elif is_retryable:  # Last attempt failed
                    logger.error(
                        "Failed staging block for part %s (Block ID: %s) after %s attempts due to retryable error: %s",
                    )
                    raise RuntimeError(
                        f"Stage_block failed persistently for part={part} (Block ID: {block_id}): {e}"
                    ) from e
                else:  # Non-retryable error
                    raise RuntimeError(
                        f"Stage_block failed for part={part} (Block ID: {block_id}): {e}"
                    ) from e
            except Exception as e:
                # Catch other unexpected errors (network issues, SDK bugs?)
                logger.exception(
                    "Unexpected error staging part %s (Block ID: %s), attempt %s/%s: %s",
                    part,
                    block_id,
                    attempt + 1,
                    max_retries,
                    e,
                )
                # Assume unexpected errors might be transient on the first few attempts
                if attempt < max_retries - 1:
                    logger.info(
                        "Retrying stage_block for part %s (Block ID: %s) in %s seconds...",
                        part,
                        block_id,
                        retry_delay,
                    )
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise RuntimeError(
                        f"Stage_block failed for part={part} (Block ID: {block_id}): {e}"
                    ) from e

        # Should not be reached if loop completes without returning/raising
        raise RuntimeError(f"Stage_block for part {part} failed after all retries.")

    def finalise(self, parts: list[dict[str, Any]]) -> str:
        """
        Commits staged blocks to finalize the blob. Verifies expected blocks
        (reported by Dask) appear, but builds the final commit list based
        on blocks found directly on Azure matching the upload prefix to handle
        potential lost Dask task results. Applies specific timeouts and logging.

        Args:
            parts: List of dicts reported by successful write_part tasks.
                   Used for verification and discrepancy checks.
        Returns:
            The ETag of the committed blob, or an empty string on failure.
        """
        self._ensure_clients_initialized()

        commit_timeout = 60
        read_timeout = 180
        lease_timeout = 60
        # End Definitions

        logger.info(
            "Finalise started for %s/%s. Prefix: %s. Dask reported %s parts.",
            self.container,
            self.blob,
            self.upload_prefix,
            len(parts),
        )
        if not parts:
            # If Dask reports no parts, we likely shouldn't commit anything.
            # Depending on requirements, maybe check Azure for orphaned blocks?
            logger.warning(
                "Finalise called with empty parts list from Dask. Committing empty blob or aborting."
            )
            # Decide: Commit empty or raise error? Let's try committing empty for now.
            # To commit empty, the commit_list should be empty.
            # We still need lease etc.
            # For now, raise error, as empty COG is invalid.
            raise ValueError("No parts reported by Dask for finalisation.")

        self._get_clients_if_needed()  # Ensure clients ready

        # Dask Parts Analysis (for verification/logging)
        valid_dask_parts = [
            p
            for p in parts
            if isinstance(p, dict) and p.get("BlockId") and "PartNumber" in p
        ]
        if len(valid_dask_parts) != len(parts):
            logger.warning(
                "Found %s invalid part entries in Dask list. ",
                len(parts) - len(valid_dask_parts),
            )
            if not valid_dask_parts:
                raise ValueError(
                    "No valid part metadata found in Dask list for finalisation."
                )
        dask_block_ids_set = {p["BlockId"] for p in valid_dask_parts}
        logger.debug(
            "Expecting %s unique blocks based on Dask results.", len(dask_block_ids_set)
        )

        # Get Definitive Uncommitted Block List from Azure
        logger.info(
            "Fetching final list of UNCOMMITTED blocks from Azure to build commit list."
        )
        server_uncommitted_blocks = []
        try:
            # Fetch only uncommitted blocks this time
            _, server_uncommitted_blocks = self._blob_client.get_block_list(
                block_list_type="uncommitted", timeout=read_timeout
            )
            logger.info(
                "Found %s uncommitted blocks on server.", len(server_uncommitted_blocks)
            )
        except (HttpResponseError, AzureError) as e:
            logger.exception(
                "Failed to get final uncommitted block list from Azure: %s",
                e,
            )
            raise RuntimeError(
                f"Could not get final uncommitted block list from Azure: {e}"
            ) from e
        except Exception as e:
            logger.exception(
                "Unexpected error getting final uncommitted block list: %s",
                e,
            )
            raise RuntimeError(
                f"Unexpected error getting final uncommitted block list: {e}"
            ) from e

        # Filter, Parse, and Sort Blocks from Server
        server_parts_map = {}  # Map PartNumber -> BlockId
        prefix_check = self.upload_prefix + "-p"  # Expected start of decoded ID
        block_prefix_check_encoded = base64.b64encode(
            prefix_check.encode("utf-8")
        ).decode("utf-8")[
            :10
        ]  # Check first few chars of encoded prefix

        for block in server_uncommitted_blocks:
            block_id = block.id
            # Quick check using encoded prefix before decoding (optimization)
            if not block_id.startswith(block_prefix_check_encoded):
                logger.debug(
                    "Skipping block ID '%s...' - does not match encoded prefix start '%s'.",
                    block_id[:10],
                    block_prefix_check_encoded,
                )
                continue

            try:
                decoded_id_bytes = base64.b64decode(block_id.encode("utf-8"))
                decoded_id_str = decoded_id_bytes.decode("utf-8")

                if decoded_id_str.startswith(prefix_check):
                    part_str = decoded_id_str[len(prefix_check) :]
                    part_num = int(part_str)  # Expecting digits here
                    if part_num in server_parts_map:
                        logger.warning(
                            "Duplicate PartNumber %s detected in server block list"
                            " for prefix %s! BlockID1=%s, BlockID2=%s",
                            self.upload_prefix,
                            server_parts_map[part_num],
                            block_id,
                        )
                        # Decide how to handle duplicates - error? ignore? For now, overwrite (last one wins)
                    server_parts_map[part_num] = block_id
                else:
                    logger.debug(
                        "Skipping decoded block ID '%s' - does not match prefix '%s'.",
                        decoded_id_str,
                        prefix_check,
                    )
                    pass

            except (ValueError, TypeError, UnicodeDecodeError) as parse_e:
                logger.warning(
                    "Could not parse block ID '%s' (Decoded: %s): %s",
                    block_id,
                    decoded_id_bytes if "decoded_id_bytes" in locals() else "N/A",
                    parse_e,
                )
            except Exception as unexpected_e:
                logger.exception(
                    "Unexpected error parsing block ID '%s': %s",
                    block_id,
                    unexpected_e,
                )

        if not server_parts_map:
            logger.error(
                "No uncommitted blocks found on server matching prefix '%s'. Cannot commit.",
                self.upload_prefix,
            )
            raise ValueError(
                f"Azure reported no uncommitted blocks matching the upload prefix '{self.upload_prefix}'."
            )

        # Sort server blocks by PartNumber
        sorted_server_parts = sorted(
            server_parts_map.items()
        )  # List of (PartNum, BlockId) tuples

        # Build Final Commit List (from Server Data)
        commit_list = [
            BlobBlock(block_id=block_id) for _, block_id in sorted_server_parts
        ]
        logger.info(
            "Constructed final commit list with %s blocks based on server data matching prefix '%s'.",
            len(commit_list),
            self.upload_prefix,
        )
        logger.debug(
            "Final commit list block IDs (first 5): %s", [b.id for b in commit_list[:5]]
        )

        # Log Discrepancies with Dask List
        if len(commit_list) != len(dask_block_ids_set):
            logger.warning(
                "DISCREPANCY: Dask reported %s successful parts, but final commit"
                " list from server has %s parts for prefix '%s'.",
                len(dask_block_ids_set),
                len(commit_list),
                self.upload_prefix,
            )
            server_block_ids_commit = {b.id for b in commit_list}
            dask_missing_on_server = dask_block_ids_set - server_block_ids_commit
            server_extras_not_in_dask = server_block_ids_commit - dask_block_ids_set
            if dask_missing_on_server:
                logger.warning(
                    "Blocks in Dask list but NOT in server commit list (%s): %s...",
                    len(dask_missing_on_server),
                    list(dask_missing_on_server)[:5],
                )
            if server_extras_not_in_dask:
                logger.warning(
                    "Blocks in server commit list but NOT reported by Dask (%s): %s...",
                    len(server_extras_not_in_dask),
                    list(server_extras_not_in_dask)[:5],
                )
            # Decide: Raise error on discrepancy? Or proceed with server list?
            # Proceeding is likely the only way to recover from lost Dask results.

        # Commit Block List
        returned_etag: str | None = None
        lease = None
        try:  # Ensure lease is released in finally block
            # Acquire Azure Lease
            try:
                lease_duration = 60
                logger.debug(
                    "Attempting acquire lease (duration: %ss) with timeout=%ss",
                    lease_duration,
                    lease_timeout,
                )
                lease = self._blob_client.acquire_lease(
                    lease_duration=lease_duration, timeout=lease_timeout
                )
                logger.info("Lease acquired successfully: %s", lease.id)
            except (HttpResponseError, AzureError) as e:
                if isinstance(e, HttpResponseError) and e.status_code == 409:
                    raise RuntimeError(
                        f"Could not acquire lease (Conflict 409): {e}"
                    ) from e
                else:
                    raise RuntimeError(f"Could not acquire lease: {e}") from e
            except Exception as e:
                raise RuntimeError(f"Unexpected error acquiring lease: {e}") from e

            # Commit Block List (protected by lease)
            try:
                content_settings = ContentSettings(content_type="image/tiff")
                logger.debug(
                    "Attempting commit_block_list (%s blocks from server list) with timeout=%ss...",
                    len(commit_list),
                    commit_timeout,
                )
                commit_response = self._blob_client.commit_block_list(
                    commit_list,  # Use SERVER derived list
                    lease=lease,
                    content_settings=content_settings,
                    timeout=commit_timeout,
                )
                logger.info(
                    "Commit_block_list call successful using server-derived list."
                )

                etag = commit_response.get("etag")
                last_modified = commit_response.get("last_modified")
                logger.info(
                    "Successfully committed blob. ETag: %s, LastModified: %s",
                    etag,
                    last_modified,
                )
                returned_etag = etag  # Store etag

                # Optional: Verify final blob size
                try:
                    logger.debug(
                        "Attempting get_blob_properties with timeout=%ss...",
                        read_timeout,
                    )
                    props = self._blob_client.get_blob_properties(
                        lease=lease, timeout=read_timeout
                    )
                    logger.debug("Get_blob_properties call successful.")
                    final_size = props.size
                    logger.info("Final blob size: %s bytes.", final_size)
                    if final_size < 1024:
                        logger.warning(
                            "Committed blob size (%s bytes) is suspiciously small.",
                            final_size,
                        )
                except (HttpResponseError, AzureError, Exception) as prop_e:
                    logger.warning(
                        "Could not verify final blob size after commit: %s", prop_e
                    )

            # ... (Commit error handling remains the same) ...
            except (HttpResponseError, AzureError) as e:
                raise RuntimeError(f"Commit_block_list failed: {e}") from e
            except Exception as e:
                raise RuntimeError(f"Unexpected error committing blob: {e}") from e

        finally:
            # Release Azure Lease
            if lease:
                logger.debug(
                    "Attempting release lease %s with timeout=%ss",
                    lease.id,
                    lease_timeout,
                )
                try:
                    lease.release(timeout=lease_timeout)
                    logger.info("Lease %s released successfully.", lease.id)
                except (HttpResponseError, AzureError, Exception) as release_e:
                    logger.error("Failed to release lease %s: %s", lease.id, release_e)

        # Return Result
        if returned_etag is None:
            logger.error("Finalise finished but no ETag was obtained.")
            return ""
        else:
            logger.info("Finalise returning ETag: %s", returned_etag)
            return returned_etag

    def cancel(self, other: str = ""):
        """
        Cancels the multipart upload. For Azure Block Blobs, there's no explicit
        'cancel' API like S3. Uncommitted blocks are garbage collected eventually.
        This method primarily logs the cancellation intent.
        """
        if other:
            logger.warning("Received unexpected parameter in cancel: %s", other)
        logger.warning(
            "Cancelling upload for %s/%s. Staged blocks will be garbage collected by Azure.",
            self.container,
            self.blob,
        )
        # No Azure API call needed here. Uncommitted blocks expire after 7 days by default.
        # See: https://learn.microsoft.com/en-us/rest/api/storageservices/put-block#remarks

    @property
    def url(self) -> str:
        """Returns the URL of the target blob."""
        # Ensure clients are initialized to construct the URL
        self._get_clients_if_needed()
        return self._blob_client.url

    @property
    def started(self) -> bool:
        """
        Indicates if the upload process has conceptually started.
        For Azure, this doesn't track an UploadId like S3. Returns True
        after initiate() is called conceptually.
        """
        # Since initiate doesn't do much state change tracked here,
        # we can perhaps check if clients are initialized as a proxy?
        # Or just return True after the object is created. Let's use client init.
        return self._clients_initialized

    def writer(self, kw: dict[str, Any], *, client: Any = None):
        """Returns a Dask-compatible writer object."""
        # Pass the Dask client if provided explicitly
        dask_client = client if client else self._dask_client
        return DelayedAzureWriter(self, kw, dask_client=dask_client)

    def dask_name_prefix(self) -> str:
        """Prefix for Dask graph task names related to finalization."""
        return f"azure-finalise-{self.container}-{self.blob}"

    # Dask Serialization
    # Make the class more friendly for Dask serialization
    # Avoid storing non-serializable objects like SDK clients directly if possible,
    # or ensure they are recreated on workers. _get_clients_if_needed helps here.

    def __getstate__(self):
        """Prepare state for pickling. Excludes non-serializable objects."""
        # Start with a copy of the instance dictionary
        state = self.__dict__.copy()

        # Explicitly remove or nullify non-serializable attributes
        state["_blob_service_client"] = None
        state["_container_client"] = None
        state["_blob_client"] = None
        state["_clients_initialized"] = False  # Ensure re-initialization on worker
        state["_dask_client"] = None  # Dask client is context-specific
        state["_init_lock"] = None  # Remove Dask or threading lock object
        state["_init_var"] = None  # Remove Dask variable object

        logger.debug(
            "Serializing AzMultiPartUpload state for %s/%s (keys: %s)",
            self.container,
            self.blob,
            list(state.keys()),
        )
        return state

    def __setstate__(self, state):
        logger.debug(
            "Deserializing AzMultiPartUpload state for %s/%s",
            state["container"],
            state["blob"],
        )
        self.__dict__.update(state)
        # Re-acquire Dask client/locks on the worker/scheduler if needed
        try:
            self._dask_client = get_client()
            lock_var_name = (
                f"odc-az-init-{self.container}-{self.blob}-{tokenize(self.account_url)}"
            )
            logger.debug("Re-linking Dask Lock: %s-lock", lock_var_name)
            self._init_lock = Lock(f"{lock_var_name}-lock", client=self._dask_client)
        except ValueError:
            logger.warning(
                "No Dask client found upon deserialization. Recreating threading lock."
            )
            self._dask_client = None
            # Recreate threading lock if Dask isn't available
            # Check if _init_lock was already a threading lock from state (unlikely due to getstate)
            # Safest is to just create a new one if Dask isn't found.
            self._init_lock = threading.Lock()
        except Exception as e:
            logger.exception(
                "Error re-linking Dask primitives during deserialization: %s",
                e,
            )
            # Fallback if lock re-linking fails?
            logger.warning(
                "Falling back to threading.Lock due to error during Dask primitive re-linking."
            )
            self._dask_client = None
            self._init_lock = threading.Lock()


class DelayedAzureWriter(AzureLimits):
    """
    A wrapper around AzMultiPartUpload methods to be used as a callable
    in Dask graphs (e.g., with dask.bag.map_partitions).
    Handles initialization and calling the appropriate AzMultiPartUpload methods.
    """

    # Define __slots__ for potentially lower memory usage if many writer instances are created
    __slots__ = ("mpu", "kw", "dask_client", "_init_checked")

    def __init__(
        self, mpu: AzMultiPartUpload, kw: dict[str, Any], dask_client: Any = None
    ):
        self.mpu = mpu
        self.kw = kw  # Keyword arguments potentially passed for initialization (not currently used)
        self.dask_client = dask_client  # Store Dask client reference if provided
        self._init_checked = (
            False  # Flag to avoid redundant init checks per writer instance
        )

    def _ensure_init(self) -> None:
        """
        Ensures the underlying AzMultiPartUpload object is initialized
        (clients ready, blob potentially deleted).
        """
        # Optimization: Check flag first to avoid repeated calls to potentially expensive methods
        if not self._init_checked:
            self.mpu._get_clients_if_needed()
            # Trigger the one-time initialization (e.g., delete) if needed
            self.mpu._ensure_clients_initialized()
            self._init_checked = True

    def __call__(self, part: int, data: bytes) -> dict[str, Any]:
        """
        Callable for Dask to stage a part. Ensures initialization first.
        """
        self._ensure_init()
        logger.debug("DelayedAzureWriter calling write_part for part %s", part)
        return self.mpu.write_part(part, data)

    def finalise(self, parts: list[dict[str, Any]]) -> dict[str, str]:
        """
        Callable for Dask to finalize the upload.
        Returns a dictionary with container, blob, and ETag.
        """
        # Initialization should have happened during write_part calls,
        # but call _ensure_init() just in case finalise is called directly
        # or if no write_part calls were made (e.g., empty file scenario handled in mpu.finalise).
        self._ensure_init()
        logger.info(
            "DelayedAzureWriter calling finalise for %s/%s",
            self.mpu.container,
            self.mpu.blob,
        )
        etag = self.mpu.finalise(parts)
        logger.infor("Finalise completed with ETag: %s", etag)
        # Return structured result
        return {"container": self.mpu.container, "blob": self.mpu.blob, "ETag": etag}

    def __dask_tokenize__(self):
        # Ensure deterministic tokenization for Dask caching/graph building
        return (
            "odc.DelayedAzureWriter",
            tokenize(self.mpu),  # Tokenize the underlying AzMultiPartUpload object
            tuple(sorted(self.kw.items())),
        )

    def __getstate__(self):
        """Prepare state for pickling, compatible with __slots__."""
        state = {}
        for slot in self.__slots__:
            try:
                # Attempt to get the value of the slot attribute
                value = getattr(self, slot)
                # Exclude the Dask client explicitly, as it's not serializable
                # and should be re-acquired on the worker via get_client() if needed.
                if slot == "dask_client":
                    continue  # Don't include dask_client in the state
                state[slot] = value
            except AttributeError:
                # If an attribute defined in __slots__ doesn't exist (shouldn't normally happen
                # after __init__ unless something went wrong), log a warning and skip it.
                logger.warning(
                    "Attribute '%s' not found on DelayedAzureWriter during __getstate__.",
                    slot,
                )

        logger.debug(
            "Serializing DelayedAzureWriter state (slots): %s", list(state.keys())
        )
        return state

    def __setstate__(self, state):
        """Restore state from pickle, compatible with __slots__."""
        logger.debug(
            "Deserializing DelayedAzureWriter state (slots): %s", list(state.keys())
        )

        # Iterate over the slots defined for the class
        for slot in self.__slots__:
            # Check if the slot key exists in the loaded state dictionary
            if slot in state:
                # If it exists, set the attribute on the object using setattr
                setattr(self, slot, state[slot])
            else:
                # Handle slots that might be missing in the state dictionary
                if slot == "dask_client":
                    # Always set dask_client to None initially after deserialization.
                    # The AzMultiPartUpload instance handles re-acquiring it if needed.
                    setattr(self, slot, None)
                elif slot == "_init_checked":
                    logger.warning(
                        "'%s' not found in deserialized state. Setting to False.",
                        slot,
                    )
                    setattr(self, slot, False)
                else:
                    # For essential slots like 'mpu' or 'kw', missing state is a problem.
                    # Log an error. Setting to None might cause errors later.
                    logger.error(
                        "Essential slot '%s' missing during DelayedAzureWriter deserialization! Setting to None.",
                        slot,
                    )
                    setattr(self, slot, None)  # Set to None as a fallback

        logger.debug("DelayedAzureWriter state restored.")
