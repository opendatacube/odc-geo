"""
S3 utils for COG to S3.
"""
from __future__ import annotations

from threading import Lock
from typing import TYPE_CHECKING, Any, Optional, Tuple

from cachetools import cached

from ._mpu import PartsWriter, SomeData, mpu_write

if TYPE_CHECKING:
    import dask.bag
    import distributed
    from botocore.credentials import ReadOnlyCredentials
    from dask.delayed import Delayed

_state: dict[str, Any] = {}


def _mpu_local_lock(k="mpu_lock") -> Lock:
    lck = _state.get(k, None)
    if lck is not None:
        return lck

    return _state.setdefault("mpu_lock", Lock())


def _dask_client() -> "distributed.Client" | None:
    # pylint: disable=import-outside-toplevel,import-error
    from distributed import get_client

    try:
        return get_client()
    except ValueError:
        return None


def s3_parse_url(url: str) -> Tuple[str, str]:
    if url.startswith("s3://"):
        bucket, *key = url[5:].split("/", 1)
        key = key[0] if len(key) else ""
        return bucket, key
    return ("", "")


class S3Limits:
    """
    Common S3 writer settings
    """

    @property
    def min_write_sz(self) -> int:
        return 5 * (1 << 20)

    @property
    def max_write_sz(self) -> int:
        return 5 * (1 << 30)

    @property
    def min_part(self) -> int:
        return 1

    @property
    def max_part(self) -> int:
        return 10_000


class MultiPartUpload(S3Limits):
    """
    Dask to S3 dumper.
    """

    def __init__(
        self,
        bucket: str,
        key: str,
        *,
        uploadId: str = "",
        profile: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        creds: Optional["ReadOnlyCredentials"] = None,
    ):
        self.bucket = bucket
        self.key = key
        self.uploadId = uploadId
        self.profile = profile
        self.endpoint_url = endpoint_url
        self.creds = creds

    @cached({})
    def s3_client(self):
        # pylint: disable=import-outside-toplevel,import-error
        from botocore.session import Session

        sess = Session(profile=self.profile)
        creds = self.creds
        if creds is None:
            return sess.create_client("s3", endpoint_url=self.endpoint_url)
        return sess.create_client(
            "s3",
            endpoint_url=self.endpoint_url,
            aws_access_key_id=creds.access_key,
            aws_secret_access_key=creds.secret_key,
            aws_session_token=creds.token,
        )

    def initiate(self, **kw) -> str:
        assert self.uploadId == ""
        s3 = self.s3_client()

        rr = s3.create_multipart_upload(Bucket=self.bucket, Key=self.key, **kw)
        uploadId = rr["UploadId"]
        self.uploadId = uploadId
        return uploadId

    def write_part(self, part: int, data: SomeData) -> dict[str, Any]:
        s3 = self.s3_client()
        assert self.uploadId != ""
        rr = s3.upload_part(
            PartNumber=part,
            Body=data,
            Bucket=self.bucket,
            Key=self.key,
            UploadId=self.uploadId,
        )
        etag = rr["ETag"]
        return {"PartNumber": part, "ETag": etag}

    @property
    def url(self) -> str:
        return f"s3://{self.bucket}/{self.key}"

    def finalise(self, parts: list[dict[str, Any]]) -> str:
        s3 = self.s3_client()
        assert self.uploadId

        rr = s3.complete_multipart_upload(
            Bucket=self.bucket,
            Key=self.key,
            UploadId=self.uploadId,
            MultipartUpload={"Parts": parts},
        )

        return rr["ETag"]

    @property
    def started(self) -> bool:
        return len(self.uploadId) > 0

    def cancel(self, other: str = ""):
        uploadId = other if other else self.uploadId
        if not uploadId:
            return

        s3 = self.s3_client()
        if uploadId.lower() in ("all", ":all:"):
            for uploadId in self.list_active():
                s3.abort_multipart_upload(
                    Bucket=self.bucket, Key=self.key, UploadId=uploadId
                )
            self.uploadId = ""
        else:
            s3.abort_multipart_upload(
                Bucket=self.bucket, Key=self.key, UploadId=uploadId
            )
            if uploadId == self.uploadId:
                self.uploadId = ""

    def list_active(self):
        s3 = self.s3_client()
        rr = s3.list_multipart_uploads(Bucket=self.bucket, Prefix=self.key)
        return [x["UploadId"] for x in rr.get("Uploads", [])]

    def read(self, **kw):
        s3 = self.s3_client()
        return s3.get_object(Bucket=self.bucket, Key=self.key, **kw)["Body"].read()

    def __dask_tokenize__(self):
        return (
            self.bucket,
            self.key,
            self.uploadId,
        )

    def writer(self, kw, *, client: Any = None) -> PartsWriter:
        if client is None:
            client = _dask_client()
        writer = DelayedS3Writer(self, kw)
        if client is not None:
            writer.prep_client(client)
        return writer

    # pylint: disable=too-many-arguments
    def upload(
        self,
        chunks: "dask.bag.Bag" | list["dask.bag.Bag"],
        *,
        mk_header: Any = None,
        mk_footer: Any = None,
        user_kw: dict[str, Any] | None = None,
        writes_per_chunk: int = 1,
        spill_sz: int = 20 * (1 << 20),
        client: Any = None,
        **kw,
    ) -> "Delayed":
        write = self.writer(kw, client=client) if spill_sz else None
        return mpu_write(
            chunks,
            write,
            mk_header=mk_header,
            mk_footer=mk_footer,
            user_kw=user_kw,
            writes_per_chunk=writes_per_chunk,
            spill_sz=spill_sz,
            dask_name_prefix="s3finalise",
        )


def _safe_get(v, timeout=0.1):
    try:
        return v.get(timeout)
    except Exception:  # pylint: disable=broad-except
        return None


class DelayedS3Writer(S3Limits):
    """
    Delay multi-part upload creation until first write.
    """

    # pylint: disable=import-outside-toplevel,import-error

    def __init__(self, mpu: MultiPartUpload, kw: dict[str, Any]):
        self.mpu = mpu
        self.kw = kw  # mostly ContentType= kinda thing
        self._shared_var: Optional["distributed.Variable"] = None

    def prep_client(self, client: "distributed.Client") -> "distributed.Variable":
        v = self._shared(client)
        v.set(None)
        return v

    def cleanup_client(self, client: "distributed.Client") -> None:
        v = self._shared(client)
        v.delete()

    def _build_name(self, prefix: str) -> str:
        from dask.base import tokenize

        return f"{prefix}-{tokenize(self)}"

    def _shared(self, client: "distributed.Client") -> "distributed.Variable":
        from distributed import Variable

        if self._shared_var is None:
            self._shared_var = Variable(self._build_name("MPUpload"), client)
        return self._shared_var

    def _ensure_init(self, final_write: bool = False) -> MultiPartUpload:
        # pylint: disable=too-many-return-statements
        mpu = self.mpu
        if mpu.started:
            return mpu

        client = _dask_client()
        if client is None:
            # Assume running locally with everyone sharing same self.mpu
            with _mpu_local_lock():
                if not final_write:
                    _ = mpu.initiate(**self.kw)
                return mpu

        from distributed import Lock as DLock

        shared_state = self._shared(client)
        uploadId = _safe_get(shared_state, 0.1)

        if uploadId is not None:
            # someone else initialized it
            mpu.uploadId = uploadId
            return mpu

        lock = DLock(self._build_name("MPULock"), client)
        with lock:
            uploadId = _safe_get(shared_state, 0.1)
            if uploadId is not None:
                # someone else initialized it while we were getting a lock
                mpu.uploadId = uploadId
                return mpu

            # We are first to Lock
            # 1. Start upload
            # 2. Share UploadId with others
            if not final_write:
                _ = mpu.initiate(**self.kw)
                shared_state.set(mpu.uploadId)

        assert mpu.started or final_write
        return mpu

    def __call__(self, part: int, data: SomeData) -> dict[str, Any]:
        mpu = self._ensure_init()
        return mpu.write_part(part, data)

    def finalise(self, parts: list[dict[str, Any]]) -> Any:
        assert len(parts) > 0
        mpu = self._ensure_init()
        etag = mpu.finalise(parts)
        client = _dask_client()
        if client:
            # remove shared variable if running on distributed cluster
            self.cleanup_client(client)
        return {"Bucket": mpu.bucket, "Key": mpu.key, "ETag": etag}

    def __dask_tokenize__(self):
        return ("odc.DelayedS3Writer", self.mpu.bucket, self.mpu.key)
