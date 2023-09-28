"""
S3 utils for COG to S3.
"""
from typing import TYPE_CHECKING, Any, Dict, Optional

from cachetools import cached
from dask import bag as dask_bag

from ._mpu import MPUChunk, PartsWriter

if TYPE_CHECKING:
    from botocore.credentials import ReadOnlyCredentials


class MultiPartUpload:
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

    # @cached({}, key=lambda _self: (_self.profile, _self.endpoint_url, _self.creds))
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

    def __call__(self, partId: int, data: bytearray) -> Dict[str, Any]:
        s3 = self.s3_client()
        assert self.uploadId != ""
        rr = s3.upload_part(
            PartNumber=partId,
            Body=data,
            Bucket=self.bucket,
            Key=self.key,
            UploadId=self.uploadId,
        )
        etag = rr["ETag"]
        return {"PartNumber": partId, "ETag": etag}

    def initiate(self) -> str:
        assert self.uploadId == ""
        s3 = self.s3_client()

        rr = s3.create_multipart_upload(Bucket=self.bucket, Key=self.key)
        uploadId = rr["UploadId"]
        self.uploadId = uploadId
        return uploadId

    @property
    def started(self) -> bool:
        return len(self.uploadId) > 0

    def cancel(self, other: str = ""):
        uploadId = other if other else self.uploadId
        if not uploadId:
            return

        s3 = self.s3_client()
        s3.abort_multipart_upload(Bucket=self.bucket, Key=self.key, UploadId=uploadId)
        if uploadId == self.uploadId:
            self.uploadId = ""

    def complete(self, root: MPUChunk) -> str:
        s3 = self.s3_client()
        rr = s3.complete_multipart_upload(
            Bucket=self.bucket,
            Key=self.key,
            UploadId=self.uploadId,
            MultipartUpload={"Parts": root.parts},
        )

        return rr["ETag"]

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

    def substream(
        self,
        partId: int,
        chunks: dask_bag.Bag,
        *,
        writes_per_chunk: int = 1,
        mark_final: bool = False,
        spill_sz: int = 20 * (1 << 20),
    ) -> dask_bag.Item:
        write: Optional[PartsWriter] = None
        if spill_sz > 0:
            if not self.started:
                self.initiate()
            write = self
        return MPUChunk.from_dask_bag(
            partId,
            chunks,
            writes_per_chunk=writes_per_chunk,
            mark_final=mark_final,
            spill_sz=spill_sz,
            write=write,
        )
