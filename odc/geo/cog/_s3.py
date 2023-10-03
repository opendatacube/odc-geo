"""
S3 utils for COG to S3.
"""
from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

from cachetools import cached

from ._mpu import MPUChunk, PartsWriter, SomeData

if TYPE_CHECKING:
    import dask.bag
    from botocore.credentials import ReadOnlyCredentials

MkHeader = Callable[[List[Tuple[int, Any]]], SomeData]
MkFooter = MkHeader


def s3_parse_url(url: str) -> Tuple[str, str]:
    if url.startswith("s3://"):
        bucket, *key = url[5:].split("/", 1)
        key = key[0] if len(key) else ""
        return bucket, key
    return ("", "")


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

    def __call__(self, part: int, data: SomeData) -> Dict[str, Any]:
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

    def finalise(self, parts: List[Dict[str, Any]]) -> str:
        s3 = self.s3_client()
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

    def upload(
        self,
        chunks: "dask.bag.Bag" | List["dask.bag.Bag"],
        *,
        mk_header: Optional[MkHeader] = None,
        mk_footer: Optional[MkFooter] = None,
        writes_per_chunk: int = 1,
        spill_sz: int = 20 * (1 << 20),
        **kw,
    ) -> "dask.bag.Item":
        if not isinstance(chunks, list):
            data_substream = self._substream(
                2,
                chunks,
                writes_per_chunk=writes_per_chunk,
                lhs_keep=self.min_write_sz,
                spill_sz=spill_sz,
                mark_final=mk_footer is None,
                **kw,
            )
        else:
            write: Optional[PartsWriter] = self if spill_sz > 0 else None
            partId = 2
            dss = []
            for ch in chunks:
                sub = self._substream(
                    partId,
                    ch,
                    writes_per_chunk=writes_per_chunk,
                    lhs_keep=self.min_write_sz,
                    spill_sz=spill_sz,
                    mark_final=False,
                    **kw,
                )
                dss.append(sub)
                partId = partId + ch.npartitions * writes_per_chunk
                assert partId <= self.max_part

            data_substream = MPUChunk.collate_substreams(
                dss,
                write=write,
                spill_sz=spill_sz,
            )

        return data_substream.apply(
            partial(
                _finalizer_dask_op, mpu=self, mk_header=mk_header, mk_footer=mk_footer
            )
        )

    def _substream(
        self,
        partId: int,
        chunks: "dask.bag.Bag",
        *,
        writes_per_chunk: int = 1,
        mark_final: bool = False,
        lhs_keep: int = 5 * (1 << 20),
        spill_sz: int = 20 * (1 << 20),
        **kw,
    ) -> "dask.bag.Item":
        write: Optional[PartsWriter] = None
        if spill_sz > 0:
            if not self.started:
                self.initiate(**kw)
            write = self
        return MPUChunk.from_dask_bag(
            partId,
            chunks,
            writes_per_chunk=writes_per_chunk,
            mark_final=mark_final,
            lhs_keep=lhs_keep,
            spill_sz=spill_sz,
            write=write,
        )

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


def _finalizer_dask_op(
    data_substream: MPUChunk,
    *,
    mpu: MultiPartUpload,
    mk_header: Optional[MkHeader] = None,
    mk_footer: Optional[MkFooter] = None,
):
    _root = data_substream
    hdr_bytes, footer_bytes = [
        None if op is None else op(data_substream.observed)
        for op in [mk_header, mk_footer]
    ]

    if footer_bytes:
        _root.append(footer_bytes)

    if hdr_bytes:
        hdr = MPUChunk(1, 1)
        hdr.append(hdr_bytes)
        _root = MPUChunk.merge(hdr, _root)

    if not mpu.started:
        return _root

    _, rr = _root.flush(mpu, leftPartId=1, finalise=True)
    return rr
