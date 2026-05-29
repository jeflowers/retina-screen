"""S3 raw-image storage.

The raw fundus image is the replay source — it is written immutably to S3
BEFORE classification runs, so any downstream re-processing can fetch the
exact bytes the pipeline acted on.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client

__all__ = ["build_raw_image_key", "get_raw_image", "put_raw_image"]


def build_raw_image_key(site_id: str, encounter_id: str) -> str:
    """Object key for a raw fundus image.

    Layout: `raw/{site_id}/{encounter_id}.jpg`. Site prefix gives a natural
    listing/partitioning unit and matches the DynamoDB partition key.
    """
    return f"raw/{site_id}/{encounter_id}.jpg"


def put_raw_image(
    client: S3Client,
    bucket: str,
    site_id: str,
    encounter_id: str,
    image_bytes: bytes,
) -> str:
    """Write the raw image to S3 immutably; return the object key."""
    key = build_raw_image_key(site_id, encounter_id)
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=image_bytes,
        ContentType="image/jpeg",
    )
    return key


def get_raw_image(client: S3Client, bucket: str, key: str) -> bytes:
    """Fetch the raw image bytes for replay."""
    response = client.get_object(Bucket=bucket, Key=key)
    body: bytes = response["Body"].read()
    return body
