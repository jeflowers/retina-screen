"""Tests for the S3 raw-image adapter (client injected, no network).

Verifies the immutable-replay-source contract: put_raw_image lands the exact
bytes at a deterministic key, get_raw_image returns them unchanged.
"""

from io import BytesIO
from unittest.mock import MagicMock

from retina_screen.storage.s3_store import (
    build_raw_image_key,
    get_raw_image,
    put_raw_image,
)


def test_build_raw_image_key_structure() -> None:
    key = build_raw_image_key("site_A", "enc_001")
    assert key == "raw/site_A/enc_001.jpg"


def test_put_raw_image_calls_s3_with_expected_args() -> None:
    client = MagicMock()
    image = b"FAKE-JPEG-BYTES"

    key = put_raw_image(
        client, bucket="raw-bucket", site_id="site_A", encounter_id="enc_001", image_bytes=image
    )

    assert key == "raw/site_A/enc_001.jpg"
    client.put_object.assert_called_once_with(
        Bucket="raw-bucket",
        Key="raw/site_A/enc_001.jpg",
        Body=image,
        ContentType="image/jpeg",
    )


def test_get_raw_image_returns_body_bytes() -> None:
    client = MagicMock()
    client.get_object.return_value = {"Body": BytesIO(b"FAKE-JPEG-BYTES")}

    body = get_raw_image(client, bucket="raw-bucket", key="raw/site_A/enc_001.jpg")

    assert body == b"FAKE-JPEG-BYTES"
    client.get_object.assert_called_once_with(Bucket="raw-bucket", Key="raw/site_A/enc_001.jpg")


def test_put_and_get_roundtrip_via_fake_client() -> None:
    """Use a stateful fake to prove put-then-get returns the same bytes."""
    storage: dict[str, bytes] = {}
    client = MagicMock()

    def fake_put(*, Bucket: str, Key: str, Body: bytes, ContentType: str) -> dict[str, str]:  # noqa: N803 — boto3 PascalCase kwargs
        storage[f"{Bucket}/{Key}"] = Body
        return {"ETag": "fake"}

    def fake_get(*, Bucket: str, Key: str) -> dict[str, BytesIO]:  # noqa: N803
        return {"Body": BytesIO(storage[f"{Bucket}/{Key}"])}

    client.put_object.side_effect = fake_put
    client.get_object.side_effect = fake_get

    key = put_raw_image(client, bucket="b", site_id="s", encounter_id="e", image_bytes=b"hello")
    fetched = get_raw_image(client, bucket="b", key=key)
    assert fetched == b"hello"
