import hashlib

import pytest

from src.data.gdc_client import GDCClient, GDCError


def test_existing_valid_file_is_reused_without_network(tmp_path):
    payload = b"raw-gdc-content\n"
    destination = tmp_path / "counts.tsv"
    destination.write_bytes(payload)
    client = GDCClient(session=None)
    receipt = client.download_file(
        "file-1",
        destination,
        hashlib.md5(payload).hexdigest(),
        len(payload),
    )
    assert receipt["reused"] is True


def test_existing_corrupt_file_fails_instead_of_overwriting(tmp_path):
    destination = tmp_path / "counts.tsv"
    destination.write_bytes(b"corrupt")
    client = GDCClient(session=None)
    with pytest.raises(GDCError, match="Size mismatch"):
        client.download_file("file-1", destination, "0" * 32, 100)
