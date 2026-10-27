"""Durable-data-root checks must follow --root, not one hard-coded path."""

from __future__ import annotations

from pathlib import Path

import pytest

from slamibot_fieldmind.backend_runtime import BackendConfig


def test_mount_path_defaults_to_the_legacy_usb_volume():
    assert BackendConfig().required_mount == Path("/mnt/usb")


def test_explicit_mount_path_is_used():
    config = BackendConfig(root="/data/semantic_index", mount_path="/data")
    assert config.required_mount == Path("/data")


def test_root_outside_the_declared_mount_is_rejected(monkeypatch):
    config = BackendConfig(root="/data/semantic_index", mount_path="/data")
    monkeypatch.setattr("os.path.ismount", lambda path: str(path) == "/data")
    monkeypatch.setattr(Path, "resolve", lambda self: self)
    assert config.data_root_is_mounted() is True

    outside = BackendConfig(root="/tmp/semantic_index", mount_path="/data")
    assert outside.data_root_is_mounted() is False


def test_unmounted_mount_path_is_rejected(monkeypatch):
    config = BackendConfig(root="/data/semantic_index", mount_path="/data")
    monkeypatch.setattr("os.path.ismount", lambda path: False)
    assert config.data_root_is_mounted() is False


def test_runtime_refuses_an_unmounted_root(monkeypatch, tmp_path):
    monkeypatch.setattr("os.path.ismount", lambda path: False)
    config = BackendConfig(root=tmp_path, min_free_bytes=0, require_mount=True)
    from slamibot_fieldmind.backend_runtime import BackendRuntime

    with pytest.raises(RuntimeError, match="must be mounted"):
        BackendRuntime(config, Encoder)


class Encoder:
    embedding_dim = 2

    def fingerprint(self):
        return "test-space"

    def encode_images(self, images):
        import numpy as np

        return np.array([[1, 0] for _ in images], dtype=np.float32)

    def encode_texts(self, texts):
        import numpy as np

        return np.array([[1, 0] for _ in texts], dtype=np.float32)
