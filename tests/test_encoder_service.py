"""The GPU encoder sidecar and its client must agree bit for bit.

On JetPack 5 the CUDA torch wheels are CPython 3.8 while FieldMind needs 3.10+,
so the encoder runs in a separate process. That is only acceptable if the bytes
crossing the wire change nothing: `HttpEncoder` has to produce the same vectors
as an in-process `AlignedEncoder`, or an index built one way could not be read
the other way.
"""

from __future__ import annotations

import json
import threading

import numpy as np
import pytest

from slamibot_fieldmind import encoder_service, remote_encoder

# A stand-in encoder: the transport is what is under test, not RADIO itself.
# `encode_images` echoes a deterministic function of the pixels so a corrupted
# payload cannot pass by accident.
class FakeReport:
    version = "fake"
    adaptor = "clip"
    checkpoint = "/nonexistent"
    checkpoint_sha256 = "0" * 64
    source_commit = "0" * 40
    checkpoint_dtype = "torch.float32"
    model_dtype = "torch.float32"
    device = "cuda:0"
    text_encoder_device = "cpu"
    text_encoder_dtype = "torch.float32"
    parameter_count = 1
    warnings: list[str] = []


class FakeEncoder:
    def __init__(self) -> None:
        self.load_report = FakeReport()
        self.environment = {"device": "cuda:0", "device_name": "Orin"}
        self.calls = 0
        self.lock = threading.Lock()

    def fingerprint(self) -> str:
        return "ab" * 32

    @property
    def embedding_dim(self) -> int:
        return 4

    @property
    def resize_multiple(self) -> int:
        return 16

    def encode_images(self, images, channel_order: str = "rgb"):
        with self.lock:
            self.calls += 1
        rows = []
        for image in images:
            array = np.asarray(image, dtype=np.float32) / 255.0
            # A cheap summary that depends on every pixel, so a truncated or
            # re-compressed payload produces a different vector.
            rows.append(
                [
                    array.mean(),
                    array[:, :, 0].mean(),
                    array[:, :, 1].std(),
                    array[:, :, 2].max(),
                ]
            )
        return np.asarray(rows, dtype=np.float32)

    def encode_texts(self, texts):
        return np.asarray([[len(text), 1.0, 2.0, 3.0] for text in texts], dtype=np.float32)


@pytest.fixture(scope="module")
def sidecar():
    encoder = FakeEncoder()
    server = encoder_service.ThreadingHTTPServer(
        ("127.0.0.1", 0), encoder_service.EncoderHandler
    )
    server.daemon_threads = True
    server.encoder = encoder
    server.verbose = False
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture
def client(sidecar) -> remote_encoder.HttpEncoder:
    host, port = sidecar.server_address[:2]
    return remote_encoder.HttpEncoder(f"http://{host}:{port}", timeout=30)


def _frame(seed: int = 0, size=(64, 96)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (*size, 3), dtype=np.uint8)


def test_health_reports_the_sidecar_identity(client):
    assert client.fingerprint() == "ab" * 32
    assert client.embedding_dim == 4
    assert client.resize_multiple == 16
    assert client.environment["device_name"] == "Orin"
    assert client.load_report.device == "cuda:0"


def test_unknown_protocol_is_refused(sidecar):
    original = encoder_service.EncoderHandler.do_GET

    def other_protocol(self):  # noqa: ANN001, ANN202
        return self._send(
            200,
            {
                "protocol": 999,
                "fingerprint": "ab" * 32,
                "embedding_dim": 4,
                "resize_multiple": 16,
                "environment": {},
                "load_report": {},
            },
        )

    try:
        encoder_service.EncoderHandler.do_GET = other_protocol
        host, port = sidecar.server_address[:2]
        with pytest.raises(RuntimeError, match="speaks protocol"):
            remote_encoder.HttpEncoder(f"http://{host}:{port}", timeout=10)
    finally:
        encoder_service.EncoderHandler.do_GET = original


def test_images_are_encoded_by_the_sidecar(client, sidecar):
    from PIL import Image

    frame = _frame()
    vectors = client.encode_images([frame])
    expected = sidecar.encoder.encode_images([Image.fromarray(frame, mode="RGB")])
    # The fake echoes pixel statistics, so an identical vector proves the raw
    # pixels survived the round trip byte for byte.
    assert np.array_equal(vectors, expected)


def test_text_round_trip(client):
    vectors = client.encode_texts(["hello", "a longer phrase"])
    assert vectors.shape == (2, 4)
    assert vectors[0, 0] == 5


def test_single_image_and_batch_agree(client):
    frame = _frame(1)
    single = client.encode_images(frame)
    batch = client.encode_images([frame])
    assert single.shape == batch.shape == (1, 4)
    assert np.array_equal(single, batch)


def test_fingerprint_mismatch_is_rejected(sidecar):
    host, port = sidecar.server_address[:2]
    with pytest.raises(RuntimeError, match="different vector space"):
        remote_encoder.HttpEncoder(
            f"http://{host}:{port}", expected_fingerprint="cd" * 32, timeout=10
        )


def test_matching_fingerprint_is_accepted(sidecar):
    host, port = sidecar.server_address[:2]
    client = remote_encoder.HttpEncoder(
        f"http://{host}:{port}", expected_fingerprint="ab" * 32, timeout=10
    )
    assert client.fingerprint() == "ab" * 32


def test_unreachable_sidecar_says_how_to_start_it():
    with pytest.raises(RuntimeError, match="encoder_service"):
        remote_encoder.HttpEncoder("http://127.0.0.1:1", timeout=2)


def test_bgr_arrays_are_reordered_before_the_wire(client, sidecar):
    frame = _frame(2)
    client.encode_images([frame[:, :, ::-1]], channel_order="bgr")
    from PIL import Image

    expected = sidecar.encoder.encode_images([Image.fromarray(frame, mode="RGB")])
    assert np.allclose(client.encode_images([frame]), expected)


def test_oversized_payload_is_refused(sidecar):
    import urllib.error
    import urllib.request

    host, port = sidecar.server_address[:2]
    request = urllib.request.Request(
        f"http://{host}:{port}/encode_images",
        data=json.dumps({"images": []}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(request, timeout=10)
    assert error.value.code == 400


def test_raw_payload_shape_mismatch_is_refused(sidecar):
    import base64
    import urllib.error
    import urllib.request

    host, port = sidecar.server_address[:2]
    body = json.dumps(
        {
            "encoding": "raw",
            "images": [
                {
                    "data": base64.b64encode(b"\x00" * 12).decode(),
                    "shape": [4, 4, 3],
                }
            ],
        }
    ).encode()
    request = urllib.request.Request(
        f"http://{host}:{port}/encode_images",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(request, timeout=10)
    assert error.value.code == 400
    assert "expected 48" in error.value.read().decode()
