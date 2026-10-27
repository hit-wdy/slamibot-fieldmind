"""Client for the GPU encoder sidecar (`encoder_service`).

`AlignedEncoder` needs CUDA torch, and on JetPack 5 the only CUDA torch build is
CPython 3.8 while the rest of FieldMind needs CPython 3.10+. `HttpEncoder` is the
drop-in replacement: it exposes the same surface the backend and query paths use
(`encode_images`, `encode_texts`, `fingerprint`, `embedding_dim`,
`resize_multiple`, `environment`, `load_report`) but forwards the work to the
sidecar over HTTP.

Vectors are the sidecar's own bytes, so a collection written through this class is
identical to one written by an in-process `AlignedEncoder` with the same config.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from PIL import Image

@dataclass
class RemoteLoadReport:
    """The subset of `LoadReport` the backend and health endpoints read."""

    version: str = ""
    adaptor: str = ""
    checkpoint: str = ""
    checkpoint_sha256: str = ""
    source_commit: str = ""
    checkpoint_dtype: str = ""
    model_dtype: str = ""
    device: str = ""
    text_encoder_device: str = ""
    text_encoder_dtype: str = ""
    parameter_count: int = 0
    warnings: list[str] = field(default_factory=list)


class HttpEncoder:
    """An `AlignedEncoder` stand-in backed by the sidecar's HTTP API."""

    def __init__(
        self,
        url: str = "http://127.0.0.1:8791",
        timeout: float = 120.0,
        verify_fingerprint: bool = True,
        expected_fingerprint: str | None = None,
    ) -> None:
        self.url = url.rstrip("/")
        self.timeout = timeout
        self._request_bytes = 0
        self._request_seconds = 0.0
        health = self._get("/health")
        if int(health.get("protocol", 0)) != 1:
            raise RuntimeError(
                f"encoder sidecar at {self.url} speaks protocol {health.get('protocol')!r}, "
                "this client speaks 1"
            )
        self._fingerprint = str(health["fingerprint"])
        self._embedding_dim = int(health["embedding_dim"])
        self._resize_multiple = int(health["resize_multiple"])
        self.environment = dict(health.get("environment") or {})
        self.load_report = RemoteLoadReport(**(health.get("load_report") or {}))
        if verify_fingerprint and expected_fingerprint:
            if self._fingerprint != expected_fingerprint:
                raise RuntimeError(
                    "the encoder sidecar produces a different vector space than the index:\n"
                    f"  index   {expected_fingerprint}\n"
                    f"  sidecar {self._fingerprint}\n"
                    "Images and text would not be comparable. Point the sidecar at the same "
                    "model config the index was built with, or rebuild the collection."
                )

    # -- surface shared with AlignedEncoder -------------------------------------

    def fingerprint(self) -> str:
        return self._fingerprint

    def fingerprint_short(self) -> str:
        return self._fingerprint[:16]

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def resize_multiple(self) -> int:
        return self._resize_multiple

    @property
    def stats(self) -> dict:
        return {"bytes": self._request_bytes, "seconds": round(self._request_seconds, 3)}

    def encode_images(self, images: Any, channel_order: str = "rgb") -> np.ndarray:
        """Encode PIL images or arrays over the raw-pixel wire format.

        The pixels travel uncompressed so the sidecar computes the same vectors an
        in-process `AlignedEncoder` would (verified bit-identical for float32).
        A 16:9 frame is ~3 MB, which loopback handles in ~1 ms; pass
        `channel_order='bgr'` for BGR arrays, exactly like `AlignedEncoder`.
        """
        single = _is_single(images)
        batch = [images] if single else list(images)
        payload = [_encode_raw(image, channel_order) for image in batch]
        started = time.monotonic()
        response = self._post(
            "/encode_images",
            {"images": payload, "encoding": "raw", "channel_order": channel_order},
        )
        self._request_seconds += time.monotonic() - started
        vectors = np.asarray(response["vectors"], dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(batch):
            raise RuntimeError(
                f"sidecar returned {vectors.shape} for {len(batch)} images"
            )
        return vectors

    def encode_texts(self, texts: Sequence[str] | str) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        batch = [str(text) for text in texts]
        if not batch:
            raise ValueError("no texts given")
        if any(not text.strip() for text in batch):
            raise ValueError("texts must not be empty")
        started = time.monotonic()
        response = self._post("/encode_texts", {"texts": batch})
        self._request_seconds += time.monotonic() - started
        vectors = np.asarray(response["vectors"], dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(batch):
            raise RuntimeError(
                f"sidecar returned {vectors.shape} for {len(batch)} texts"
            )
        return vectors

    def close(self) -> None:
        """Nothing to release: every call is its own connection."""

    # -- transport --------------------------------------------------------------

    def _get(self, path: str) -> dict:
        request = urllib.request.Request(self.url + path, method="GET")
        return self._send(request)

    def _post(self, path: str, payload: dict) -> dict:
        body = json.dumps(payload).encode("utf-8")
        self._request_bytes += len(body)
        request = urllib.request.Request(
            self.url + path,
            data=body,
            method="POST",
            headers={"Content-Type": "application/json; charset=utf-8"},
        )
        return self._send(request)

    def _send(self, request: urllib.request.Request) -> dict:
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", "replace")
            raise RuntimeError(
                f"encoder sidecar returned HTTP {error.code} for {request.full_url}: {detail}"
            ) from error
        except urllib.error.URLError as error:
            raise RuntimeError(
                f"cannot reach the encoder sidecar at {self.url}: {error.reason}. "
                "Start it with: python -m slamibot_fieldmind.encoder_service"
            ) from error


def _encode_raw(image: Any, channel_order: str) -> dict:
    import base64

    pil = _to_rgb_pil(image, channel_order)
    array = np.asarray(pil, dtype=np.uint8)
    return {
        "data": base64.b64encode(array.tobytes()).decode("ascii"),
        "shape": [int(v) for v in array.shape],
    }


def _to_rgb_pil(image: Any, channel_order: str = "rgb") -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, np.ndarray):
        array = image
        if array.dtype != np.uint8:
            raise TypeError(f"expected a uint8 array, got {array.dtype}")
        if array.ndim == 2:
            array = np.repeat(array[:, :, None], 3, axis=2)
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError(f"expected an HxW or HxWx3 array, got shape {array.shape}")
        if channel_order == "bgr":
            array = array[:, :, ::-1]
        elif channel_order != "rgb":
            raise ValueError(f"channel_order must be 'rgb' or 'bgr', got {channel_order!r}")
        return Image.fromarray(np.ascontiguousarray(array), mode="RGB")
    raise TypeError(f"unsupported image type: {type(image).__name__}")


def _is_single(images: Any) -> bool:
    """A numpy array is one image (2-D grey or 3-D HWC), never a batch of rows."""
    if isinstance(images, np.ndarray):
        return images.ndim in (2, 3)
    return not isinstance(images, (list, tuple))


def connect(url: str, expected_fingerprint: str | None = None, timeout: float = 120.0) -> HttpEncoder:
    """Wait for the sidecar to finish loading, then return a client for it."""
    deadline = time.monotonic() + timeout
    last: Exception | None = None
    while time.monotonic() < deadline:
        try:
            return HttpEncoder(
                url, timeout=timeout, expected_fingerprint=expected_fingerprint
            )
        except Exception as error:  # noqa: BLE001 - the sidecar may not be up yet
            last = error
            time.sleep(1.0)
    raise RuntimeError(f"encoder sidecar at {url} never became ready: {last}")
