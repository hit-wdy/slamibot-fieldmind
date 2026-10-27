"""HTTP sidecar that serves the RADIO encoder from a separate interpreter.

JetPack 5 (L4T 35.x) is the only supported OS on Orin NX boards that still run
CUDA 11.4, and NVIDIA publishes CUDA-enabled torch wheels for it only for CPython
3.8 (`torch-2.1.0a0+...cp38`). The MCP stack needs CPython 3.10+, so the two
cannot share an interpreter. This process is the GPU half: it runs the encoder in
a Python 3.8 virtualenv and speaks a small JSON protocol to the Python 3.11
backend through `remote_encoder.HttpEncoder`.

Run it with the interpreter that owns the CUDA torch build:

    python -m slamibot_fieldmind.encoder_service --model-config configs/jetson.yaml

Images travel as JPEG bytes and are decoded here, so preprocessing stays in one
place and the vectors are the ones `AlignedEncoder` would have produced locally.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import numpy as np
from PIL import Image

PROTOCOL_VERSION = 1


class EncoderHandler(BaseHTTPRequestHandler):
    """Routes for the encoder sidecar; the model lives in `self.server.encoder`."""

    protocol_version = "HTTP/1.1"
    server_version = "fieldmind-encoder/1"

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: D102 - stdlib hook
        if self.server.verbose:  # type: ignore[attr-defined]
            sys.stderr.write(f"{self.address_string()} {fmt % args}\n")

    # -- plumbing ---------------------------------------------------------------

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            raise ValueError("empty request body")
        if length > 64 * 1024 * 1024:
            raise ValueError("request body too large")
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _send(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _guard(self, handler):
        try:
            return self._send(200, handler())
        except ValueError as error:
            return self._send(400, {"error": str(error)})
        except Exception as error:  # noqa: BLE001 - report, never kill the server
            return self._send(500, {"error": f"{type(error).__name__}: {error}"})

    # -- routes -----------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        if self.path != "/health":
            return self._send(404, {"error": "not found"})
        encoder = self.server.encoder  # type: ignore[attr-defined]
        return self._send(
            200,
            {
                "protocol": PROTOCOL_VERSION,
                "fingerprint": encoder.fingerprint(),
                "embedding_dim": encoder.embedding_dim,
                "resize_multiple": encoder.resize_multiple,
                "environment": encoder.environment,
                "load_report": _report_dict(encoder.load_report),
            },
        )

    def do_POST(self) -> None:  # noqa: N802 - stdlib naming
        if self.path == "/encode_images":
            return self._guard(self._encode_images)
        if self.path == "/encode_texts":
            return self._guard(self._encode_texts)
        return self._send(404, {"error": "not found"})

    def _encode_images(self) -> dict:
        payload = self._read_json()
        images = payload.get("images") or []
        if not isinstance(images, list) or not images:
            raise ValueError("images must be a non-empty list")
        order = payload.get("channel_order", "rgb")
        encoding = payload.get("encoding", "raw")
        decoded = [_decode_image(item, encoding) for item in images]
        vectors = self.server.encoder.encode_images(  # type: ignore[attr-defined]
            decoded, channel_order=order
        )
        return {"vectors": vectors.tolist(), "count": int(vectors.shape[0])}

    def _encode_texts(self) -> dict:
        payload = self._read_json()
        texts = payload.get("texts") or []
        if not isinstance(texts, list) or not texts:
            raise ValueError("texts must be a non-empty list of strings")
        vectors = self.server.encoder.encode_texts(texts)  # type: ignore[attr-defined]
        return {"vectors": vectors.tolist(), "count": int(vectors.shape[0])}


def _decode_image(item: Any, encoding: str) -> Image.Image:
    """Rebuild one image from the wire form.

    `raw` carries the exact uint8 pixels so the sidecar produces bit-identical
    vectors to an in-process encode; `jpeg` is a smaller fallback for callers that
    accept the ~1e-3 cosine drift lossy compression introduces.
    """
    if encoding not in ("raw", "jpeg"):
        raise ValueError(f"images.encoding must be 'raw' or 'jpeg', got {encoding!r}")
    if not isinstance(item, dict):
        raise ValueError("each image must be an object")
    try:
        data = base64.b64decode(item["data"], validate=True)
    except Exception as error:  # noqa: BLE001
        raise ValueError(f"image data is not valid base64: {error}") from error

    if encoding == "jpeg":
        with Image.open(io.BytesIO(data)) as handle:
            return handle.convert("RGB")

    shape = item.get("shape")
    if not isinstance(shape, list) or len(shape) != 3 or shape[2] != 3:
        raise ValueError(f"raw image needs shape [h, w, 3], got {shape!r}")
    height, width, _ = (int(v) for v in shape)
    expected = height * width * 3
    if len(data) != expected:
        raise ValueError(
            f"raw image payload is {len(data)} bytes, expected {expected} for {shape}"
        )
    array = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)
    return Image.fromarray(np.ascontiguousarray(array), mode="RGB")


def _report_dict(report: Any) -> dict:
    if report is None:
        return {}
    keys = (
        "version",
        "adaptor",
        "checkpoint",
        "checkpoint_sha256",
        "source_commit",
        "checkpoint_dtype",
        "model_dtype",
        "device",
        "text_encoder_device",
        "text_encoder_dtype",
        "parameter_count",
        "warnings",
    )
    return {key: getattr(report, key, None) for key in keys}


def serve(encoder: Any, host: str, port: int, verbose: bool = False) -> None:
    server = ThreadingHTTPServer((host, port), EncoderHandler)
    server.daemon_threads = True
    server.encoder = encoder  # type: ignore[attr-defined]
    server.verbose = verbose  # type: ignore[attr-defined]
    print(
        f"encoder sidecar on http://{host}:{port} "
        f"(fingerprint {encoder.fingerprint()[:16]}, dim {encoder.embedding_dim}, "
        f"device {encoder.load_report.device})",
        file=sys.stderr,
        flush=True,
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-config", help="YAML model config; defaults to configs/default.yaml")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8791)
    parser.add_argument("--no-verify-checkpoint", action="store_true", help="skip the SHA-256 pass")
    parser.add_argument("--verbose", action="store_true", help="log every request")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from .config import Config
    from .encoder import AlignedEncoder

    config = Config.from_yaml(args.model_config) if args.model_config else Config.default()
    started = time.monotonic()
    encoder = AlignedEncoder(config, verify_checkpoint=not args.no_verify_checkpoint)
    print(f"encoder ready in {time.monotonic() - started:.1f}s", file=sys.stderr, flush=True)
    serve(encoder, args.host, args.port, verbose=args.verbose)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
