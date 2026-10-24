"""Replay must preserve frame identity and distinguish no detections from failure."""

import importlib.util
import json
from pathlib import Path
from email import policy
from email.parser import BytesParser

import pytest
from starlette.testclient import TestClient

pytest.importorskip("openvino")
pytest.importorskip("cv2")
spec = importlib.util.spec_from_file_location(
    "replay_yolo_gateway", Path(__file__).resolve().parents[1] / "scripts/replay_yolo_gateway.py"
)
gateway = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gateway)


def test_no_objects_is_available_and_exact_jpeg_is_forwarded(monkeypatch):
    class Detector:
        roi = 0.15
        fingerprint = "test-replay-model"

        def detect(self, jpeg):
            assert jpeg == b"original-exact-frame"
            return [], 25.0

    requests = []

    class Upstream:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return b'{"accepted":true}'

    def forward(request, **kwargs):
        requests.append(request)
        return Upstream()

    monkeypatch.setattr(gateway.urllib.request, "urlopen", forward)
    metadata = dict(source_sequence=123, source_stamp_ns=456, detection_status="unavailable")
    with TestClient(gateway.create_app(Detector(), "http://backend")) as client:
        response = client.post("/v1/observations", data={"metadata": json.dumps(metadata)},
                               files={"image": ("frame.jpg", b"original-exact-frame", "image/jpeg")})
    assert response.status_code == 200 and len(requests) == 1
    request = requests[0]
    message = BytesParser(policy=policy.default).parsebytes(
        ("Content-Type: " + request.get_header("Content-type") + "\r\n\r\n").encode() + request.data
    )
    parts = {p.get_param("name", header="content-disposition"): p.get_payload(decode=True)
             for p in message.iter_parts()}
    output = json.loads(parts["metadata"])
    assert parts["image"] == b"original-exact-frame"
    assert output["source_sequence"] == 123 and output["source_stamp_ns"] == 456
    assert output["detection_status"] == "available" and output["detections"] == []
    assert output["source_detection_status"] == "unavailable"
    assert output["detection_execution"] == "host_cpu_openvino_replay"


def test_inference_failure_never_becomes_empty_success(monkeypatch):
    class Detector:
        def detect(self, jpeg):
            raise RuntimeError("inference failed")

    def unexpected_forward(*args, **kwargs):
        pytest.fail("failed inference must not forward a success observation")

    monkeypatch.setattr(gateway.urllib.request, "urlopen", unexpected_forward)
    with TestClient(gateway.create_app(Detector(), "http://backend")) as client:
        response = client.post("/v1/observations", data={"metadata": "{}"},
                               files={"image": ("frame.jpg", b"jpeg", "image/jpeg")})
    assert response.status_code == 503
    assert "inference failed" in response.json()["error"]


@pytest.mark.parametrize("mirror_support", [False, True])
def test_ambiguous_class_requires_matching_mirrored_evidence(mirror_support):
    import io
    import threading
    import numpy as np
    from PIL import Image
    from slamibot_fieldmind.detection_types import Detection

    original = Detection("refrigerator", 72, 0.96, (0.1, 0.2, 0.3, 0.8))
    reference = Detection("refrigerator", 72, 0.95, (0.7, 0.2, 0.9, 0.8))
    calls = []

    class Model:
        def __call__(self, inputs):
            calls.append(inputs[0].copy())
            return {"head": len(calls)}

    class Decoder:
        def decode_tensors(self, values):
            detections = [original] if values[0] == 1 else ([reference] if mirror_support else [])
            self.last_raw_detections = [d.to_dict() for d in detections]
            self.last_postprocess = dict(rejected=[], raw_count=len(detections))
            return detections

    detector = gateway.ReplayYolo.__new__(gateway.ReplayYolo)
    detector.roi, detector.verify_ambiguous = 0.15, True
    detector.lock, detector.model, detector.decoder = threading.Lock(), Model(), Decoder()
    detector.outputs, detector.parameters = ["head"], {}
    detector.count = detector.flip_checks = 0
    detector.total_ms = 0
    jpeg = io.BytesIO()
    pixels = np.tile(np.arange(64, dtype=np.uint8)[None, :, None], (32, 1, 3))
    Image.fromarray(pixels).save(jpeg, format="JPEG")
    result, _, audit = detector.detect_observation(jpeg.getvalue())
    np.testing.assert_array_equal(calls[1], calls[0][..., ::-1])
    assert bool(result) is mirror_support
    assert audit["detections_raw"] == [original.to_dict()]
    assert detector.flip_checks == 1
    assert bool(audit["detection_postprocess"]["rejected"]) is not mirror_support
