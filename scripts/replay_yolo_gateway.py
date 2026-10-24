"""Explicit CPU YOLO gateway for offline ROS replay; never used by the OAK driver.

VM -> gateway (YOLOv6 OpenVINO CPU) -> existing RADIO backend. Every detection is
computed from the same JPEG forwarded downstream, never from a neighbouring frame.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import threading
import time
import urllib.error
import urllib.request
import uuid

import cv2
import numpy as np
import openvino as ov
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from slamibot_fieldmind.detect import YoloV6Detector
from slamibot_fieldmind.protocol import MAX_JPEG_BYTES
from slamibot_fieldmind.yolo_postprocess import overlap, policy_fingerprint


class ReplayYolo:
    def __init__(self, xml, roi=0.15, policy=None, verify_ambiguous=True):
        self.decoder = YoloV6Detector(roi_inset=roi, postprocess_policy=policy)
        self.roi = roi
        self.verify_ambiguous = verify_ambiguous
        self.parameters = dict(roi_inset=roi, score_formula="objectness * class_score",
                               decoder_version="yolov6r1_objcls_v2", candidate_threshold=self.decoder.candidate_threshold,
                               native_confidence_threshold=self.decoder.confidence_threshold,
                               native_iou_threshold=self.decoder.iou_threshold,
                               postprocess=self.decoder.policy, verify_ambiguous_flip=verify_ambiguous)
        weights = xml.with_suffix(".bin")
        self.fingerprint = "yolov6r1-openvino-cpu:" + hashlib.sha256(
            xml.read_bytes() + weights.read_bytes() + json.dumps(self.parameters, sort_keys=True).encode()
        ).hexdigest()
        self.core = ov.Core()
        self.model = self.core.compile_model(
            self.core.read_model(str(xml), str(weights)), "CPU",
            {"INFERENCE_NUM_THREADS": 2, "PERFORMANCE_HINT": "LATENCY", "INFERENCE_PRECISION_HINT": "f32"},
        )
        self.outputs = sorted(self.model.outputs, key=lambda output: int(output.shape[2]), reverse=True)
        shapes = [list(output.shape) for output in self.outputs]
        if shapes != [[1, 85, 36, 64], [1, 85, 18, 32], [1, 85, 9, 16]]:
            raise ValueError("Unexpected YOLOv6 IR heads: " + repr(shapes))
        self.lock = threading.Lock()
        self.count = 0
        self.total_ms = 0.0
        self.flip_checks = 0

    def detect(self, jpeg):
        return self.detect_observation(jpeg)[:2]

    def detect_observation(self, jpeg):
        image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Cannot decode image")
        height, width = image.shape[:2]
        r = self.roi
        image = image[int(height*r):int(height*(1-r)), int(width*r):int(width*(1-r))]
        tensor = cv2.resize(image, (512, 288)).astype(np.float32).transpose(2, 0, 1)[None]
        with self.lock:
            started = time.perf_counter()
            raw = self.model([tensor])
            values = [raw[output] for output in self.outputs]
            detections = [d.to_dict() for d in self.decoder.decode_tensors(values)]
            raw_detections = copy.deepcopy(self.decoder.last_raw_detections)
            audit = copy.deepcopy(self.decoder.last_postprocess)
            uncertain = {"bed", "dining table", "refrigerator", "microwave", "train", "surfboard", "suitcase"}
            needs_check = self.verify_ambiguous and any(d["label"] in uncertain for d in detections)
            audit["flip_verified"] = needs_check
            if needs_check:
                flipped = self.model([tensor[..., ::-1].copy()])
                reference = [d.to_dict() for d in self.decoder.decode_tensors([flipped[o] for o in self.outputs])]
                for d in reference:
                    x1, y1, x2, y2 = d["bbox_norm"]
                    d["bbox_norm"] = [1-x2, y1, 1-x1, y2]
                accepted = []
                for item in detections:
                    supported = item["label"] not in uncertain or any(
                        item["label"] == other["label"] and overlap(item["bbox_norm"], other["bbox_norm"])[0] >= 0.50
                        for other in reference
                    )
                    if supported:
                        accepted.append(item)
                    else:
                        audit["rejected"].append(dict(reason="flip_not_supported", detection=item))
                detections = accepted
                self.flip_checks += 1
            audit["kept_count"] = len(detections)
            elapsed = (time.perf_counter()-started)*1000
            self.count += 1
            self.total_ms += elapsed
        return detections, elapsed, dict(detections_raw=raw_detections,
            detection_postprocess=audit, detector_parameters=copy.deepcopy(self.parameters))


def multipart(metadata, jpeg):
    boundary = uuid.uuid4().hex
    body = (("--"+boundary+'\r\nContent-Disposition: form-data; name="metadata"\r\n\r\n').encode()
            + json.dumps(metadata, allow_nan=False).encode()
            + ("\r\n--"+boundary+'\r\nContent-Disposition: form-data; name="image"; filename="frame.jpg"\r\nContent-Type: image/jpeg\r\n\r\n').encode()
            + jpeg + ("\r\n--"+boundary+"--\r\n").encode())
    return body, "multipart/form-data; boundary="+boundary


def create_app(detector, upstream):
    def forward(path, body=None, content_type=None):
        headers = {"Content-Type": content_type} if content_type else {}
        request = urllib.request.Request(upstream.rstrip("/")+path, data=body, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                return Response(response.read(), status_code=response.status, media_type="application/json")
        except urllib.error.HTTPError as error:
            return Response(error.read(), status_code=error.code, media_type="application/json")

    async def health(request):
        # Validation runner reads the upstream model/queue state through this route.
        return await run_in_threadpool(forward, "/v1/health")

    async def sessions(request):
        return await run_in_threadpool(forward, "/v1/sessions", await request.body(), "application/json")

    async def observations(request):
        length = int(request.headers.get("content-length", "0"))
        if not 0 < length <= MAX_JPEG_BYTES+1024*1024:
            return JSONResponse({"error": "Missing or excessive Content-Length"}, status_code=413)
        try:
            async with request.form(max_part_size=MAX_JPEG_BYTES) as form:
                metadata = json.loads(form["metadata"])
                jpeg = await form["image"].read()
            result = await run_in_threadpool(getattr(detector, "detect_observation", detector.detect), jpeg)
            detections, elapsed = result[:2]
            if len(result) == 3:
                metadata.update(result[2])
            metadata.update(
                source_detection_status=metadata.get("detection_status"),
                detection_status="available", detections=detections,
                detector_fingerprint=detector.fingerprint, roi_inset=detector.roi,
                detection_execution="host_cpu_openvino_replay", detection_inference_ms=elapsed,
            )
            body, content_type = multipart(metadata, jpeg)
            return await run_in_threadpool(forward, "/v1/observations", body, content_type)
        except Exception as error:
            # Never convert inference failure into an apparently empty detection list.
            return JSONResponse({"error": str(error)}, status_code=503)

    async def metrics(request):
        with detector.lock:
            stats = dict(frames=detector.count, total_inference_ms=detector.total_ms,
                         execution="host_cpu_openvino_replay", detector_fingerprint=detector.fingerprint,
                         flip_checks=detector.flip_checks, detector_parameters=detector.parameters)
        return JSONResponse(stats)

    return Starlette(routes=[Route("/v1/health", health), Route("/v1/sessions", sessions, methods=["POST"]),
                            Route("/v1/observations", observations, methods=["POST"]), Route("/metrics", metrics)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--upstream", default="http://127.0.0.1:8765")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--postprocess-config", type=Path, help="JSON overrides for precision policy")
    parser.add_argument("--no-flip-verification", action="store_true", help="Disable CPU ambiguous-class mirror check")
    args = parser.parse_args()
    import uvicorn
    policy = json.loads(args.postprocess_config.read_text(encoding="utf8")) if args.postprocess_config else None
    uvicorn.run(create_app(ReplayYolo(args.model, policy=policy, verify_ambiguous=not args.no_flip_verification), args.upstream), host=args.host, port=args.port)
