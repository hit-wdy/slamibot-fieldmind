"""YOLO object detection on the OAK VPU, alongside RADIO on the host GPU.

This is what makes "YOLO and RADIO at the same time" cheap: the network runs on
the device's SHAVEs, so the host GPU is left entirely to RADIO. Nothing here
competes for the GPU.

Only the detection *results* are kept (label, confidence, box) -- no images, per
the requirement. Boxes are reported in normalised full-frame coordinates, i.e.
relative to the frame RADIO embeds, not to the crop the network actually sees.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from .detection_types import COCO_CLASSES, Detection
from .detection_types import summarise as summarise, detections_to_json as detections_to_json
from .paths import PROJECT_ROOT
from .yolo_postprocess import decode_heads, filter_detections, load_policy, policy_fingerprint

DEFAULT_ARCHIVE = PROJECT_ROOT / "models" / "yolov6" / "yolov6nr1_coco_512x288_rvc2.tar.xz"
# Their tuned value: crops the strongly distorted fisheye border before inference.
DEFAULT_ROI_INSET = 0.15
OUTPUT_TENSORS = ("259", "483", "707")
STRIDES = (8, 16, 32)


class YoloV6Detector:
    """A YOLOv6 RVC2 network attached to a camera output.

    `attach` wires an ImageManip crop plus the network into an existing pipeline
    and returns the queue the caller should read results from.
    """

    def __init__(
        self,
        archive: Path | str = DEFAULT_ARCHIVE,
        roi_inset: float = DEFAULT_ROI_INSET,
        candidate_threshold: float = 0.20,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_detections: int = 300,
        input_size: tuple[int, int] = (512, 288),
        queue_size: int = 8,
        postprocess_policy: dict | None = None,
    ) -> None:
        if not 0.0 <= roi_inset < 0.45:
            raise ValueError("roi_inset must be in [0, 0.45)")
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1]")
        self.archive = Path(archive)
        if not self.archive.is_file():
            raise FileNotFoundError(
                f"YOLO archive not found: {self.archive}\n"
                "run: python scripts/fetch_assets.py --asset yolov6_rvc2"
            )
        self.roi_inset = roi_inset
        self.candidate_threshold = candidate_threshold
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections
        self.input_size = input_size  # (width, height)
        self.policy = load_policy(postprocess_policy)
        self.last_raw_detections = []
        self.last_postprocess = None

    def attach(self, pipeline: Any, camera_output: Any, dai: Any) -> Any:
        """Add the crop + network to `pipeline`, fed by `camera_output`."""
        roi_size = 1.0 - 2.0 * self.roi_inset
        manip = pipeline.create(dai.node.ImageManip)
        manip.initialConfig.addCrop(
            dai.Rect(self.roi_inset, self.roi_inset, roi_size, roi_size, True), True
        )
        manip.initialConfig.setOutputSize(
            self.input_size[0], self.input_size[1], dai.ImageManipConfig.ResizeMode.STRETCH
        )
        manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
        manip.setMaxOutputFrameSize(self.input_size[0] * self.input_size[1] * 3)
        camera_output.link(manip.inputImage)

        archive = dai.NNArchive(str(self.archive))
        names = [output.name for output in archive.getConfig().model.outputs]
        if names != list(OUTPUT_TENSORS):
            raise RuntimeError(
                f"unexpected YOLOv6 output tensors {names}; expected {list(OUTPUT_TENSORS)}. "
                "The decoder below is written for this archive only."
            )
        network = pipeline.create(dai.node.NeuralNetwork).build(manip.out, archive)
        network.input.setBlocking(False)
        return network.out.createOutputQueue(maxSize=8, blocking=False)

    def decode(self, message: Any, dai: Any) -> list[Detection]:
        values = [
            message.getTensor(name, dequantize=True, storageOrder=dai.TensorInfo.StorageOrder.NCHW)
            for name in OUTPUT_TENSORS
        ]
        return self.decode_tensors(values)

    def decode_tensors(self, values: Sequence[Any]) -> list[Detection]:
        """Decode ordered YOLO heads from either the VPU or the replay CPU IR."""
        self.last_raw_detections = decode_heads(
            values, COCO_CLASSES, roi=self.roi_inset, candidate=self.candidate_threshold,
            confidence=self.confidence_threshold, iou=self.iou_threshold, max_det=self.max_detections,
        )
        filtered, self.last_postprocess = filter_detections(self.last_raw_detections, self.roi_inset, self.policy)
        return [Detection(d["label"], d["label_id"], d["confidence"], tuple(d["bbox_norm"])) for d in filtered]

    def _to_full_frame(self, box: Sequence[float], score: float, label_id: int) -> Detection:
        """Map a box from the cropped network input back to the full camera frame.

        The crop is a straight scale of the centre `1 - 2*roi_inset` of the frame
        (the source is 16:9 and the network input is 16:9, stretched), so the map
        is a scale and offset on each axis. Without this, a box would be reported
        against an image RADIO never saw.
        """
        cx, cy, bw, bh = (float(v) for v in box)
        scale = 1.0 - 2.0 * self.roi_inset
        left = self.roi_inset + (cx - bw / 2) * scale
        top = self.roi_inset + (cy - bh / 2) * scale
        right = self.roi_inset + (cx + bw / 2) * scale
        bottom = self.roi_inset + (cy + bh / 2) * scale
        return Detection(
            label=COCO_CLASSES[label_id],
            label_id=label_id,
            confidence=score,
            bbox_norm=(
                max(0.0, min(1.0, left)),
                max(0.0, min(1.0, top)),
                max(0.0, min(1.0, right)),
                max(0.0, min(1.0, bottom)),
            ),
        )

    def describe(self) -> dict:
        return {
            "network": "yolov6nr1-coco-512x288",
            "runs_on": "RVC2 VPU (device)",
            "archive": self.archive.name,
            "input_size_wh": list(self.input_size),
            "roi_inset": self.roi_inset,
            "candidate_threshold": self.candidate_threshold,
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold,
            "n_classes": len(COCO_CLASSES),
            "score_formula": "objectness * class_score",
            "postprocess": self.policy,
            "postprocess_sha256": policy_fingerprint(self.policy),
        }
