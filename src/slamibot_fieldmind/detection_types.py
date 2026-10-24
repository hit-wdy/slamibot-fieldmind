"""Portable detection records; deliberately independent of DepthAI."""

from dataclasses import dataclass
from typing import Sequence
import json

COCO_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


@dataclass(frozen=True)
class Detection:
    """One detected object, in normalised full-frame xyxy coordinates."""

    label: str
    label_id: int
    confidence: float
    bbox_norm: tuple[float, float, float, float]

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "label_id": self.label_id,
            "confidence": round(self.confidence, 4),
            "bbox_norm": [round(v, 4) for v in self.bbox_norm],
        }


def summarise(detections: Sequence[Detection]) -> dict[str, int]:
    """Counts per label, for cheap filtering without parsing every box."""
    counts: dict[str, int] = {}
    for detection in detections:
        counts[detection.label] = counts.get(detection.label, 0) + 1
    return dict(sorted(counts.items()))


def detections_to_json(detections: Sequence[Detection]) -> str:
    return json.dumps([d.to_dict() for d in detections], sort_keys=True)
