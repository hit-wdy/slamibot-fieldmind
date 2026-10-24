from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from slamibot_fieldmind.annotation import class_colour, draw_detections, load_font
from slamibot_fieldmind.detection_types import COCO_CLASSES
from slamibot_fieldmind.yolo_postprocess import decode_heads, filter_detections, load_policy


def detection(label, score, box):
    return dict(label=label, label_id=COCO_CLASSES.index(label), confidence=score, bbox_norm=list(box))


def test_objectness_really_changes_confidence_and_does_not_mutate_heads():
    heads = [np.zeros((1, 85, 288//s, 512//s), np.float32) for s in (8, 16, 32)]
    heads[0][0, :5, 10, 10] = [0.5, 0.5, 1, 1, 0.9]
    heads[0][0, 5, 10, 10] = 0.95
    snapshots = [h.copy() for h in heads]
    result = decode_heads(heads, COCO_CLASSES, confidence=0.5)
    assert result[0]["confidence"] == pytest.approx(0.855)
    assert decode_heads(heads, COCO_CLASSES, confidence=0.9) == []
    assert decode_heads(heads, COCO_CLASSES, confidence=0.9, score_mode="class_only")
    for h, snapshot in zip(heads, snapshots):
        np.testing.assert_array_equal(h, snapshot)
    heads[0][0, 4, 10, 10] = 0.21
    assert decode_heads(heads, COCO_CLASSES) == []


def test_vehicle_conflict_and_contained_duplicate_removed():
    rows = [detection("car", 0.95, [0.2, 0.2, 0.6, 0.6]),
            detection("truck", 0.9, [0.201, 0.201, 0.599, 0.599]),
            detection("car", 0.8, [0.21, 0.21, 0.45, 0.45])]
    kept, audit = filter_detections(rows)
    assert kept == [rows[0]]
    assert {r["reason"] for r in audit["rejected"]} == {"vehicle_class_conflict", "vehicle_containment"}
    assert len(rows) == 3  # input evidence remains intact


@pytest.mark.parametrize("labels,boxes", [
    (("car", "car"), ([0.1, 0.2, 0.4, 0.6], [0.35, 0.2, 0.65, 0.6])),
    (("car", "car"), ([0.1, 0.1, 0.9, 0.9], [0.3, 0.3, 0.4, 0.4])),
    (("person", "bicycle"), ([0.2, 0.2, 0.6, 0.8], [0.2, 0.2, 0.6, 0.8])),
    (("potted plant", "vase"), ([0.2, 0.2, 0.6, 0.8], [0.2, 0.2, 0.6, 0.8])),
    (("bicycle", "bicycle"), ([0.1, 0.1, 0.7, 0.9], [0.3, 0.3, 0.6, 0.7])),
    (("chair", "chair"), ([0.1, 0.1, 0.7, 0.9], [0.3, 0.3, 0.6, 0.7])),
])
def test_legitimate_overlap_is_preserved(labels, boxes):
    rows = [detection(labels[0], 0.96, boxes[0]), detection(labels[1], 0.95, boxes[1])]
    assert len(filter_detections(rows)[0]) == 2


def test_truncated_surface_guard_is_configurable_and_not_a_class_blacklist():
    suspicious = detection("bed", 0.91, [0.15, 0.5, 0.85, 0.85])
    assert filter_detections([suspicious], roi=0.15)[1]["rejected"][0]["reason"] == "large_truncated_surface"
    assert filter_detections([suspicious], roi=0.15, policy={"large_surface_guard": False})[0]
    assert filter_detections([detection("bed", 0.85, [0.2, 0.3, 0.7, 0.7])], roi=0.15)[0]
    assert filter_detections([detection("bed", 0.99, suspicious["bbox_norm"])], roi=0.15)[0]


def test_close_up_plant_is_not_lost_to_the_vase_label():
    rows = [detection("potted plant", 0.3924, [0.18, 0.15, 0.61, 0.84]),
            detection("vase", 0.908, [0.29, 0.39, 0.47, 0.84])]
    assert {d["label"] for d in filter_detections(rows, roi=0.15)[0]} == {"potted plant", "vase"}


def test_invalid_boxes_and_policy_fail_cleanly():
    bad = [detection("person", 0.9, [0.2, 0.2, 0.1, 0.3]),
           detection("person", float("nan"), [0.1, 0.1, 0.2, 0.2])]
    assert filter_detections(bad)[0] == []
    with pytest.raises(ValueError):
        load_policy({"vehicle_conflict_iou": 1.5})
    with pytest.raises(ValueError):
        load_policy({"invented_option": 1})


def test_class_colours_are_distinct_and_labels_readable_at_image_edges():
    assert len({class_colour(label) for label in COCO_CLASSES}) == len(COCO_CLASSES)
    assert class_colour("person") == class_colour("person")
    draw = ImageDraw.Draw(Image.new("RGB", (1280, 800)))
    bounds = draw.textbbox((0, 0), "potted plant 0.95", font=load_font(27))
    assert bounds[3]-bounds[1] >= 20
    image = Image.new("RGB", (1280, 800))
    draw_detections(image, [detection("person", 0.9, [0, 0, 0.1, 0.2]),
                            detection("potted plant", 0.95, [0.9, 0.8, 1, 1])])
    assert image.getbbox() is not None


def test_ros_and_backend_share_identical_decoder_source():
    root = Path(__file__).resolve().parents[1]
    candidates = [root.parent.parent / path / "src/semantic_perception/src/semantic_bridge/yolo_postprocess.py"
                  for path in ("D360/slamibot_d360_framework", "0801/SLAMIBOT_D360_Framework")]
    ros = next((path for path in candidates if path.exists()), candidates[0])
    if not ros.exists():
        pytest.skip("device framework not present")
    assert ros.read_bytes() == (root / "src/slamibot_fieldmind/yolo_postprocess.py").read_bytes()
