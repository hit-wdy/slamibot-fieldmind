"""Detection geometry: mapping network boxes back onto the frame RADIO sees.

The network sees a centre crop; the embedding covers the whole frame. Reporting
boxes against the crop would make them uncomparable with the embedding, so this
mapping is the part of the detection path worth pinning down by test.
"""

from __future__ import annotations

import pytest

from slamibot_fieldmind.detect import COCO_CLASSES, Detection, YoloV6Detector, detections_to_json, summarise


def detector(inset: float) -> YoloV6Detector:
    return YoloV6Detector(roi_inset=inset)


def test_crop_centre_maps_to_frame_centre():
    box = detector(0.15)._to_full_frame((0.5, 0.5, 0.0, 0.0), 0.9, 0)
    assert box.bbox_norm == pytest.approx((0.5, 0.5, 0.5, 0.5), abs=1e-9)


def test_crop_origin_maps_to_the_inset():
    """A box at the crop's top-left corner sits at the inset in the full frame."""
    box = detector(0.15)._to_full_frame((0.0, 0.0, 0.0, 0.0), 0.9, 0)
    assert box.bbox_norm == pytest.approx((0.15, 0.15, 0.15, 0.15), abs=1e-9)


def test_zero_inset_maps_cxcywh_to_xyxy():
    """With no crop the only change is the box convention."""
    det = detector(0.0)
    for cx, cy, bw, bh in [(0.1, 0.2, 0.3, 0.4), (0.5, 0.5, 0.2, 0.2), (0.5, 0.5, 1.0, 1.0)]:
        expected = (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)
        clamped = tuple(min(1.0, max(0.0, v)) for v in expected)
        assert det._to_full_frame((cx, cy, bw, bh), 0.5, 0).bbox_norm == pytest.approx(clamped, abs=1e-9)


@pytest.mark.parametrize("inset", [0.0, 0.05, 0.15, 0.3])
def test_mapping_is_monotonic_and_contained(inset):
    det = detector(inset)
    scale = 1.0 - 2 * inset
    for cx in (0.0, 0.25, 0.5, 0.75, 1.0):
        mapped = det._to_full_frame((cx, 0.5, 0.0, 0.0), 0.5, 0).bbox_norm[0]
        assert mapped == pytest.approx(inset + cx * scale, abs=1e-9)
        assert 0.0 <= mapped <= 1.0


def test_labels_come_from_coco():
    det = detector(0.15)._to_full_frame((0.5, 0.5, 0.1, 0.1), 0.8, 7)
    assert det.label == "truck"
    assert det.label == COCO_CLASSES[7]
    assert det.confidence == 0.8


def test_detector_rejects_an_out_of_range_inset():
    with pytest.raises(ValueError):
        YoloV6Detector(roi_inset=0.45)
    with pytest.raises(ValueError):
        YoloV6Detector(roi_inset=-0.1)


def test_missing_archive_is_reported_with_the_fetch_command(tmp_path):
    with pytest.raises(FileNotFoundError, match="fetch_assets"):
        YoloV6Detector(archive=tmp_path / "nope.tar.xz")


def test_summarise_counts_per_label():
    detections = [
        Detection("person", 0, 0.9, (0.1, 0.1, 0.2, 0.2)),
        Detection("person", 0, 0.7, (0.3, 0.3, 0.4, 0.4)),
        Detection("truck", 7, 0.6, (0.5, 0.5, 0.6, 0.6)),
    ]
    assert summarise(detections) == {"person": 2, "truck": 1}
    assert summarise([]) == {}


def test_detections_serialise_round_trip():
    import json

    detections = [Detection("truck", 7, 0.65432, (0.1, 0.2, 0.3, 0.4))]
    payload = json.loads(detections_to_json(detections))
    assert payload == [{"label": "truck", "label_id": 7, "confidence": 0.6543, "bbox_norm": [0.1, 0.2, 0.3, 0.4]}]
