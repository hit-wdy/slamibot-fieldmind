"""YOLOv6 R1 decoding and auditable precision policy, portable to ROS Python 3.8.

This file is mirrored verbatim into semantic_bridge/yolo_postprocess.py.
Upstream R1 NMS scores are objectness * class score, not class score alone.
"""

import copy
import hashlib
import json
import math

import numpy as np


DEFAULT_POLICY = {
    "version": "precision_v1",
    "min_confidence": 0.60,
    "class_thresholds": {
        "person": 0.40, "car": 0.45, "bicycle": 0.40, "chair": 0.40,
        "potted plant": 0.35, "vase": 0.75, "bed": 0.80,
        "dining table": 0.70, "refrigerator": 0.90, "microwave": 0.90,
        "train": 0.90, "surfboard": 0.90, "skateboard": 0.85, "suitcase": 0.80,
    },
    "same_class_iou": 0.45,
    "vehicle_conflict_iou": 0.70,
    "vehicle_containment": 0.88,
    "vehicle_min_area_ratio": 0.30,
    "large_surface_guard": True,
    "large_surface_min_confidence": 0.97,
    "max_detections": 100,
}
VEHICLES = {"car", "truck", "bus", "train"}
TWO_WHEELERS = {"bicycle", "motorcycle"}
LARGE_SURFACES = {"bed", "dining table", "train", "surfboard"}


def load_policy(overrides=None):
    policy = copy.deepcopy(DEFAULT_POLICY)
    if overrides:
        unknown = set(overrides) - set(policy)
        if unknown:
            raise ValueError("Unknown detection policy options: " + repr(sorted(unknown)))
        for key, value in overrides.items():
            if key == "class_thresholds":
                policy[key].update(value)
            else:
                policy[key] = value
    for key in ("min_confidence", "same_class_iou", "vehicle_conflict_iou",
                "vehicle_containment", "vehicle_min_area_ratio", "large_surface_min_confidence"):
        if not isinstance(policy[key], (int, float)) or not math.isfinite(policy[key]) or not 0 <= policy[key] <= 1:
            raise ValueError("Invalid detection policy " + key)
    if any(not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1
           for v in policy["class_thresholds"].values()):
        raise ValueError("Invalid per-class confidence threshold")
    if type(policy["large_surface_guard"]) is not bool:
        raise ValueError("large_surface_guard must be boolean")
    if type(policy["max_detections"]) is not int or not 1 <= policy["max_detections"] <= 1000:
        raise ValueError("max_detections must be 1..1000")
    return policy


def policy_fingerprint(policy):
    return hashlib.sha256(json.dumps(load_policy(policy), sort_keys=True).encode()).hexdigest()


def overlap(a, b):
    aa = max(0.0, a[2]-a[0]) * max(0.0, a[3]-a[1])
    bb = max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1])
    intersection = max(0.0, min(a[2], b[2])-max(a[0], b[0])) * max(0.0, min(a[3], b[3])-max(a[1], b[1]))
    return (intersection/max(aa+bb-intersection, 1e-12),
            intersection/max(min(aa, bb), 1e-12), min(aa, bb)/max(aa, bb, 1e-12))


def filter_detections(detections, roi=0.0, policy=None):
    """Return filtered dicts and rejection reasons. Never mutate caller evidence.

    Containment is restricted to vehicles: nested chairs/bikes and plant/vase
    pairs can be separate, legitimately overlapping objects.
    """
    config = load_policy(policy)
    if not 0 <= roi < 0.45:
        raise ValueError("Invalid ROI inset")
    kept, rejected, candidates = [], [], []
    for index, original in enumerate(detections):
        item = copy.deepcopy(original)
        box = item.get("bbox_norm", [])
        score = item.get("confidence", float("nan"))
        valid = (len(box) == 4 and all(isinstance(x, (int, float)) and math.isfinite(x) for x in box)
                 and isinstance(score, (int, float)) and math.isfinite(score) and 0 <= score <= 1)
        if not valid:
            rejected.append(dict(index=index, reason="invalid_detection", detection=item))
            continue
        box = [min(1.0, max(0.0, float(x))) for x in box]
        item["bbox_norm"] = box
        if box[2] <= box[0] or box[3] <= box[1] or (box[2]-box[0])*(box[3]-box[1]) < 1e-6:
            rejected.append(dict(index=index, reason="degenerate_box", detection=item))
            continue
        threshold = config["class_thresholds"].get(item["label"], config["min_confidence"])
        if score < threshold:
            rejected.append(dict(index=index, reason="confidence", threshold=threshold, detection=item))
            continue
        span = 1-2*roi
        margin = 0.025*span
        touches = sum((box[0] <= roi+margin, box[1] <= roi+margin,
                       box[2] >= 1-roi-margin, box[3] >= 1-roi-margin))
        # A broad surface filling/truncated by the crop has weak object evidence.
        # Precision tradeoff is explicit and configurable, not a class blacklist.
        if (config["large_surface_guard"] and item["label"] in LARGE_SURFACES
                and (box[2]-box[0])/span >= 0.90 and (box[3]-box[1])/span >= 0.35
                and touches >= 2 and score < config["large_surface_min_confidence"]):
            rejected.append(dict(index=index, reason="large_truncated_surface", detection=item))
            continue
        candidates.append((index, item))
    for index, item in sorted(candidates, key=lambda pair: -pair[1]["confidence"]):
        reason, winner = None, None
        for accepted in kept:
            iou, containment, ratio = overlap(item["bbox_norm"], accepted["bbox_norm"])
            same = item["label"] == accepted["label"]
            vehicle_pair = item["label"] in VEHICLES and accepted["label"] in VEHICLES
            bike_pair = item["label"] in TWO_WHEELERS and accepted["label"] in TWO_WHEELERS
            if same and iou > config["same_class_iou"]:
                reason = "same_class_overlap"
            elif not same and (vehicle_pair or bike_pair) and iou > config["vehicle_conflict_iou"]:
                reason = "vehicle_class_conflict"
            elif same and vehicle_pair and containment >= config["vehicle_containment"] and ratio >= config["vehicle_min_area_ratio"]:
                reason = "vehicle_containment"
            if reason:
                winner = accepted
                break
        if reason is None and len(kept) >= config["max_detections"]:
            reason = "max_detections"
        if reason:
            rejected.append(dict(index=index, reason=reason, detection=item, kept=copy.deepcopy(winner)))
        else:
            kept.append(item)
    return kept, dict(version=config["version"], score_formula="objectness * class_score",
                      policy_sha256=policy_fingerprint(config), raw_count=len(detections),
                      kept_count=len(kept), rejected=rejected)


def decode_heads(heads, labels, roi=0.15, candidate=0.2, confidence=0.25, iou=0.45,
                 max_det=300, score_mode="objectness_class"):
    """Pinned YOLOv6 R1 heads -> class NMS -> full-frame normalised xyxy.

    class_only is an explicit legacy comparison mode, never the new default.
    """
    if len(heads) != 3 or not 0 <= roi < 0.45 or score_mode not in ("objectness_class", "class_only"):
        raise ValueError("Invalid YOLOv6 decoder configuration")
    if not all(0 <= value <= 1 for value in (candidate, confidence, iou)):
        raise ValueError("Invalid confidence/NMS threshold")
    boxes, scores, classes = [], [], []
    for raw, stride in zip(heads, (8, 16, 32)):
        h, w = 288//stride, 512//stride
        array = np.asarray(raw, dtype=np.float32)
        if array.size != (len(labels)+5)*h*w:
            raise ValueError("Unexpected YOLOv6 head shape")
        out = array.reshape(len(labels)+5, h, w).transpose(1, 2, 0)
        yy, xx = np.nonzero(out[..., 4] > candidate)
        p = out[yy, xx]
        if not len(p):
            continue
        cls = np.argmax(p[:, 5:], axis=1)
        score = p[np.arange(len(p)), cls+5]
        if score_mode == "objectness_class":
            score = score*p[:, 4]
        good = np.isfinite(score) & (score >= confidence)
        p, xx, yy, cls, score = p[good], xx[good], yy[good], cls[good], score[good]
        center = (p[:, :2]+np.stack([xx, yy], axis=1))*stride
        size = np.exp(np.clip(p[:, 2:4], -20, 20))*stride
        boxes.extend(np.concatenate([center-size/2, center+size/2], axis=1))
        scores.extend(score)
        classes.extend(cls)
    if not boxes:
        return []
    boxes, scores, classes = np.array(boxes), np.array(scores), np.array(classes)
    valid = np.isfinite(boxes).all(axis=1)
    boxes, scores, classes = boxes[valid], scores[valid], classes[valid]
    order = np.argsort(-scores, kind="stable")[:3000]
    area = np.maximum(boxes[:, 2:]-boxes[:, :2], 0).prod(axis=1)
    result = []
    while len(order) and len(result) < max_det:
        index = order[0]
        norm = np.clip(roi+(1-2*roi)*(boxes[index]/[512, 288, 512, 288]), 0, 1)
        result.append(dict(label=labels[int(classes[index])], label_id=int(classes[index]),
                           confidence=float(scores[index]), bbox_norm=norm.tolist()))
        rest = order[1:]
        lo, hi = np.maximum(boxes[index, :2], boxes[rest, :2]), np.minimum(boxes[index, 2:], boxes[rest, 2:])
        intersection = np.maximum(hi-lo, 0).prod(axis=1)
        overlap_iou = intersection/np.maximum(area[index]+area[rest]-intersection, 1e-12)
        order = rest[(classes[rest] != classes[index]) | (overlap_iou <= iou)]
    return result
