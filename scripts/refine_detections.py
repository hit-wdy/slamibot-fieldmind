"""Re-run saved JPEGs with new YOLO policy; optionally update detections offline.

Default is a read-only comparison. --apply requires the backend root lock, makes
a SQLite backup, and atomically updates observation + legacy detection tables.
JPEGs, RADIO embeddings, timestamps and poses are never rewritten.
"""

import argparse
import collections
import copy
import datetime as dt
import hashlib
import json
from pathlib import Path
import sqlite3

from PIL import Image, ImageDraw

from slamibot_fieldmind.annotation import draw_detections, load_font
from slamibot_fieldmind.backend_lock import BackendLock
from replay_yolo_gateway import ReplayYolo


def vector_hashes(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (root / "vectors").rglob("*.f16")}


def comparison(source, old, new, output):
    with Image.open(source) as handle:
        image = handle.convert("RGB")
        image.thumbnail((1280, 900))
    left, right = image.copy(), image.copy()
    draw = ImageDraw.Draw(left)
    for item in old:
        box = [v*(left.width if i%2 == 0 else left.height) for i, v in enumerate(item["bbox_norm"])]
        draw.rectangle(box, outline="lime", width=2)
        draw.text((box[0], box[1]), "{} {:.2f}".format(item["label"], item["confidence"]), fill="lime")
    draw_detections(right, new)
    canvas = Image.new("RGB", (2*image.width, image.height+52), "#151922")
    canvas.paste(left, (0, 52))
    canvas.paste(right, (image.width, 52))
    draw = ImageDraw.Draw(canvas)
    font = load_font(25)
    draw.text((12, 10), "BEFORE: {} boxes / legacy class scores".format(len(old)), font=font, fill="white")
    draw.text((image.width+12, 10), "AFTER: {} boxes / objectness * class".format(len(new)), font=font, fill="white")
    canvas.save(output, quality=92)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--session-id", action="append", required=True)
    parser.add_argument("--example-frame", action="append", default=[])
    parser.add_argument("--postprocess-config", type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    args.root = args.root.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    owner = BackendLock(args.root) if args.apply else None
    db = None
    try:
        uri = (args.root / "records.sqlite").as_uri() + ("?mode=rw" if args.apply else "?mode=ro")
        db = sqlite3.connect(uri, uri=True, timeout=10)
        db.row_factory = sqlite3.Row
        if args.apply:
            if db.execute("SELECT COUNT(*) FROM observations WHERE radio_status='pending'").fetchone()[0]:
                raise RuntimeError("Drain pending RADIO work before offline refinement")
            with sqlite3.connect(args.out / "records.before.sqlite") as backup:
                db.backup(backup)
        placeholders = ",".join("?" for _ in args.session_id)
        rows = list(db.execute("SELECT * FROM observations WHERE session_id IN (" + placeholders +
                               ") AND image_path IS NOT NULL AND radio_status='ready' ORDER BY source_stamp_ns", args.session_id))
        if not rows:
            raise ValueError("No saved images in selected sessions")
        hashes = vector_hashes(args.root)
        config = json.loads(args.postprocess_config.read_text(encoding="utf8")) if args.postprocess_config else None
        detector = ReplayYolo(args.model, policy=config)
        records = []
        summary = dict(applied=args.apply, sessions=args.session_id, frames=len(rows),
                       before_counts=collections.Counter(), after_counts=collections.Counter(),
                       baseline_counts=collections.Counter(), baseline_total=0,
                       rejected_reasons=collections.Counter(), before_total=0, after_total=0,
                       model_fingerprint=detector.fingerprint, parameters=detector.parameters)
        with (args.out / "audit.jsonl").open("w", encoding="utf8") as audit:
            for index, row in enumerate(rows):
                original = json.loads(row["metadata_json"])
                source = (args.root / row["image_path"]).resolve()
                if not source.is_relative_to(args.root):
                    raise ValueError("Image path escapes root")
                jpeg = source.read_bytes()
                detections, elapsed, details = detector.detect_observation(jpeg)
                updated = copy.deepcopy(original)
                updated.setdefault("detections_original", copy.deepcopy(original.get("detections", [])))
                updated.setdefault("detector_fingerprint_original", original.get("detector_fingerprint"))
                updated.update(details)
                updated.update(detections=detections, detection_status="available", detector_fingerprint=detector.fingerprint,
                               detection_execution="host_cpu_openvino_replay", detection_inference_ms=elapsed,
                               detection_refinement=dict(version="precision_v1", image_sha256=hashlib.sha256(jpeg).hexdigest(),
                                                         at=dt.datetime.now(dt.timezone.utc).isoformat()))
                records.append((row, updated))
                old = original.get("detections", [])
                baseline = original.get("detections_original", old)
                summary["before_total"] += len(old)
                summary["baseline_total"] += len(baseline)
                summary["baseline_counts"].update(d["label"] for d in baseline)
                summary["after_total"] += len(detections)
                summary["before_counts"].update(d["label"] for d in old)
                summary["after_counts"].update(d["label"] for d in detections)
                summary["rejected_reasons"].update(d["reason"] for d in details["detection_postprocess"]["rejected"])
                audit.write(json.dumps(dict(frame_id=row["frame_id"], old=old, baseline=baseline, new=detections, **details), ensure_ascii=False)+"\n")
                if row["frame_id"] in args.example_frame:
                    comparison(source, baseline, detections, args.out / (row["frame_id"] + ".jpg"))
                if (index+1)%25 == 0:
                    print("Processed {}/{}".format(index+1, len(rows)), flush=True)
        if args.apply:
            # The root lock excludes all backend writers; one SQL transaction also
            # prevents legacy readers from seeing a partially updated class index.
            with db:
                for row, updated in records:
                    result = db.execute("UPDATE observations SET metadata_json=? WHERE frame_id=? AND metadata_json=?",
                                        (json.dumps(updated, allow_nan=False), row["frame_id"], row["metadata_json"]))
                    if result.rowcount != 1:
                        raise RuntimeError("Frame changed during refinement")
                    if not db.execute("SELECT 1 FROM frames WHERE frame_id=?", (row["frame_id"],)).fetchone():
                        raise RuntimeError("Ready observation has no vector record")
                    db.execute("UPDATE frames SET detection_count=? WHERE frame_id=?",
                               (len(updated["detections"]), row["frame_id"]))
                    db.execute("DELETE FROM detections WHERE frame_id=?", (row["frame_id"],))
                    db.executemany("INSERT INTO detections(frame_id,label,label_id,confidence,x1,y1,x2,y2) VALUES (?,?,?,?,?,?,?,?)",
                                   [(row["frame_id"], d["label"], d["label_id"], d["confidence"], *d["bbox_norm"]) for d in updated["detections"]])
        summary.update(flip_checks=detector.flip_checks, total_inference_ms=detector.total_ms,
                       vector_hashes=hashes, vector_files_unchanged=hashes == vector_hashes(args.root))
        if not summary["vector_files_unchanged"]:
            raise RuntimeError("RADIO vector files unexpectedly changed")
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf8")
        print(json.dumps({k: v for k, v in summary.items() if k not in ("vector_hashes", "parameters")}, ensure_ascii=False, indent=2))
    finally:
        if db is not None:
            db.close()
        if owner is not None:
            owner.close()


if __name__ == "__main__":
    main()
