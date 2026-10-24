"""Replay a record_video manifest through the real backend without owning a camera.

Uses the real RADIO model and recorded YOLO results. No SLAM pose is fabricated.
An optional --serve keeps the same loaded runtime available over HTTP/MCP.
"""

import argparse
import datetime as dt
import json
import time
import uuid
from pathlib import Path

from slamibot_fieldmind.backend import create_app
from slamibot_fieldmind.backend_query import BackendQuery
from slamibot_fieldmind.backend_runtime import BackendConfig, BackendRuntime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="dedicated replay output; do not use the original record directory",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = manifest["indexed"][: args.limit or None]
    if not records:
        raise ValueError("manifest contains no indexed frames")
    runtime = BackendRuntime(
        BackendConfig(root=args.root, require_mount=False, min_free_bytes=0)
    )
    query = BackendQuery(runtime)
    session_id = "replay-" + uuid.uuid4().hex
    session = dict(
        session_id=session_id,
        coordinate_id=session_id,
        task_id="offline-" + manifest["capture_id"],
        state="running",
        mode="offline_replay",
        pose_source="unavailable",
    )
    try:
        deadline = time.monotonic() + 180
        while runtime.model_state == "loading" and time.monotonic() < deadline:
            time.sleep(0.2)
        if runtime.model_state != "ready":
            raise RuntimeError(runtime.model_error or "model initialization timed out")
        runtime.session(session)
        accepted = []
        for row in records:
            path = args.manifest.parent / row["jpeg"].replace("\\", "/")
            stamp = row.get("device_timestamp_ns")
            if stamp is None:
                stamp = int(
                    dt.datetime.fromisoformat(row["captured_at"]).timestamp() * 1e9
                )
            m = dict(
                protocol_version=1,
                camera_id="CAM_A",
                device_id=manifest["camera"]["device_id"],
                source_id=session_id,
                source_sequence=row["source_sequence"],
                source_stamp_ns=stamp,
                device_timestamp_ns=row.get("device_timestamp_ns"),
                captured_at=row["captured_at"],
                received_at=row["captured_at"],
                timestamp_source="offline_recording",
                sync_quality="offline_recording",
                session_id=session_id,
                coordinate_id=session_id,
                body_pose=None,
                camera_pose=None,
                detection_status="available",
                detections=row["detections"],
                replay=True,
                original_frame_id=row["frame_id"],
            )
            result = runtime.ingest(m, path.read_bytes())
            if result["status"] == "accepted":
                accepted.append(result["frame_id"])
        runtime.session(dict(session, state="stopped"))
        deadline = time.monotonic() + 180
        while (
            runtime.health()["states"].get("pending", 0) and time.monotonic() < deadline
        ):
            time.sleep(0.1)
        start = (
            dt.datetime.fromisoformat(records[0]["captured_at"])
            - dt.timedelta(seconds=1)
        ).isoformat()
        end = (
            dt.datetime.fromisoformat(records[-1]["captured_at"])
            + dt.timedelta(seconds=1)
        ).isoformat()
        report = dict(
            mode="offline_replay_real_radio_recorded_yolo",
            real_slam=False,
            jetson_validated=False,
            input_frames=len(records),
            accepted_frame_ids=accepted,
            status=runtime.health(),
            person=query.search(
                "a person", start_time=start, end_time=end, session_id=session_id
            ),
            whale=query.search(
                "a large whale", start_time=start, end_time=end, session_id=session_id
            ),
            current_view=query.describe(),
            sessions=query.list_sessions(),
        )
        target = args.root / "replay_report.json"
        target.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if accepted:
            (args.root / "evidence.jpg").write_bytes(
                query.image(accepted[0], overlay=True)
            )
        print(
            json.dumps(
                dict(
                    report=str(target.resolve()),
                    input_frames=len(records),
                    keyframes=len(accepted),
                    states=report["status"]["states"],
                ),
                indent=2,
            )
        )
        if report["status"]["states"].get("pending", 0) or report["status"][
            "states"
        ].get("failed", 0):
            raise RuntimeError(
                "replay did not complete successfully; inspect replay_report.json"
            )
        if args.serve:
            import uvicorn

            uvicorn.run(create_app(runtime), host="127.0.0.1", port=args.port)
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
