"""Extract the pinned YOLOv6 blob/config for the framework's DepthAI v2 driver."""

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((ROOT / "models/assets.json").read_text())
    asset = next(a for a in manifest["assets"] if a["name"] == "yolov6_rvc2")
    archive = ROOT / asset["path"]
    if hashlib.sha256(archive.read_bytes()).hexdigest() != asset["sha256"]:
        raise RuntimeError("YOLO archive digest differs from pinned manifest")
    args.out.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as handle:
        config = json.load(handle.extractfile("config.json"))
        blob = config["model"]["metadata"]["path"]
        # Only copy the two known members; do not unpack arbitrary archive paths.
        (args.out / "model.blob").write_bytes(handle.extractfile(blob).read())
        (args.out / "config.json").write_text(
            json.dumps(config, indent=2), encoding="utf-8"
        )
    print(
        json.dumps(
            dict(
                output=str(args.out.resolve()),
                blob_sha256=hashlib.sha256(
                    (args.out / "model.blob").read_bytes()
                ).hexdigest(),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
