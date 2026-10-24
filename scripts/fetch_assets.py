"""Download pinned model assets and verify their SHA-256 digests.

Assets are declared in models/assets.json. HTTP assets resume from partial files;
Hugging Face assets go through `hf_hub_download` into the project cache, which is
where open_clip later looks for the CLIP text tower. Either way the digest is
checked, so a failed or substituted transfer can never be mistaken for a usable
checkpoint.

Usage:
    python scripts/fetch_assets.py                 # fetch everything missing
    python scripts/fetch_assets.py --list          # show what is declared
    python scripts/fetch_assets.py --check         # verify local files only
    python scripts/fetch_assets.py --asset radio_v2.5_b
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "models" / "assets.json"
CHUNK = 1 << 20


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            hasher.update(block)
    return hasher.hexdigest()


def resolved_path(asset: dict) -> Path | None:
    """Where the asset is expected to live, or None if it is not present yet."""
    kind = asset.get("kind", "http")
    if kind == "local_copy":
        path = ROOT / asset["path"]
        return path if path.is_file() else None
    if kind == "http":
        return ROOT / asset["path"]

    from huggingface_hub import hf_hub_download

    os.environ["HF_HUB_CACHE"] = str(ROOT / asset["path"])
    import huggingface_hub.constants as constants

    constants.HF_HUB_CACHE = str(ROOT / asset["path"])
    try:
        cached = hf_hub_download(
            repo_id=asset["repo_id"], filename=asset["filename"], local_files_only=True
        )
    except Exception:
        return None
    path = Path(cached)
    # A snapshot still downloading is a partial blob; treat it as absent.
    return path if path.is_file() else None


def verify(asset: dict, path: Path) -> tuple[bool, str]:
    if not path.is_file():
        return False, "missing"
    size = path.stat().st_size
    if size != asset["size"]:
        return False, f"size {size} != {asset['size']}"
    actual = digest(path)
    if actual != asset["sha256"]:
        return False, f"sha256 {actual} != {asset['sha256']}"
    return True, actual


def fetch_http(asset: dict, dest: Path, attempts: int = 6) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".part")
    expected = asset["size"]

    for attempt in range(1, attempts + 1):
        have = partial.stat().st_size if partial.exists() else 0
        if have > expected:
            partial.unlink()
            have = 0
        if have == expected:
            break

        headers = {"User-Agent": "slamibot-fieldmind-fetch/1.0"}
        if have:
            headers["Range"] = f"bytes={have}-"
        request = urllib.request.Request(asset["url"], headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                if have and response.status != 206:
                    have = 0
                    partial.unlink(missing_ok=True)
                done = have
                started = time.monotonic()
                with partial.open("ab" if have else "wb") as handle:
                    while True:
                        block = response.read(CHUNK)
                        if not block:
                            break
                        handle.write(block)
                        done += len(block)
                        elapsed = max(time.monotonic() - started, 1e-6)
                        sys.stdout.write(
                            f"\r  {asset['name']}: {done / 1e6:8.1f}/{expected / 1e6:.1f} MB "
                            f"({100.0 * done / expected:5.1f}%) {(done - have) / elapsed / 1e6:5.2f} MB/s"
                        )
                        sys.stdout.flush()
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as error:
            sys.stdout.write(f"\n  attempt {attempt}/{attempts} failed: {error}\n")
            time.sleep(min(5 * attempt, 20))
            continue

        sys.stdout.write("\n")
        if partial.stat().st_size == expected:
            break

    if not partial.is_file() or partial.stat().st_size != expected:
        have = partial.stat().st_size if partial.is_file() else 0
        raise RuntimeError(f"{asset['name']}: incomplete download ({have}/{expected} bytes)")
    partial.replace(dest)
    return dest


def fetch_hf_hub(asset: dict) -> Path:
    from huggingface_hub import hf_hub_download

    cache_dir = ROOT / asset["path"]
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HUB_CACHE"] = str(cache_dir)
    import huggingface_hub.constants as constants

    constants.HF_HUB_CACHE = str(cache_dir)
    print(f"  {asset['name']}: downloading {asset['repo_id']}/{asset['filename']} ...")
    return Path(hf_hub_download(repo_id=asset["repo_id"], filename=asset["filename"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--asset", action="append", help="only handle this asset name")
    parser.add_argument("--list", action="store_true", help="list declared assets and exit")
    parser.add_argument("--check", action="store_true", help="verify local files without downloading")
    args = parser.parse_args()

    manifest = load_manifest()
    assets = manifest["assets"]
    if args.asset:
        wanted = set(args.asset)
        assets = [a for a in assets if a["name"] in wanted]
        missing = wanted - {a["name"] for a in assets}
        if missing:
            raise SystemExit(f"unknown asset(s): {', '.join(sorted(missing))}")

    if args.list:
        for asset in manifest["assets"]:
            print(f"{asset['name']:28} {asset['size'] / 1e9:7.3f} GB  {asset.get('note', '')}")
        return 0

    failures: list[str] = []
    for asset in assets:
        path = resolved_path(asset)
        if path is not None:
            ok, detail = verify(asset, path)
            if ok:
                print(f"  {asset['name']}: present, sha256 verified")
                continue
            if args.check:
                print(f"  {asset['name']}: {detail}")
                failures.append(asset["name"])
                continue
            print(f"  {asset['name']}: cached copy rejected ({detail}), refetching")

        if args.check:
            print(f"  {asset['name']}: missing")
            failures.append(asset["name"])
            continue

        kind = asset.get("kind", "http")
        if kind == "local_copy":
            raise SystemExit(
                f"{asset['name']}: not found at {ROOT / asset['path']}\n"
                f"  copy it from {asset.get('source', 'its source')}"
            )
        path = fetch_http(asset, ROOT / asset["path"]) if kind == "http" else fetch_hf_hub(asset)
        ok, detail = verify(asset, path)
        if not ok:
            raise SystemExit(f"{asset['name']}: {detail}\n  at {path}")
        print(f"  {asset['name']}: verified sha256 {detail[:16]}...  size {asset['size'] / 1e9:.3f} GB")

    if failures:
        print(f"\nNot verified: {', '.join(failures)}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
