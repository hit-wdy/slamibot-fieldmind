"""Read-only Jetson inventory; never installs packages or upgrades the host."""

import argparse
import datetime
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path


def read(path):
    try:
        return Path(path).read_text().replace("\x00", "").strip()
    except OSError:
        return None


def command(args):
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=10)
        return p.stdout.strip() if p.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--torch",
        action="store_true",
        help="also run a CUDA operation in the selected inference environment",
    )
    args = parser.parse_args()
    report = dict(
        observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        platform=platform.platform(),
        machine=platform.machine(),
        python=platform.python_version(),
        model=read("/proc/device-tree/model"),
        l4t=read("/etc/nv_tegra_release"),
        os_release=read("/etc/os-release"),
        memory=read("/proc/meminfo"),
        jetpack=command(["dpkg-query", "-W", "nvidia-jetpack"]),
        l4t_core=command(["dpkg-query", "-W", "nvidia-l4t-core"]),
        nvidia_runtime=command(["docker", "info", "--format", "{{json .Runtimes}}"]),
        usb_mounted=os.path.ismount("/mnt/usb"),
        disk_free_bytes=shutil.disk_usage(Path.cwd()).free,
        real_hardware_validated=False,
        compatibility_matrix="https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform-release-notes/pytorch-jetson-rel.html",
    )
    if args.torch:
        try:
            import torch

            report["torch"] = dict(
                version=torch.__version__,
                cuda=torch.version.cuda,
                available=torch.cuda.is_available(),
            )
            if torch.cuda.is_available():
                report["torch"]["device"] = torch.cuda.get_device_name(0)
                report["torch"]["cuda_smoke"] = (
                    float(
                        (
                            torch.ones(4, device="cuda") @ torch.ones(4, device="cuda")
                        ).cpu()
                    )
                    == 4
                )
            report["packages"] = {
                d.metadata["Name"]: d.version
                for d in importlib.metadata.distributions()
                if d.metadata["Name"]
            }
        except Exception as error:
            report["torch_error"] = str(error)
    output = json.dumps(report, indent=2, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
