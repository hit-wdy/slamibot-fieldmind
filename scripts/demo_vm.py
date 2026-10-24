"""Windows control of the walk02 demo through VMware Tools; never logs credentials."""

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import time
import uuid


ROOT = Path(__file__).resolve().parents[1]
ACTIVE = ROOT / "data/demo_walk02/active_demo.json"
VMRUN = r"C:\Program Files (x86)\VMware\VMware Workstation\vmrun.exe"
VMX = r"E:\vm20\Ubuntu20.vmx"
REMOTE = "/home/wdy/radio_demo"


def vmrun(operation, *arguments):
    password = os.environ.get("SLAMIBOT_VM_PASSWORD")
    if not password:
        raise RuntimeError("Set SLAMIBOT_VM_PASSWORD from the authorized session; do not store it in a file")
    try:
        result = subprocess.run([VMRUN, "-gu", "wdy", "-gp", password, operation, VMX] + list(arguments),
                                capture_output=True, timeout=30)
    except subprocess.TimeoutExpired:
        raise RuntimeError("VMware operation timed out") from None
    if result.returncode:
        # VMware stdout/stderr does not include the argument array.
        output = (result.stdout + result.stderr).decode("utf-8", errors="replace").replace(password, "[redacted]")
        raise RuntimeError(operation + " failed: " + output)


def put(local, remote):
    vmrun("copyFileFromHostToGuest", str(Path(local).resolve()), remote)


def get(remote, local):
    vmrun("copyFileFromGuestToHost", remote, str(Path(local).resolve()))


def save_active(status):
    ACTIVE.parent.mkdir(parents=True, exist_ok=True)
    temporary = ACTIVE.with_suffix(".tmp")
    temporary.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(ACTIVE)


def remote_command(action, manifest):
    token = uuid.uuid4().hex
    remote = "/tmp/demo-reply-" + token + ".json"
    local = ROOT / "data/demo_walk02" / ("reply-" + token + ".json")
    local.parent.mkdir(parents=True, exist_ok=True)
    vmrun("runProgramInGuest", "/usr/bin/python3", REMOTE + "/demo_replay.py", action,
          "--url", manifest.get("control_url", "http://127.0.0.1:8767"), "--output", remote)
    get(remote, local)
    result = json.loads(local.read_text("utf-8"))
    if "error" in result:
        raise RuntimeError(result["error"])
    if result.get("session", {}).get("session_id") != manifest.get("session", {}).get("session_id"):
        raise RuntimeError("Demo ownership mismatch; refusing to use another session's controller")
    save_active(result)
    return result


def deploy():
    # mkdir via a Python argument avoids nested shell quoting through VIX.
    vmrun("runProgramInGuest", "/usr/bin/python3", "-c",
          "from pathlib import Path; Path('/home/wdy/radio_demo').mkdir(exist_ok=True)")
    for name in ("demo_replay.py", "demo_support.py"):
        put(ROOT / "scripts" / name, REMOTE + "/" + name)
    put(ROOT / "config/demo_walk02.rviz", REMOTE + "/demo_walk02.rviz")


def start(args):
    if ACTIVE.exists():
        previous = json.loads(ACTIVE.read_text("utf-8"))
        try:
            current = remote_command("status", previous)
        except RuntimeError:
            current = None
        if current and current["state"] not in ("complete", "stopped", "failed"):
            raise RuntimeError("A demo is already active; use status/start-navigation/stop")
        if current and current["state"] == "complete":
            remote_command("stop", current)
            time.sleep(2)
    deploy()
    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
    out = "/home/wdy/radio_replay_runs/demo_" + run_id
    local = ROOT / "data/demo_walk02" / run_id
    local.mkdir(parents=True, exist_ok=False)
    argv = ["serve", "--out", out, "--task-id", args.task_id,
            "--start-offset", str(args.start_offset),
            "--speed", str(args.speed)]
    if args.limit_seconds is not None:
        argv.extend(["--limit-seconds", str(args.limit_seconds)])
    # Only static shell setup is a command string; dynamic arguments use bash "$@".
    launcher = local / "launch.py"
    launcher.write_text(
        "import os, subprocess, json\nfrom pathlib import Path\n"
        "env=dict(os.environ);env['PATH']='/usr/bin:/bin:/usr/sbin:/sbin'\n"
        "env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None)\n"
        "command='source /opt/ros/noetic/setup.bash; source /home/wdy/radio_replay_ws/devel_ros/setup.bash; "
        "exec /usr/bin/python3 /home/wdy/radio_demo/demo_replay.py \"$@\"'\n"
        "with open(" + repr(out + ".launcher.log") + ",'w') as log:\n"
        " result=subprocess.run(['/bin/bash','--noprofile','--norc','-c',command,'demo-launcher']+"
        + repr(argv) + ",env=env,stdout=log,stderr=subprocess.STDOUT)\n"
        "Path(" + repr(out + ".exit.json") + ").write_text(json.dumps({'exit':result.returncode}))\n",
        encoding="utf-8")
    remote_launcher = REMOTE + "/launch_" + run_id + ".py"
    put(launcher, remote_launcher)
    vmrun("runProgramInGuest", "-noWait", "/usr/bin/python3", remote_launcher)
    end = time.monotonic() + 90
    while time.monotonic() < end:
        try:
            get(out + "/status.json", local / "status.json")
            status = json.loads((local / "status.json").read_text("utf-8"))
        except (RuntimeError, ValueError):
            time.sleep(1)
            continue
        if status["state"] == "failed":
            raise RuntimeError(status.get("error", "demo failed") + "; inspect " + out + ".launcher.log")
        if status["state"] == "paused":
            save_active(status)
            return status
        time.sleep(1)
    raise RuntimeError("Demo did not arm within 90 seconds; inspect " + out + ".launcher.log")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["deploy", "start", "status", "start-navigation", "pause", "resume", "stop"])
    parser.add_argument("--limit-seconds", type=float, help="Rehearsal only")
    parser.add_argument("--speed", type=float, default=1.)
    parser.add_argument("--start-offset", type=float, default=21.7)
    parser.add_argument("--task-id", default="walk02_demo")
    args = parser.parse_args()
    try:
        if args.action == "deploy":
            deploy()
            result = dict(status="deployed", remote=REMOTE)
        elif args.action == "start":
            result = start(args)
        else:
            if not ACTIVE.exists():
                raise RuntimeError("No active demo; run start first")
            manifest = json.loads(ACTIVE.read_text("utf-8"))
            # Check identity before any mutation, not only after it.
            current = remote_command("status", manifest)
            result = current if args.action == "status" else remote_command(args.action, current)
        print(json.dumps(result, ensure_ascii=True, indent=2))
    except RuntimeError as error:
        parser.exit(1, str(error) + "\n")


if __name__ == "__main__":
    main()
