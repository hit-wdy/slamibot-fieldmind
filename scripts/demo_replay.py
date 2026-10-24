#!/usr/bin/env python3
"""Interactive walk02 replay, run on Ubuntu Noetic. No inference model in this process."""

import argparse
import copy
import datetime as dt
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from demo_support import ReplayControl, navigation_points


def http_json(url, value=None, timeout=10, method=None):
    body = None if value is None else json.dumps(value).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def serve(args):
    args.out.mkdir(parents=True, exist_ok=False)
    control, children, logs = ReplayControl(args.speed), [], []
    token = uuid.uuid4().hex
    event = dict(session_id="demo-" + token, coordinate_id="sim-" + token,
                 task_id=args.task_id, state="running", pose_source="simulated",
                 pose_selection_reason="walk02 demo: RTK has no fix; synthetic trajectory")
    data = dict(mode="bag_demo_simulated_navigation", session=event.copy(),
                bag=str(args.bag), image_topic="/SLB_CAM_A/compressed", camera_id="SLB_CAM_A",
                backend_url=args.backend, gateway_url=args.gateway, ros_port=args.ros_port,
                control_url="http://127.0.0.1:%d" % args.control_port,
                image_display_topic="/demo/camera/image_raw", published_images=0,
                navigation_points=[], reached_points=[], history_complete=False,
                timestamp_note="Original image header retained; current-view MCP reports stale historical source.")
    lock = threading.RLock()

    def snapshot():
        with lock:
            return dict(data, **control.snapshot())

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *unused):
            pass

        def handle_request(self, mutate):
            try:
                if self.path == "/status" and not mutate:
                    value, code = snapshot(), 200
                elif mutate and self.path in ("/start-navigation", "/pause", "/resume", "/stop"):
                    control.command(self.path[1:])
                    value, code = snapshot(), 200
                else:
                    value, code = dict(error="unknown endpoint"), 404
            except ValueError as error:
                value, code = dict(error=str(error)), 409
            body = json.dumps(value, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self.handle_request(False)

        def do_POST(self):
            self.handle_request(True)

    server = None
    session_pub = None

    def spawn(name, argv):
        log = (args.out / (name + ".log")).open("w")
        logs.append(log)
        child = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        children.append(child)
        return child

    def until(condition, timeout, message):
        end = time.monotonic() + timeout
        while not condition():
            if control.snapshot()["state"] == "stopped":
                raise RuntimeError("demo stopped during initialization/drain")
            if time.monotonic() >= end:
                raise RuntimeError(message)
            time.sleep(.1)

    def stop_history():
        event["state"] = "stopped"
        if session_pub:
            session_pub.publish(json.dumps(event))
        # Explicitly notify the backend as well; do not depend on shutdown callbacks.
        http_json(args.backend + "/v1/sessions", event)
        with lock:
            data["session"] = event.copy()

    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.control_port), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        atomic_json(args.out / "status.json", snapshot())
        if http_json(args.backend + "/v1/health")["model_state"] != "ready":
            raise RuntimeError("RADIO is not ready")
        http_json(args.gateway + "/metrics")
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", args.ros_port))
        os.environ.update(ROS_MASTER_URI="http://127.0.0.1:%d" % args.ros_port,
                          ROS_IP="127.0.0.1", ROS_HOME=str(args.out / ".ros"))
        import cv2
        import numpy as np
        import rosbag
        import rospy
        from nav_msgs.msg import Path as RosPath
        from geometry_msgs.msg import PoseStamped
        from rosgraph_msgs.msg import Clock
        from sensor_msgs.msg import CompressedImage, Image
        from std_msgs.msg import String
        from visualization_msgs.msg import Marker, MarkerArray

        with rosbag.Bag(str(args.bag)) as bag:
            info = bag.get_type_and_topic_info().topics.get(data["image_topic"])
            if not info or info.msg_type != "sensor_msgs/CompressedImage":
                raise ValueError("CAM_A compressed image topic is missing")
            _, original_first, original_record = next(bag.read_messages(topics=[data["image_topic"]]))
            if args.start_offset >= bag.get_end_time() - original_record.to_sec():
                raise ValueError("Demo start offset is beyond the bag")
            iterator = bag.read_messages(topics=[data["image_topic"]],
                        start_time=rospy.Time.from_sec(original_record.to_sec() + args.start_offset))
            _, first, first_record = next(iterator)
            first_stamp = first.header.stamp.to_sec()
            duration = bag.get_end_time() - first_record.to_sec()
            if args.limit_seconds is not None:
                duration = min(duration, args.limit_seconds)
            with lock:
                data.update(duration_s=duration, first_image_stamp=first_stamp,
                            clip_start_offset_s=first_record.to_sec() - original_record.to_sec(),
                            original_first_image_stamp=original_first.header.stamp.to_sec(),
                            query_start=dt.datetime.fromtimestamp(first_stamp - 1, dt.timezone.utc).isoformat(),
                            query_end=dt.datetime.fromtimestamp(first_stamp + duration + 2, dt.timezone.utc).isoformat(),
                            navigation_points=navigation_points(event["coordinate_id"], token, duration))
            master = spawn("roscore", ["/opt/ros/noetic/bin/roscore", "-p", str(args.ros_port)])

            def master_ready():
                if master.poll() is not None:
                    raise RuntimeError("roscore exited")
                try:
                    with socket.create_connection(("127.0.0.1", args.ros_port), timeout=.2):
                        return True
                except OSError:
                    return False
            until(master_ready, 20, "roscore did not start")
            rospy.set_param("/use_sim_time", True)
            rospy.init_node("walk02_interactive_demo", disable_signals=True)
            session_pub = rospy.Publisher("/perception/session", String, queue_size=1, latch=True)
            images = rospy.Publisher(data["image_topic"], CompressedImage, queue_size=2, latch=True)
            display = rospy.Publisher(data["image_display_topic"], Image, queue_size=1, latch=True)
            clock = rospy.Publisher("/clock", Clock, queue_size=1, latch=True)
            markers = rospy.Publisher("/demo/navigation_points", MarkerArray, queue_size=1, latch=True)
            path_pub = rospy.Publisher("/demo/path", RosPath, queue_size=1, latch=True)
            adapter = spawn("adapter", ["/opt/ros/noetic/bin/roslaunch", "semantic_perception", "rosbag_replay.launch",
                            "backend_url:=" + args.gateway, "start_time:=" + str(first_stamp),
                            "source_id:=" + event["session_id"], "image_topic:=" + data["image_topic"],
                            "camera_id:=SLB_CAM_A"])
            until(lambda: session_pub.get_num_connections() >= 1 and images.get_num_connections() >= 2,
                  25, "ROS adapters did not subscribe")
            session_pub.publish(json.dumps(event))
            http_json(args.backend + "/v1/sessions", event)
            time.sleep(.8)
            # Registration belongs to this coordinate session; future replays get new IDs.
            marker_array = MarkerArray()
            for i, p in enumerate(data["navigation_points"]):
                http_json(args.backend + "/v1/places", {k: v for k, v in p.items() if k != "offset_s"}, method="PUT")
                marker = Marker()
                marker.header.frame_id, marker.ns, marker.id = "simulated_map", "demo_navigation", i
                marker.type, marker.action = Marker.TEXT_VIEW_FACING, Marker.ADD
                marker.pose.position.x, marker.pose.position.z = p["x"], .7
                marker.pose.orientation.w = 1.
                marker.scale.z, marker.color.r, marker.color.g, marker.color.a = .6, .2, 1., 1.
                marker.text = "Position %d (sim)" % (i + 1)
                marker_array.markers.append(marker)
            markers.publish(marker_array)
            path = RosPath()
            path.header.frame_id = "simulated_map"

            def publish(image, offset):
                from demo_support import simulated_pose
                original_seq = image.header.seq
                decoded = cv2.imdecode(np.frombuffer(image.data, np.uint8), cv2.IMREAD_COLOR)
                if decoded is None:
                    raise ValueError("JPEG cannot be decoded")
                raw = Image(header=copy.deepcopy(image.header), height=decoded.shape[0], width=decoded.shape[1],
                            encoding="bgr8", is_bigendian=0, step=decoded.shape[1]*3, data=decoded.tobytes())
                clock.publish(Clock(clock=image.header.stamp))
                images.publish(image)
                display.publish(raw)
                pose = simulated_pose(image.header.stamp.to_sec() - first_stamp)
                pt = PoseStamped()
                pt.header.frame_id, pt.header.stamp = "simulated_map", image.header.stamp
                for k in ("x", "y", "z"):
                    setattr(pt.pose.position, k, pose[k])
                for k in ("x", "y", "z", "w"):
                    setattr(pt.pose.orientation, k, pose["q" + k])
                path.header.stamp = image.header.stamp
                path.poses.append(pt)
                path_pub.publish(path)
                with lock:
                    data["published_images"] += 1
                    data["last_image_stamp"] = image.header.stamp.to_sec()
                    data["last_original_header_seq"] = original_seq
                    data["source_elapsed_s"] = round(offset, 3)
                    data["reached_points"] = [p["place_id"] for p in data["navigation_points"] if offset >= p["offset_s"]]

            publish(first, 0.)

            def first_indexed():
                sessions = http_json(args.backend + "/v1/sessions")["sessions"]
                return any(s["session_id"] == event["session_id"] and s["frames"] > 0 for s in sessions)
            until(first_indexed, 45, "First frame was not accepted; inspect adapter/YOLO logs")
            control.ready()
            atomic_json(args.out / "status.json", snapshot())
            print("Demo armed: first frame indexed, playback paused", flush=True)
            for _, image, recorded in iterator:
                offset = recorded.to_sec() - first_record.to_sec()
                if offset > duration:
                    break
                while True:
                    state = control.snapshot()
                    atomic_json(args.out / "status.json", snapshot())
                    if state["state"] == "stopped" or rospy.is_shutdown():
                        break
                    if adapter.poll() is not None or master.poll() is not None:
                        raise RuntimeError("ROS process disconnected during demo")
                    if state["state"] == "playing" and state["playback_elapsed_s"] >= offset:
                        break
                    time.sleep(.04)
                if control.snapshot()["state"] == "stopped" or rospy.is_shutdown():
                    break
                publish(image, offset)
            if control.snapshot()["state"] != "stopped":
                control.drain()
                # Let the latest-frame slot finish before closing history.
                time.sleep(2)
                stop_history()
                until(lambda: not http_json(args.backend + "/v1/health")["states"].get("pending", 0),
                      120, "RADIO did not drain")
                with lock:
                    data["history_complete"] = True
                control.finish()
                print("Navigation replay complete; history remains queryable", flush=True)
                while control.snapshot()["state"] == "complete" and not rospy.is_shutdown():
                    atomic_json(args.out / "status.json", snapshot())
                    time.sleep(.3)
    except Exception as error:
        with lock:
            data["error"] = str(error)
        control.finish("failed")
        print("Demo failed: " + str(error), flush=True)
        raise
    finally:
        if session_pub:
            try:
                stop_history()
            except Exception as error:
                with lock:
                    data["shutdown_error"] = str(error)
        atomic_json(args.out / "status.json", snapshot())
        if server:
            server.shutdown()
            server.server_close()
        for child in reversed(children):
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGINT)
                try:
                    child.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGTERM)
        for log in logs:
            log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("serve")
    run.add_argument("--bag", type=Path, default=Path("/home/wdy/bag/walk02.bag"))
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--backend", default="http://192.168.1.4:8765")
    run.add_argument("--gateway", default="http://192.168.1.4:8766")
    run.add_argument("--ros-port", type=int, default=11322)
    run.add_argument("--control-port", type=int, default=8767)
    run.add_argument("--task-id", default="walk02_demo")
    run.add_argument("--speed", type=float, default=1.)
    run.add_argument("--start-offset", type=float, default=21.7,
                     help="Explicit demo opening in seconds from CAM_A start; 0 replays the complete bag")
    run.add_argument("--limit-seconds", type=float, help="Short rehearsal only; omit for the full recording")
    for action in ("status", "start-navigation", "pause", "resume", "stop"):
        cmd = commands.add_parser(action)
        cmd.add_argument("--url", default="http://127.0.0.1:8767")
        cmd.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "serve":
        if not __import__('math').isfinite(args.start_offset) or args.start_offset < 0:
            parser.error("--start-offset must be finite and non-negative")
        if args.limit_seconds is not None and (not __import__('math').isfinite(args.limit_seconds) or args.limit_seconds <= 0):
            parser.error("--limit-seconds must be positive and finite")
        serve(args)
    else:
        try:
            result = http_json(args.url + ("/status" if args.command == "status" else "/" + args.command),
                               None if args.command == "status" else {})
        except urllib.error.HTTPError as error:
            result = dict(error=error.read().decode(), http_status=error.code)
        except urllib.error.URLError as error:
            result = dict(error="Demo control endpoint unavailable: " + str(error.reason))
        if args.output:
            atomic_json(args.output, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if "error" in result:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
