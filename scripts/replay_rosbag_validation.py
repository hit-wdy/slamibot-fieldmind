#!/usr/bin/env python3
"""Run inside ROS Noetic after sourcing ROS and the semantic_perception workspace.

Uses an isolated ROS master and only image messages from a bag. GPU inference stays
in the existing HTTP backend. Does not require Livox message packages or ROS on Windows.
"""

import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import urllib.request
import uuid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--port", type=int, default=11321)
    parser.add_argument("--image-topic", default="/hikrobot_camera0/rgb/compressed")
    parser.add_argument("--camera-id", default="bag_cam_a")
    parser.add_argument("--task-id")
    parser.add_argument("--pose-note", default="Synthetic trajectory; not measured navigation data")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    os.environ["ROS_MASTER_URI"] = "http://127.0.0.1:%d" % args.port
    os.environ["ROS_HOME"] = str(args.out / ".ros")
    os.environ["ROS_IP"] = "127.0.0.1"
    children, logs = [], []
    report = dict(mode="rosbag_with_simulated_pose", bag=str(args.bag), counts={},
                  image_topic=args.image_topic, camera_id=args.camera_id)
    event = dict(session_id="bag-" + uuid.uuid4().hex, coordinate_id="sim-" + uuid.uuid4().hex,
                 task_id=args.task_id or args.bag.stem + "_bag_simulated", state="running",
                 pose_source="simulated", pose_selection_reason=args.pose_note)
    report["session"] = event

    def save():
        temporary = args.out / "report.json.tmp"
        temporary.write_text(json.dumps(report, indent=2))
        temporary.replace(args.out / "report.json")

    def spawn(name, command):
        log = (args.out / (name + ".log")).open("w")
        logs.append(log)
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        children.append((name, process))
        report.setdefault("processes", {})[name] = process.pid
        return process

    def health():
        with urllib.request.urlopen(args.url.rstrip("/") + "/v1/health", timeout=5) as response:
            return json.load(response)

    session_pub = None
    try:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", args.port))  # Fail before using any existing ROS master.
        import rosbag
        with rosbag.Bag(str(args.bag)) as bag:
            start, end = bag.get_start_time(), bag.get_end_time()
            report.update(bag_start=start, bag_end=end, duration=end-start)
            topic_info = bag.get_type_and_topic_info().topics.get(args.image_topic)
            if topic_info is None or topic_info.msg_type != "sensor_msgs/CompressedImage":
                raise ValueError("Selected topic must exist and contain sensor_msgs/CompressedImage")
            report["expected_image_messages"] = topic_info.message_count
            _, first_image, recorded_at = next(bag.read_messages(topics=[args.image_topic]))
            image_start = first_image.header.stamp.to_sec()
            report.update(first_image_stamp=image_start, first_record_stamp=recorded_at.to_sec(),
                          header_minus_record_s=image_start-recorded_at.to_sec())
        report["backend_before"] = health()
        if report["backend_before"]["model_state"] != "ready":
            raise RuntimeError("Wait for backend model_state=ready before playback")
        spawn("roscore", ["/opt/ros/noetic/bin/roscore", "-p", str(args.port)])
        deadline = time.monotonic() + 20
        while True:
            try:
                with socket.create_connection(("127.0.0.1", args.port), timeout=1):
                    break
            except OSError:
                if time.monotonic() > deadline:
                    raise RuntimeError("isolated ROS master did not start")
                time.sleep(0.2)
        import rospy
        from nav_msgs.msg import Odometry
        from sensor_msgs.msg import CompressedImage
        from std_msgs.msg import String
        from semantic_perception.msg import FrameObservation
        rospy.set_param("/use_sim_time", True)
        rospy.init_node("walk_replay_validation", disable_signals=True)
        session_pub = rospy.Publisher("/perception/session", String, queue_size=1, latch=True)

        def count(name):
            def receive(msg):
                counts = report["counts"]
                counts[name] = counts.get(name, 0) + 1
                if name == "observations":
                    report["last_observation"] = dict(stamp=msg.image.header.stamp.to_sec(),
                        sequence=msg.source_sequence, detection_status=msg.detection_status,
                        timestamp_source=msg.timestamp_source, sync_quality=msg.sync_quality)
            return receive

        subscribers = [
            rospy.Subscriber(args.image_topic, CompressedImage, count("images"), queue_size=10),
            rospy.Subscriber("/Odometry", Odometry, count("odometry"), queue_size=100),
            rospy.Subscriber("/perception/cam_a/observation", FrameObservation, count("observations"), queue_size=10),
        ]
        spawn("adapter", ["/opt/ros/noetic/bin/roslaunch", "semantic_perception", "rosbag_replay.launch",
                          "backend_url:=" + args.url, "start_time:=" + str(image_start), "source_id:=" + event["session_id"],
                          "image_topic:=" + args.image_topic, "camera_id:=" + args.camera_id])
        deadline = time.monotonic() + 25
        while session_pub.get_num_connections() < 1:
            if time.monotonic() > deadline:
                raise RuntimeError("bridge did not subscribe to session notifications")
            time.sleep(0.2)
        session_pub.publish(String(data=json.dumps(event)))
        time.sleep(3)
        player = spawn("rosbag", ["/usr/bin/python3", "/opt/ros/noetic/bin/rosbag", "play", str(args.bag),
                                  "--clock", "--hz=100", "--delay=3", "--topics", args.image_topic])
        report["state"] = "playing"
        wall_start = time.monotonic()
        while player.poll() is None:
            report["elapsed_wall_s"] = time.monotonic() - wall_start
            save()
            if report["elapsed_wall_s"] > end-start+90:
                raise RuntimeError("bag playback exceeded timeout")
            time.sleep(1)
        report["rosbag_exit"] = player.returncode
        if player.returncode:
            raise RuntimeError("rosbag playback failed")
        time.sleep(2)  # Drain the bridge's latest-frame slot before stopping history.
        event["state"] = "stopped"
        session_pub.publish(String(data=json.dumps(event)))
        time.sleep(1)
        deadline = time.monotonic() + 120
        while True:
            report["backend_after"] = health()
            if not report["backend_after"]["states"].get("pending", 0):
                break
            if time.monotonic() > deadline:
                raise RuntimeError("RADIO queue did not drain")
            time.sleep(1)
        if not report["counts"].get("observations") or not report["backend_after"]["indexed_frames"]:
            raise RuntimeError("no observations or indexed keyframes")
        report["state"] = "complete"
    except Exception as error:
        report.update(state="failed", error=str(error))
        raise
    finally:
        if session_pub is not None:
            event["state"] = "stopped"
            session_pub.publish(json.dumps(event))
            time.sleep(0.5)
        for name, process in reversed(children):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
                try:
                    process.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
        for log in logs:
            log.close()
        save()


if __name__ == "__main__":
    main()
