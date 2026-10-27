"""Camera input adapter (spec section 4).

Outputs the one shape the rest of the pipeline expects: an RGB uint8 array plus
`source_sequence`, `captured_at` and `camera_id`. BGR conversion happens here, at
the adapter, not downstream.

The device-open and USB-re-enumeration handling follows `oak4p-yolov6/main.py`,
which is the working reference for this hardware. Backlog handling, sampling and
the queue are section 5 and are deliberately not here yet.
"""

from __future__ import annotations

import datetime as dt
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np

DEFAULT_RESOLUTION = (1024, 576)
# CAM_A on an OAK-4P-New is a wide, strongly distorted lens; the full field of
# view is kept because retrieval has to see the whole scene (spec section 13).
DEFAULT_SOCKET = "CAM_A"


@dataclass(frozen=True)
class Frame:
    """One captured frame, in the shape the encoder and storage layers want."""

    image: np.ndarray  # RGB uint8, HxWx3
    captured_at: dt.datetime  # timezone-aware, UTC
    timestamp_source: str
    source_sequence: int
    camera_id: str
    device_timestamp_ns: int | None = None
    exposure_time_us: int | None = None
    iso: int | None = None
    detections: tuple[Any, ...] = ()
    detections_available: bool = False

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def brightness(self) -> float:
        """Mean luma. Used to check exposure rather than to make decisions."""
        return float(self.image.mean())

    @property
    def saturated_fraction(self) -> float:
        """Fraction of pixels with every channel at or above 250.

        A healthy frame is well under 1%; anything near 1.0 means the sensor is
        blown out and the embedding describes nothing.
        """
        return float((self.image >= 250).all(axis=-1).mean())


def depthai():
    """Import DepthAI on demand.

    The device deployment path (ROS bridge -> HTTP ingest) never opens the OAK:
    the SLAMIBOT driver owns the camera and the OAK runtime there is DepthAI
    2.30, which cannot share this package's environment. Keeping the import lazy
    lets the backend, MCP proxy and test suite run on a host without DepthAI.
    """
    try:
        import depthai as dai
    except ImportError as error:
        raise ImportError(
            "depthai is required only for the standalone OAK capture path; install "
            "the 'oak' extra in a DepthAI 2.30-compatible environment, or feed the "
            "backend from the ROS bridge instead of opening the camera here"
        ) from error
    return dai


def list_devices() -> list[str]:
    dai = depthai()
    return [info.getDeviceId() for info in dai.Device.getAllAvailableDevices()]


def _discover_one_device(selector: str | None) -> Any:
    dai = depthai()
    devices = dai.Device.getAllAvailableDevices()
    if selector:
        matches = [d for d in devices if d.getDeviceId() == selector]
        if not matches:
            raise RuntimeError(f"DepthAI device {selector!r} was not found")
        return matches[0]
    if not devices:
        raise RuntimeError("No DepthAI device found")
    if len(devices) != 1:
        ids = ", ".join(d.getDeviceId() for d in devices)
        raise RuntimeError(f"Multiple devices found ({ids}); pass --device")
    return devices[0]


def wait_for_device(selector: str | None, timeout: float) -> Any:
    """Wait out the USB re-enumeration that happens while the device boots."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            return _discover_one_device(selector)
        except RuntimeError as error:
            missing = str(error) == "No DepthAI device found" or (
                selector is not None and "was not found" in str(error)
            )
            if not missing:
                raise
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"the OAK device did not appear within {timeout:.0f}s. Check the USB cable "
                    "and that nothing else is holding the device open, then retry."
                ) from error
            time.sleep(0.5)


def open_device(selector: str | None, usb_speed: str, timeout: float = 30.0, attempts: int = 3) -> Any:
    """Boot the device, retrying the transient failures seen on first enumeration."""
    dai = depthai()
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        info = wait_for_device(selector, timeout=timeout)
        try:
            if usb_speed == "AUTO":
                return dai.Device(info)
            return dai.Device(info, getattr(dai.UsbSpeed, usb_speed))
        except RuntimeError as error:
            last_error = error
            if "X_LINK_DEVICE_NOT_FOUND" not in str(error) or attempt == attempts:
                raise
            time.sleep(1.0)
    raise RuntimeError("could not open the DepthAI device") from last_error


def _duration_ns(value: Any) -> int | None:
    if isinstance(value, dt.timedelta):
        return int(value.total_seconds() * 1e9)
    return None


def _int_or_none(message: Any, accessor: str) -> int | None:
    method = getattr(message, accessor, None)
    if method is None:
        return None
    try:
        return int(method())
    except Exception:
        return None


class CameraSource:
    """A single colour camera, opened for the lifetime of the context manager."""

    def __init__(
        self,
        camera_id: str = "camera_01",
        socket: str = DEFAULT_SOCKET,
        resolution: tuple[int, int] = DEFAULT_RESOLUTION,
        fps: float | None = None,
        device_id: str | None = None,
        usb_speed: str = "AUTO",
        queue_size: int = 4,
        detector: Any = None,
        detection_wait: float = 0.25,
        exposure_us: int | None = None,
        iso: int | None = None,
        auto_exposure_limit_us: int | None = None,
        auto_exposure_compensation: int | None = None,
        # Auto-exposure on this camera overshoots badly and then converges slowly;
        # see `_settle`. Waiting for the brightness to stop moving is the only
        # control that survives a change of frame rate.
        settle: bool = True,
        settle_window: int = 15,
        settle_tolerance: float = 2.0,
        settle_timeout: float = 15.0,
    ) -> None:
        self.camera_id = camera_id
        self.socket = socket
        self.resolution = tuple(resolution)
        self.fps = fps
        self.device_id = device_id
        self.usb_speed = usb_speed
        self.queue_size = queue_size
        # None means "leave it to the camera". Setting either pins a manual
        # exposure, which is how you tell a broken auto-exposure loop apart from a
        # scene that is genuinely brighter than the sensor can handle.
        self.exposure_us = exposure_us
        self.iso = iso
        self.auto_exposure_limit_us = auto_exposure_limit_us
        self.auto_exposure_compensation = auto_exposure_compensation
        self.settle = settle
        self.settle_window = settle_window
        self.settle_tolerance = settle_tolerance
        self.settle_timeout = settle_timeout
        self.settle_info: dict[str, Any] = {}
        # An optional on-device detector. It shares this pipeline because a camera
        # output is the only thing that can feed it; running it on the VPU leaves
        # the host GPU entirely to the embedding model.
        self.detector = detector
        self.detection_wait = detection_wait
        self.detector_info: dict[str, Any] = {}

        self._device: Any = None
        self._pipeline: Any = None
        self._queue: Any = None
        self._detection_queue: Any = None
        self._pending_detections: dict[int, tuple[Any, Any]] = {}

    def open(self) -> "CameraSource":
        if self._pipeline is not None:
            return self
        dai = depthai()
        self._device = open_device(self.device_id, self.usb_speed)
        socket = getattr(dai.CameraBoardSocket, self.socket)
        available = {feature.socket for feature in self._device.getConnectedCameraFeatures()}
        if socket not in available:
            raise RuntimeError(
                f"{self.socket} is not connected; the device reports {sorted(s.name for s in available)}"
            )
        self.device_info = {
            "platform": self._device.getPlatformAsString(),
            "usb_speed": self._device.getUsbSpeed().name,
            "device_id": self._device.getDeviceId(),
            "socket": self.socket,
            "requested_resolution": list(self.resolution),
        }

        self._pipeline = dai.Pipeline(self._device)
        camera = self._pipeline.create(dai.node.Camera).build(socket)
        self._apply_exposure(camera)
        if self.fps is None:
            output = camera.requestOutput(self.resolution, dai.ImgFrame.Type.BGR888p)
        else:
            output = camera.requestOutput(self.resolution, dai.ImgFrame.Type.BGR888p, fps=self.fps)
        self._queue = output.createOutputQueue(maxSize=self.queue_size, blocking=False)
        if self.detector is not None:
            self._detection_queue = self.detector.attach(self._pipeline, output, dai)
            self.detector_info = self.detector.describe()
        self._pipeline.start()

        if self.settle:
            self.settle_info = self._settle()
        return self

    def flush(self, max_frames: int = 256) -> int:
        """Discard queued frames so the next grab returns a current one.

        Useful after a long gap (a model load, for instance) when the pipeline
        kept running but nobody was reading it.
        """
        if self._queue is None:
            return 0
        dropped = 0
        while dropped < max_frames and self._queue.tryGet() is not None:
            dropped += 1
        self._pending_detections.clear()
        return dropped

    def _drain_detections(self) -> None:
        """Move everything currently queued from the network into the pairing buffer."""
        if self._detection_queue is None:
            return
        while True:
            message = self._detection_queue.tryGet()
            if message is None:
                return
            sequence = int(message.getSequenceNum())
            self._pending_detections[sequence] = (message, time.monotonic())
            # The network keeps up with the camera, so a small buffer is enough.
            for stale in sorted(self._pending_detections)[:-8]:
                del self._pending_detections[stale]

    def _detections_for(self, sequence: int, timeout: float) -> tuple[Any, bool]:
        """The network result for one frame, waiting briefly for it to arrive.

        The two branches leave the device together but the network adds VPU
        latency, so a result for the frame in hand may still be in flight.
        """
        if self._detection_queue is None:
            return (), False
        deadline = time.monotonic() + timeout
        while True:
            if sequence in self._pending_detections:
                message, _ = self._pending_detections.pop(sequence)
                return self.detector.decode(message, dai), True
            self._drain_detections()
            if sequence in self._pending_detections:
                continue
            if time.monotonic() >= deadline:
                return (), False
            time.sleep(0.002)

    def _settle(self) -> dict[str, Any]:
        """Wait for auto-exposure to stop moving.

        Measured on this camera at 10 fps: the loop starts dark, overshoots to
        ~69% saturated pixels around frame 10, and only stabilises near frame 90
        (about 9 seconds). An over-exposed frame still yields a unit-norm
        embedding, so the failure is silent further down -- worth waiting for.

        A fixed warmup count would tie the wait to the frame rate, so the stop
        condition is the brightness spread over a trailing window instead. The
        absolute tolerance has a relative floor for scenes that genuinely
        flicker; when it never settles we proceed anyway and say so.
        """
        recent: deque[float] = deque(maxlen=self.settle_window)
        started = time.monotonic()
        seen = 0
        settled = False
        while True:
            frame = self.grab(timeout=5.0)
            seen += 1
            recent.append(frame.brightness)
            if len(recent) == self.settle_window:
                spread = max(recent) - min(recent)
                mean = sum(recent) / len(recent)
                if spread <= max(self.settle_tolerance, 0.02 * mean):
                    settled = True
                    break
            if time.monotonic() - started >= self.settle_timeout:
                break

        info: dict[str, Any] = {
            "settled": settled,
            "frames": seen,
            "seconds": round(time.monotonic() - started, 2),
            "brightness": round(recent[-1], 1),
            "saturated_fraction": round(frame.saturated_fraction, 4),
        }
        if not settled:
            info["reason"] = (
                f"brightness did not stabilise within {self.settle_timeout:.0f}s "
                f"(last {self.settle_window} frames spread {max(recent) - min(recent):.1f}); "
                "the scene may be flickering, or exposure is still drifting"
            )
        return info

    def _apply_exposure(self, camera: Any) -> None:
        control = camera.initialControl
        if self.auto_exposure_compensation is not None:
            control.setAutoExposureCompensation(self.auto_exposure_compensation)
        if self.auto_exposure_limit_us is not None:
            control.setAutoExposureLimit(self.auto_exposure_limit_us)
        if self.exposure_us is not None or self.iso is not None:
            # setManualExposure(exposure_us, iso); 0 means "auto" for each field.
            control.setManualExposure(self.exposure_us or 0, self.iso or 0)
        else:
            control.setAutoExposureEnable()

    def grab(self, timeout: float = 5.0) -> Frame:
        if self._queue is None:
            raise RuntimeError("CameraSource is not open; call open() or use it as a context manager")
        message = self._queue.get() if timeout is None else self._get_with_timeout(timeout)
        if message is None:
            raise TimeoutError(f"no frame from {self.camera_id} within {timeout:.1f}s")

        # DepthAI publishes BGR; the pipeline works in RGB (spec section 4).
        bgr = message.getCvFrame()
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])

        captured_at, source = _captured_at(message)
        sequence = int(message.getSequenceNum())
        detections: tuple[Any, ...] = ()
        available = False
        if self.detector is not None:
            detections, available = self._detections_for(sequence, self.detection_wait)
        return Frame(
            image=rgb,
            captured_at=captured_at,
            timestamp_source=source,
            source_sequence=sequence,
            camera_id=self.camera_id,
            device_timestamp_ns=_duration_ns(message.getTimestampDevice()),
            exposure_time_us=_int_or_none(message, "getExposureTime"),
            iso=_int_or_none(message, "getSensitivity"),
            detections=tuple(detections),
            detections_available=available,
        )

    def _get_with_timeout(self, timeout: float) -> Any:
        deadline = time.monotonic() + timeout
        while True:
            message = self._queue.tryGet()
            if message is not None:
                return message
            if time.monotonic() >= deadline:
                return None
            time.sleep(0.002)

    def frames(self, count: int | None = None, timeout: float = 5.0) -> Iterator[Frame]:
        produced = 0
        while count is None or produced < count:
            yield self.grab(timeout=timeout)
            produced += 1

    def close(self) -> None:
        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None
        self._queue = None
        self._detection_queue = None
        self._pending_detections.clear()
        if self._device is not None:
            self._device.close()
            self._device = None

    def __enter__(self) -> "CameraSource":
        return self.open()

    def __exit__(self, *_exc: object) -> None:
        self.close()


def _captured_at(message: Any) -> tuple[dt.datetime, str]:
    """Wall-clock capture time, preferring the camera's own if it qualifies.

    DepthAI's device clock is a monotonic duration since boot, not wall time, so
    it cannot be `captured_at`. `getTimestampSystem()` is the host clock stamped
    when the frame was received, which is the best wall-clock value available
    without correlating clocks; the device duration is kept separately.
    """
    system = getattr(message, "getTimestampSystem", None)
    if system is not None:
        try:
            value = system()
        except Exception:
            value = None
        if isinstance(value, dt.datetime):
            return (
                value if value.tzinfo else value.replace(tzinfo=dt.timezone.utc),
                "camera_system_clock",
            )
    return dt.datetime.now(dt.timezone.utc), "host_receive"
