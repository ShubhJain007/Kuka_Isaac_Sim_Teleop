#!/usr/bin/env python3
"""Stream a ROS2 Image topic to Apple Vision Pro companion window.

Two modes (both served simultaneously):
  MJPEG  — ~100-200 ms latency, frame-by-frame JPEG over HTTP (best for real-time)
  HLS    — ~0.5-1 s  latency, H.264 segments via ffmpeg (best for compatibility)

Usage:
    python camera_hls_stream.py [--topic /camera_arm/camera/color/image_rect_raw]
                                [--port 8080] [--width 640] [--height 480]
                                [--fps 30] [--jpeg-quality 70]

Endpoints:
  http://<IP>:8080/mjpeg          ← MJPEG stream  (low latency, use this)
  http://<IP>:8080/stream.m3u8   ← HLS stream    (higher latency, AVPlayer compat)

Requires: ffmpeg, rclpy (Isaac Sim bundled or system ROS2)
"""
from __future__ import annotations

import argparse
import http.server
import io
import os
import signal
import subprocess
import sys
import threading
import time
import functools

import numpy as np

# ── ROS2 library bootstrap (same pattern as med7_ros_bones.setup_ros2_libs) ──
def _setup_ros2():
    try:
        import isaacsim
        root = os.path.join(os.path.dirname(isaacsim.__file__),
                            "exts/isaacsim.ros2.bridge/humble")
        lib = os.path.join(root, "lib")
        pyp = os.path.join(root, "rclpy")

        # Strip system ROS Python paths so Isaac Sim's bundled rclpy wins
        sys.path[:] = [p for p in sys.path
                       if "/opt/ros/" not in p
                       and "dist-packages/rclpy" not in p
                       and "site-packages/rclpy" not in p]

        # Prepend bundled rclpy (highest priority)
        if pyp not in sys.path:
            sys.path.insert(0, pyp)

        # Re-exec if LD_LIBRARY_PATH still missing the bundled libs
        cur = os.environ.get("LD_LIBRARY_PATH", "")
        if lib not in cur:
            os.environ["LD_LIBRARY_PATH"] = f"{lib}:{cur}" if cur else lib
            os.execv(sys.executable, [sys.executable] + sys.argv)
    except ImportError:
        pass


_setup_ros2()

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image


def parse_args():
    p = argparse.ArgumentParser(description="ROS2 camera → MJPEG + HLS stream for AVP")
    p.add_argument("--topic", default="/camera_arm/camera/color/image_rect_raw",
                   help="ROS2 Image topic to subscribe to")
    p.add_argument("--port", type=int, default=8080,
                   help="HTTP port for streams")
    p.add_argument("--width", type=int, default=640,
                   help="Output video width (input is scaled)")
    p.add_argument("--height", type=int, default=480,
                   help="Output video height (input is scaled)")
    p.add_argument("--fps", type=int, default=30,
                   help="Target framerate")
    p.add_argument("--jpeg-quality", type=int, default=70,
                   help="MJPEG quality 1-100 (lower = smaller/faster, 70 is good)")
    return p.parse_args()


# ── Shared MJPEG frame buffer ────────────────────────────────────────────

class MJPEGBuffer:
    """Thread-safe single-frame buffer for MJPEG streaming."""

    def __init__(self):
        self._jpeg: bytes = b""
        self._lock = threading.Lock()
        self._event = threading.Event()

    def update(self, jpeg_bytes: bytes):
        with self._lock:
            self._jpeg = jpeg_bytes
        self._event.set()

    def wait_and_get(self, timeout: float = 1.0) -> bytes:
        self._event.wait(timeout=timeout)
        self._event.clear()
        with self._lock:
            return self._jpeg


_mjpeg_buf = MJPEGBuffer()


class CameraStreamBridge(Node):
    """ROS2 node that feeds both MJPEG buffer and ffmpeg HLS."""

    def __init__(self, topic: str, width: int, height: int, fps: int,
                 hls_dir: str, jpeg_quality: int):
        super().__init__("camera_stream_bridge")
        self._width = width
        self._height = height
        self._fps = fps
        self._hls_dir = hls_dir
        self._jpeg_quality = jpeg_quality
        self._frame_count = 0
        self._ffmpeg: subprocess.Popen | None = None
        self._started = False
        self._min_interval = 1.0 / fps
        self._last_frame_time = 0.0

        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(Image, topic, self._on_image, qos)
        self.get_logger().info(
            f"Subscribed to {topic}  →  MJPEG (q={jpeg_quality}) + HLS @ {width}x{height} {fps}fps")

    def _ensure_ffmpeg(self, in_w: int, in_h: int):
        if self._started:
            return
        # back off: don't relaunch more than once per 2 s when ffmpeg keeps dying
        now = time.monotonic()
        if now - getattr(self, "_ffmpeg_last_start", 0.0) < 2.0:
            return
        self._ffmpeg_last_start = now
        self._started = True

        m3u8 = os.path.join(self._hls_dir, "stream.m3u8")
        cmd = [
            "ffmpeg", "-y",
            # Minimize input buffering
            "-fflags", "nobuffer",
            "-flags", "low_delay",
            "-f", "rawvideo",
            "-pixel_format", "rgb24",
            "-video_size", f"{in_w}x{in_h}",
            "-framerate", str(self._fps),
            "-i", "pipe:0",
            "-vf", f"scale={self._width}:{self._height},format=yuv420p",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-tune", "zerolatency",
            # Aggressive low-latency H.264 settings
            "-profile:v", "main",
            "-level", "3.1",
            "-b:v", "800k",
            "-maxrate", "1000k",
            "-bufsize", "500k",
            "-g", str(max(self._fps // 2, 5)),  # keyframe every 0.5 s
            "-sc_threshold", "0",
            "-flags", "+cgop",
            # HLS with short segments for low latency
            "-f", "hls",
            "-hls_time", "0.5",
            "-hls_list_size", "3",
            "-hls_flags", "delete_segments+append_list+omit_endlist",
            "-hls_segment_type", "mpegts",
            "-hls_segment_filename", os.path.join(self._hls_dir, "seg_%05d.ts"),
            m3u8,
        ]
        self.get_logger().info(f"Starting ffmpeg: {in_w}x{in_h} -> {self._width}x{self._height}")
        self._ffmpeg = subprocess.Popen(
            cmd, stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        self._ffmpeg_in_w = in_w
        self._ffmpeg_in_h = in_h

    def _on_image(self, msg: Image):
        now = time.monotonic()
        if (now - self._last_frame_time) < self._min_interval:
            return
        self._last_frame_time = now

        h, w = msg.height, msg.width
        enc = msg.encoding.lower()

        try:
            if enc in ("rgb8",):
                frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 3)
            elif enc in ("bgr8",):
                raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 3)
                frame = raw[:, :, ::-1].copy()
            elif enc in ("rgba8",):
                frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 4)[:, :, :3].copy()
            elif enc in ("bgra8",):
                raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 4)
                frame = raw[:, :, 2::-1].copy()
            elif enc in ("mono8",):
                gray = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w)
                frame = np.stack([gray, gray, gray], axis=-1)
            else:
                if self._frame_count < 3:
                    self.get_logger().warn(f"Unsupported encoding: {enc}")
                return
        except Exception as e:
            if self._frame_count < 5:
                self.get_logger().error(f"Frame decode error: {e}")
            return

        self._frame_count += 1

        # frame is rgb24 at ORIGINAL size (h, w). Keep it that way for ffmpeg.

        # ── HLS: pipe ORIGINAL-size raw frame to ffmpeg; ffmpeg handles scaling ──
        self._ensure_ffmpeg(w, h)
        if self._ffmpeg and self._ffmpeg.stdin:
            try:
                # ensure contiguous bytes (copy() handles views from slicing)
                self._ffmpeg.stdin.write(np.ascontiguousarray(frame).tobytes())
            except (BrokenPipeError, ValueError, OSError):
                self.get_logger().warn("ffmpeg pipe broken — restarting")
                try:
                    self._ffmpeg.kill()
                except Exception:
                    pass
                self._ffmpeg = None
                self._started = False

        # ── MJPEG: resize (if needed) then encode JPEG ──
        try:
            import cv2 as _cv2
            if w != self._width or h != self._height:
                mjpeg_frame = _cv2.resize(frame, (self._width, self._height),
                                          interpolation=_cv2.INTER_LINEAR)
            else:
                mjpeg_frame = frame
            bgr = _cv2.cvtColor(mjpeg_frame, _cv2.COLOR_RGB2BGR)
            _, jpeg = _cv2.imencode(".jpg", bgr,
                                    [_cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality])
            _mjpeg_buf.update(jpeg.tobytes())
        except ImportError:
            from PIL import Image as _PILImage
            pil_img = _PILImage.fromarray(frame)
            if (w, h) != (self._width, self._height):
                pil_img = pil_img.resize((self._width, self._height))
            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=self._jpeg_quality)
            _mjpeg_buf.update(buf.getvalue())

        if self._frame_count <= 2 or self._frame_count % 300 == 0:
            self.get_logger().info(f"Frame #{self._frame_count}: {w}x{h} {enc}")

    def shutdown(self):
        if self._ffmpeg:
            self.get_logger().info("Stopping ffmpeg...")
            try:
                self._ffmpeg.stdin.close()
            except Exception:
                pass
            self._ffmpeg.wait(timeout=5)


# ── HTTP server: serves both MJPEG and HLS ───────────────────────────────

class StreamRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Serves /mjpeg as multipart JPEG stream, everything else as HLS files."""

    def do_GET(self):
        if self.path == "/mjpeg":
            self._serve_mjpeg()
        else:
            super().do_GET()

    def _serve_mjpeg(self):
        self.send_response(200)
        self.send_header("Content-Type",
                         "multipart/x-mixed-replace; boundary=--frame")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        try:
            while True:
                jpeg = _mjpeg_buf.wait_and_get(timeout=2.0)
                if not jpeg:
                    continue
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpeg)}\r\n".encode())
                self.wfile.write(b"\r\n")
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def end_headers(self):
        if not self.path.startswith("/mjpeg"):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        super().end_headers()

    def log_message(self, format, *a):
        pass


def start_http_server(directory: str, port: int):
    handler = functools.partial(StreamRequestHandler, directory=directory)
    http.server.HTTPServer.allow_reuse_address = True
    srv = http.server.HTTPServer(("0.0.0.0", port), handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv


def get_local_ips() -> list[str]:
    """Return all non-loopback IPv4 addresses via `ip addr`."""
    import re
    import subprocess
    try:
        out = subprocess.check_output(["ip", "-4", "addr", "show"], text=True)
        ips = re.findall(r"inet (\d+\.\d+\.\d+\.\d+)/", out)
        return [ip for ip in ips if not ip.startswith("127.")] or ["127.0.0.1"]
    except Exception:
        return ["127.0.0.1"]


def main():
    args = parse_args()

    hls_dir = "/tmp/camera_hls_stream"
    os.makedirs(hls_dir, exist_ok=True)
    for f in os.listdir(hls_dir):
        os.remove(os.path.join(hls_dir, f))

    http_srv = start_http_server(hls_dir, args.port)

    rclpy.init()
    node = CameraStreamBridge(args.topic, args.width, args.height, args.fps,
                              hls_dir, args.jpeg_quality)

    local_ips = get_local_ips()
    print("=" * 64)
    print(f"  Camera Stream Server")
    print(f"  ROS topic  : {args.topic}")
    print(f"  Resolution : {args.width}x{args.height} @ {args.fps} fps")
    print(f"  JPEG quality: {args.jpeg_quality}")
    print(f"")
    print(f"  MJPEG (low latency ~100ms) — use this on AVP Safari:")
    for ip in local_ips:
        print(f"    http://{ip}:{args.port}/mjpeg")
    print(f"")
    print(f"  HLS (AVPlayer compat, ~0.5-1s latency):")
    for ip in local_ips:
        print(f"    http://{ip}:{args.port}/stream.m3u8")
    print("=" * 64)

    def _shutdown(*_):
        print("\nShutting down...")
        node.shutdown()
        http_srv.shutdown()
        rclpy.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        _shutdown()


if __name__ == "__main__":
    main()
