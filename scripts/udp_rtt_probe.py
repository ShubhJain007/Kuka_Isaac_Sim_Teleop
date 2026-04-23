"""UDP round-trip-time probe for Apple Vision Pro echo server.

AVP listens on UDP port 9998 and reflects any incoming packet back
unchanged.  Each probe packet is padded to a configurable payload size
(default 1400 B) so the measured RTT reflects the realistic packet size
used for AVP pose-stream traffic, not a tiny 16-byte packet.

First 16 bytes of every probe are the seq_id + send_time_ns header used
to compute RTT on echo reply.  Remaining bytes are zero padding; the
echo reply must return at least the first 16 bytes unchanged.

Usage standalone (quick test):
    python udp_rtt_probe.py 192.168.1.42
"""

from __future__ import annotations

import socket
import struct
import threading
import time
from collections import deque


_RTT_PORT = 9998
_PACKET_FMT = ">QQ"  # seq_id (uint64) + send_time_ns (uint64), big-endian
_PACKET_SIZE = struct.calcsize(_PACKET_FMT)  # 16 bytes
_SUMMARY_EVERY = 50       # per-probe line print: 1-in-N to avoid log flood
_MAX_STORED = 1000        # cap RTT history to avoid unbounded memory
_BATCH_SIZE  = 10         # flush rolling stats every N received acks
_DEFAULT_PAYLOAD_SIZE = 1400  # bytes on the wire, matches typical AVP pose
                              # packet (stays under standard 1500 B MTU so no
                              # IP fragmentation — RTT reflects single-packet
                              # transit latency honestly).


class UdpRttProbe:
    """Non-blocking UDP RTT measurement against AVP echo server."""

    def __init__(self, avp_ip: str, port: int = _RTT_PORT,
                 payload_size: int = _DEFAULT_PAYLOAD_SIZE):
        if payload_size < _PACKET_SIZE:
            raise ValueError(
                f"payload_size must be >= {_PACKET_SIZE} (room for header)")
        self._ip = avp_ip
        self._port = port
        self._seq = 0
        self._payload_size = int(payload_size)
        # Preallocate a reusable padding buffer for send_probe(); the 16-byte
        # header is overwritten in place each send.
        self._pad_tail = b"\x00" * (self._payload_size - _PACKET_SIZE)

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind(("0.0.0.0", 0))
        self._sock.settimeout(1.0)

        self._lock = threading.Lock()
        self._rtts: deque[float] = deque(maxlen=_MAX_STORED)  # in ms
        self._latest_rtt_ms: float | None = None
        self._recv_count = 0
        self._send_count = 0

        # Rolling _BATCH_SIZE-ack aggregator (for periodic ROS publishing).
        # Every _BATCH_SIZE successful acks, _batch_bucket is flushed into
        # _latest_batch_stats and consumed by poll_batch_avg().
        self._batch_bucket: list[float] = []
        self._latest_batch_stats: dict | None = None

        self._running = threading.Event()
        self._running.set()
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()

    def send_probe(self) -> int:
        """Send one echo probe (size = self._payload_size). Returns seq."""
        self._seq += 1
        header = struct.pack(_PACKET_FMT, self._seq, time.time_ns())
        self._sock.sendto(header + self._pad_tail, (self._ip, self._port))
        self._send_count += 1
        return self._seq

    @property
    def latest_rtt_ms(self) -> float | None:
        with self._lock:
            return self._latest_rtt_ms

    def summary(self) -> str:
        """Return min/avg/max/jitter over stored RTTs, then clear."""
        with self._lock:
            if not self._rtts:
                return "[RTT] No data"
            vals = list(self._rtts)
            self._rtts.clear()
        n = len(vals)
        avg = sum(vals) / n
        mn = min(vals)
        mx = max(vals)
        jitter = (sum((v - avg) ** 2 for v in vals) / n) ** 0.5
        lost = self._send_count - self._recv_count
        return (f"[RTT] {n} probes: min={mn:.1f}ms avg={avg:.1f}ms "
                f"max={mx:.1f}ms jitter={jitter:.1f}ms lost={lost}")

    def close(self):
        """Stop receiver thread and close socket."""
        self._running.clear()
        try:
            self._sock.close()
        except OSError:
            pass
        self._thread.join(2.0)
        print(self.summary())

    def _recv_loop(self):
        # Large recv buffer so we don't truncate the full padded payload
        # (we only need the first 16 bytes, but the datagram is larger).
        _recv_bufsize = max(2048, self._payload_size + 64)
        while self._running.is_set():
            try:
                data, _ = self._sock.recvfrom(_recv_bufsize)
            except socket.timeout:
                continue
            except OSError:
                break

            if len(data) < _PACKET_SIZE:
                continue

            recv_ns = time.time_ns()
            seq, send_ns = struct.unpack(_PACKET_FMT, data[:_PACKET_SIZE])
            rtt_ns = recv_ns - send_ns
            rtt_ms = rtt_ns / 1e6

            with self._lock:
                self._latest_rtt_ms = rtt_ms
                self._rtts.append(rtt_ms)
                self._recv_count += 1
                count = self._recv_count
                # Rolling _BATCH_SIZE-ack bucket — flush on the Nth ack.
                self._batch_bucket.append(rtt_ms)
                if len(self._batch_bucket) >= _BATCH_SIZE:
                    _vals = self._batch_bucket
                    _n = len(_vals)
                    _avg = sum(_vals) / _n
                    _mn  = min(_vals)
                    _mx  = max(_vals)
                    _jit = (sum((v - _avg) ** 2 for v in _vals) / _n) ** 0.5
                    _lost = self._send_count - self._recv_count
                    self._latest_batch_stats = {
                        "n":      _n,
                        "avg_ms": _avg,
                        "min_ms": _mn,
                        "max_ms": _mx,
                        "jitter_ms": _jit,
                        "lost":   _lost,
                        "payload_bytes": self._payload_size,
                    }
                    self._batch_bucket = []

            if count % _SUMMARY_EVERY == 0:
                print(f"[RTT] seq={seq}, RTT={rtt_ms:.1f}ms, one_way≈{rtt_ms / 2:.1f}ms  "
                      f"(payload={self._payload_size}B, printed 1-in-{_SUMMARY_EVERY})")

    def poll_batch_avg(self) -> dict | None:
        """Return the latest batch-of-N stats (consumed once) or None.

        Caller (main loop) polls this each iteration; publishes on ROS topic
        when non-None. Bucket is cleared atomically on flush, so each stats
        dict is delivered exactly once. N = _BATCH_SIZE (currently 10).
        """
        with self._lock:
            stats = self._latest_batch_stats
            self._latest_batch_stats = None
        return stats


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <AVP_IP>")
        sys.exit(1)
    probe = UdpRttProbe(sys.argv[1])
    print(f"Sending probes to {sys.argv[1]}:{_RTT_PORT} every 1s (Ctrl+C to stop)")
    try:
        while True:
            probe.send_probe()
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        probe.close()
