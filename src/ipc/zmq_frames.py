import json
import os
import time
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import zmq


@dataclass
class FrameHeader:
    frame_id: int
    ts_ms: int
    w: int
    h: int
    format: str = "bgr8"

    def to_dict(self) -> Dict:
        return {
            "frame_id": self.frame_id,
            "ts_ms": self.ts_ms,
            "w": self.w,
            "h": self.h,
            "format": self.format,
        }

    @staticmethod
    def from_dict(d: Dict) -> "FrameHeader":
        return FrameHeader(
            frame_id=int(d["frame_id"]),
            ts_ms=int(d["ts_ms"]),
            w=int(d["w"]),
            h=int(d["h"]),
            format=str(d.get("format", "bgr8")),
        )


def now_ms() -> int:
    return int(time.time() * 1000)


class FramePublisher:
    def __init__(self, bind_addr: str):
        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.PUB)
        hwm = int(os.environ.get("FRAMES_SNDHWM", "4"))
        self.sock.setsockopt(zmq.SNDHWM, hwm)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.bind(bind_addr)

    def send(self, frame_bgr: np.ndarray, frame_id: int, ts_ms: int | None = None) -> None:
        if ts_ms is None:
            ts_ms = now_ms()
        h, w = frame_bgr.shape[:2]
        header = FrameHeader(frame_id=frame_id, ts_ms=ts_ms, w=w, h=h, format="bgr8")
        self.sock.send_multipart(
            [json.dumps(header.to_dict()).encode("utf-8"), frame_bgr.tobytes()],
            copy=False,
        )

    def close(self) -> None:
        self.sock.close(linger=0)


class FrameSubscriber:
    def __init__(self, connect_addr: str, conflate: bool = False):
        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.SUB)
        # NOTE: do not use CONFLATE with multipart messages; it can corrupt message
        # boundaries and trigger libzmq assertions. We instead rely on low HWM and
        # draining the queue in recv_latest().
        if conflate:
            self.sock.setsockopt(zmq.CONFLATE, 1)
        # Queue size tunable to balance drops vs latency.
        rcv_hwm = int(os.environ.get("FRAMES_RCVHWM", "4"))
        self.sock.setsockopt(zmq.RCVHWM, rcv_hwm)
        self.sock.setsockopt(zmq.SUBSCRIBE, b"")
        self.sock.connect(connect_addr)

    def recv(self, flags: int = 0) -> Tuple[FrameHeader, np.ndarray]:
        header_b, data_b = self.sock.recv_multipart(flags=flags)
        header = FrameHeader.from_dict(json.loads(header_b.decode("utf-8")))
        if header.format != "bgr8":
            raise ValueError(f"Unsupported frame format: {header.format}")
        frame = np.frombuffer(data_b, dtype=np.uint8).reshape((header.h, header.w, 3))
        return header, frame

    def recv_latest(self, timeout_ms: int = 0) -> Tuple[FrameHeader, np.ndarray] | None:
        """Return the latest available frame, draining any backlog."""
        poller = zmq.Poller()
        poller.register(self.sock, zmq.POLLIN)
        socks = dict(poller.poll(timeout=timeout_ms))
        if socks.get(self.sock) != zmq.POLLIN:
            return None

        latest = None
        while True:
            try:
                latest = self.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
        return latest

    def close(self) -> None:
        self.sock.close(linger=0)
