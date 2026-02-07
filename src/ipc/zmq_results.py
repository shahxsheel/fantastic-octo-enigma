import json
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import zmq


def now_ms() -> int:
    return int(time.time() * 1000)


@dataclass
class InferResult:
    frame_id: int
    ts_ms: int
    objects: list
    face_bbox: Optional[list] = None  # [x1,y1,x2,y2] in infer coords
    eyes: Optional[dict] = None  # e.g. {left_pct,right_pct,left_state,right_state,...}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "ts_ms": self.ts_ms,
            "objects": self.objects,
            "face_bbox": self.face_bbox,
            "eyes": self.eyes,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "InferResult":
        return InferResult(
            frame_id=int(d["frame_id"]),
            ts_ms=int(d["ts_ms"]),
            objects=list(d.get("objects", [])),
            face_bbox=d.get("face_bbox"),
            eyes=d.get("eyes"),
        )


class ResultPublisher:
    def __init__(self, bind_addr: str):
        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.PUB)
        self.sock.bind(bind_addr)

    def send(self, result: InferResult) -> None:
        self.sock.send_json(result.to_dict())

    def close(self) -> None:
        self.sock.close(linger=0)


class ResultSubscriber:
    def __init__(self, connect_addr: str, conflate: bool = True):
        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.SUB)
        if conflate:
            self.sock.setsockopt(zmq.CONFLATE, 1)
        self.sock.setsockopt(zmq.RCVHWM, 2)
        self.sock.setsockopt(zmq.SUBSCRIBE, b"")
        self.sock.connect(connect_addr)

    def recv(self, flags: int = 0) -> InferResult:
        d = self.sock.recv_json(flags=flags)
        return InferResult.from_dict(d)

    def recv_latest(self, timeout_ms: int = 0) -> InferResult | None:
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
