import os
from typing import Dict

import zmq


class AlertPublisher:
    def __init__(self, bind_addr: str):
        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.PUB)
        hwm = int(os.environ.get("ALERTS_SNDHWM", "2"))
        self.sock.setsockopt(zmq.SNDHWM, hwm)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.bind(bind_addr)

    def send(self, payload: Dict) -> None:
        self.sock.send_json(payload)

    def close(self) -> None:
        self.sock.close(linger=0)
