import os
from typing import List

import numpy as np
from ultralytics import YOLO


class YoloDetector:
    def __init__(self):
        self.conf = float(os.environ.get("YOLO_CONF", "0.25"))
        self.model_path = os.environ.get("YOLO_MODEL", "yolov8s_ncnn_model")

        filt = os.environ.get("YOLO_FILTER", "person,cell phone,bottle,cup")
        self.filter_names = {
            self._norm_name(x)
            for x in (p.strip() for p in filt.split(","))
            if x and x.strip()
        }

        resolved = self.model_path
        if os.path.isdir(resolved) and resolved.endswith("_ncnn_model"):
            try:
                import ncnn  # noqa: F401
            except Exception:
                resolved = os.environ.get("YOLO_FALLBACK", "yolov8s.pt")

        if isinstance(resolved, str) and not os.path.exists(resolved):
            resolved = os.environ.get("YOLO_FALLBACK", "yolov8s.pt")

        self.resolved = resolved
        print(f"[infer] YOLO loading: {self.resolved}")
        self.model = YOLO(self.resolved, task="detect")

    @staticmethod
    def _norm_name(name: str) -> str:
        n = name.strip().lower()
        if n in ("phone", "cellphone", "mobile", "mobile phone"):
            return "cell phone"
        return n

    def detect(self, frame_bgr: np.ndarray) -> List[dict]:
        results = self.model.predict(source=frame_bgr, verbose=False, conf=self.conf)
        objects: List[dict] = []
        for r in results:
            if r.boxes is None:
                continue
            for b in r.boxes:
                cls = int(b.cls[0])
                conf = float(b.conf[0])
                xyxy = b.xyxy[0].cpu().numpy().tolist()
                x1, y1, x2, y2 = [int(v) for v in xyxy]
                name = self.model.names.get(cls, str(cls))
                if self.filter_names and self._norm_name(name) not in self.filter_names:
                    continue
                objects.append(
                    {"cls": cls, "name": name, "conf": conf, "bbox": [x1, y1, x2, y2]}
                )
        objects.sort(key=lambda o: o["conf"], reverse=True)
        return objects
