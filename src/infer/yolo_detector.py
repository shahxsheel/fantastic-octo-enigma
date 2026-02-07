"""
YOLOv8s object detector with dual backend:
  - ultralytics (Pi 5 — uses torch)
  - raw ncnn (Pi 4B — no torch dependency)

The backend is selected automatically at startup: if ultralytics is importable,
it is used; otherwise the raw ncnn backend is used.
"""

import os
from typing import Dict, List

import cv2
import numpy as np

# ── COCO class names (80 classes) ────────────────────────────────
COCO_NAMES: Dict[int, str] = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
    10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
    14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
    20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack",
    25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee",
    30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite",
    34: "baseball bat", 35: "baseball glove", 36: "skateboard", 37: "surfboard",
    38: "tennis racket", 39: "bottle", 40: "wine glass", 41: "cup", 42: "fork",
    43: "knife", 44: "spoon", 45: "bowl", 46: "banana", 47: "apple",
    48: "sandwich", 49: "orange", 50: "broccoli", 51: "carrot", 52: "hot dog",
    53: "pizza", 54: "donut", 55: "cake", 56: "chair", 57: "couch",
    58: "potted plant", 59: "bed", 60: "dining table", 61: "toilet", 62: "tv",
    63: "laptop", 64: "mouse", 65: "remote", 66: "keyboard", 67: "cell phone",
    68: "microwave", 69: "oven", 70: "toaster", 71: "sink", 72: "refrigerator",
    73: "book", 74: "clock", 75: "vase", 76: "scissors", 77: "teddy bear",
    78: "hair drier", 79: "toothbrush",
}


class YoloDetector:
    def __init__(self):
        self.conf = float(os.environ.get("YOLO_CONF", "0.25"))
        self.nms_thresh = float(os.environ.get("YOLO_NMS", "0.45"))
        self.model_path = os.environ.get("YOLO_MODEL", "yolov8s_ncnn_model")

        filt = os.environ.get("YOLO_FILTER", "person,cell phone,bottle,cup")
        self.filter_names = {
            self._norm_name(x)
            for x in (p.strip() for p in filt.split(","))
            if x and x.strip()
        }

        self._backend = None
        self.names: Dict[int, str] = {}

        # Try ultralytics first (Pi 5), fall back to raw ncnn (Pi 4B)
        try:
            from ultralytics import YOLO  # noqa: F811

            resolved = self.model_path
            if os.path.isdir(resolved) and resolved.endswith("_ncnn_model"):
                try:
                    import ncnn  # noqa: F401
                except Exception:
                    resolved = os.environ.get("YOLO_FALLBACK", "yolov8s.pt")
            if not os.path.exists(resolved):
                resolved = os.environ.get("YOLO_FALLBACK", "yolov8s.pt")

            self._ul_model = YOLO(resolved, task="detect")
            self.names = dict(self._ul_model.names)
            self._backend = "ultralytics"
            print(f"[infer] YOLO backend: ultralytics ({resolved})", flush=True)

        except (ImportError, Exception) as exc:
            print(f"[infer] ultralytics not available ({exc}), using raw ncnn", flush=True)
            self._init_ncnn()

    # ── raw ncnn init ────────────────────────────────────────────
    def _init_ncnn(self) -> None:
        import ncnn as _ncnn

        model_dir = self.model_path
        param_path = os.path.join(model_dir, "model.ncnn.param")
        bin_path = os.path.join(model_dir, "model.ncnn.bin")
        meta_path = os.path.join(model_dir, "metadata.yaml")

        if not os.path.exists(param_path):
            raise FileNotFoundError(
                f"NCNN model not found at {param_path}. "
                "Run the setup script or set YOLO_MODEL."
            )

        # Load class names from metadata.yaml (written by ultralytics export)
        self.names = dict(COCO_NAMES)
        if os.path.exists(meta_path):
            try:
                import yaml  # type: ignore

                with open(meta_path) as f:
                    meta = yaml.safe_load(f)
                if "names" in meta:
                    self.names = {int(k): str(v) for k, v in meta["names"].items()}
            except Exception:
                pass  # fall back to hardcoded COCO names

        self._net = _ncnn.Net()
        self._net.opt.use_vulkan_compute = False
        self._net.opt.num_threads = int(os.environ.get("NCNN_THREADS", "4"))
        self._net.load_param(param_path)
        self._net.load_model(bin_path)

        self._input_size = 640
        self._backend = "ncnn"
        print(
            f"[infer] YOLO backend: raw ncnn ({model_dir}, "
            f"threads={self._net.opt.num_threads})",
            flush=True,
        )

    # ── public API ───────────────────────────────────────────────
    def detect(self, frame_bgr: np.ndarray) -> List[dict]:
        if self._backend == "ultralytics":
            return self._detect_ultralytics(frame_bgr)
        else:
            return self._detect_ncnn(frame_bgr)

    # ── ultralytics backend (Pi 5) ───────────────────────────────
    def _detect_ultralytics(self, frame_bgr: np.ndarray) -> List[dict]:
        results = self._ul_model.predict(source=frame_bgr, verbose=False, conf=self.conf)
        objects: List[dict] = []
        for r in results:
            if r.boxes is None:
                continue
            for b in r.boxes:
                cls = int(b.cls[0])
                conf = float(b.conf[0])
                xyxy = b.xyxy[0].cpu().numpy().tolist()
                x1, y1, x2, y2 = [int(v) for v in xyxy]
                name = self.names.get(cls, str(cls))
                if self.filter_names and self._norm_name(name) not in self.filter_names:
                    continue
                objects.append(
                    {"cls": cls, "name": name, "conf": conf, "bbox": [x1, y1, x2, y2]}
                )
        objects.sort(key=lambda o: o["conf"], reverse=True)
        return objects

    # ── raw ncnn backend (Pi 4B) ─────────────────────────────────
    def _detect_ncnn(self, frame_bgr: np.ndarray) -> List[dict]:
        import ncnn as _ncnn

        h, w = frame_bgr.shape[:2]
        sz = self._input_size

        # Preprocess: resize to 640x640, BGR→RGB, normalize to 0–1
        mat_in = _ncnn.Mat.from_pixels_resize(
            frame_bgr, _ncnn.Mat.PixelType.PIXEL_BGR2RGB, w, h, sz, sz
        )
        mat_in.substract_mean_normalize(
            [0.0, 0.0, 0.0],
            [1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0],
        )

        # Run model
        ex = self._net.create_extractor()
        ex.input("in0", mat_in)
        _ret, mat_out = ex.extract("out0")

        # Output shape: (84, 8400) — 4 bbox + 80 class scores per detection
        out = np.array(mat_out)
        if out.ndim == 1:
            out = out.reshape(84, -1)
        if out.shape[0] == 84:
            out = out.T  # → (8400, 84)

        num_dets = out.shape[0]
        cx = out[:, 0]
        cy = out[:, 1]
        bw = out[:, 2]
        bh = out[:, 3]
        scores = out[:, 4:]  # (8400, 80)

        # Max class score per detection
        class_ids = np.argmax(scores, axis=1)
        max_scores = scores[np.arange(num_dets), class_ids]

        # Confidence filter
        mask = max_scores > self.conf
        if not np.any(mask):
            return []

        cx = cx[mask]
        cy = cy[mask]
        bw = bw[mask]
        bh = bh[mask]
        class_ids = class_ids[mask]
        max_scores = max_scores[mask]

        # Scale from 640x640 back to original frame size
        scale_x = w / sz
        scale_y = h / sz

        # Convert cx,cy,w,h → x,y,w,h (top-left) for NMS
        x_tl = (cx - bw / 2.0) * scale_x
        y_tl = (cy - bh / 2.0) * scale_y
        w_box = bw * scale_x
        h_box = bh * scale_y

        boxes = list(zip(x_tl.tolist(), y_tl.tolist(), w_box.tolist(), h_box.tolist()))
        confidences = max_scores.tolist()

        # NMS via OpenCV
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf, self.nms_thresh)

        objects: List[dict] = []
        for i in indices:
            idx = int(i) if isinstance(i, (int, np.integer)) else int(i[0])
            cls = int(class_ids[idx])
            name = self.names.get(cls, str(cls))
            if self.filter_names and self._norm_name(name) not in self.filter_names:
                continue
            bx, by, bwidth, bheight = boxes[idx]
            objects.append(
                {
                    "cls": cls,
                    "name": name,
                    "conf": float(max_scores[idx]),
                    "bbox": [int(bx), int(by), int(bx + bwidth), int(by + bheight)],
                }
            )

        objects.sort(key=lambda o: o["conf"], reverse=True)
        return objects

    # ── helpers ──────────────────────────────────────────────────
    @staticmethod
    def _norm_name(name: str) -> str:
        n = name.strip().lower()
        if n in ("phone", "cellphone", "mobile", "mobile phone"):
            return "cell phone"
        return n
