"""
YOLO26s object detector — raw ncnn backend only.

Uses the ncnn Python library directly for inference on both Pi 5 and Pi 4B.
No torch or ultralytics needed at runtime (only for export during setup on Pi 5).
"""

import os
from typing import Dict, List

import cv2
import numpy as np

# ── COCO class names (80 classes) ────────────────────────────────
COCO_NAMES: Dict[int, str] = {
    0: "person",
    39: "bottle", 40: "wine glass", 41: "cup", 42: "fork",
    43: "knife", 44: "spoon", 45: "bowl", 46: "banana", 47: "apple",
    48: "sandwich", 49: "orange", 50: "broccoli", 51: "carrot", 52: "hot dog",
    53: "pizza", 54: "donut", 55: "cake",63: "laptop", 64: "mouse", 65: "remote",67: "cell phone",73: "book",
}


class YoloDetector:
    def __init__(self):
        import ncnn as _ncnn

        self.conf = float(os.environ.get("YOLO_CONF", "0.25"))
        self.nms_thresh = float(os.environ.get("YOLO_NMS", "0.45"))
        self.model_path = os.environ.get("YOLO_MODEL", "yolo26s_ncnn_model")
        self.max_person = int(os.environ.get("YOLO_MAX_PERSON", "1"))
        self._is_pi4 = self._detect_pi4()

        filt = os.environ.get("YOLO_FILTER", "person,cell phone,bottle,cup")
        self.filter_names = {
            self._norm_name(x)
            for x in (p.strip() for p in filt.split(","))
            if x and x.strip()
        }

        # Load class names from metadata.yaml (written by ultralytics export)
        self.names: Dict[int, str] = dict(COCO_NAMES)
        meta_path = os.path.join(self.model_path, "metadata.yaml")
        if os.path.exists(meta_path):
            try:
                import yaml  # type: ignore

                with open(meta_path) as f:
                    meta = yaml.safe_load(f)
                if "names" in meta:
                    self.names = {int(k): str(v) for k, v in meta["names"].items()}
            except Exception:
                pass  # fall back to hardcoded COCO names

        # Load ncnn model
        param_path = os.path.join(self.model_path, "model.ncnn.param")
        bin_path = os.path.join(self.model_path, "model.ncnn.bin")

        if not os.path.exists(param_path):
            raise FileNotFoundError(
                f"NCNN model not found at {param_path}. "
                "Run the setup script or set YOLO_MODEL."
            )

        self._net = _ncnn.Net()
        # Pi 4B: no stable Vulkan; enable FP16 fast paths + packing for ARM NEON.
        self._net.opt.use_vulkan_compute = False
        self._net.opt.use_packing_layout = True
        self._net.opt.use_fp16_storage = True
        self._net.opt.use_fp16_arithmetic = True
        default_threads = "2" if self._is_pi4 else "3"
        self._net.opt.num_threads = int(os.environ.get("NCNN_THREADS", default_threads))
        self._net.load_param(param_path)
        self._net.load_model(bin_path)

        default_input_size = "416" if self._is_pi4 else "640"
        self._input_size = int(os.environ.get("YOLO_INPUT_SIZE", default_input_size))

        print(
            f"[infer] YOLO backend: raw ncnn ({self.model_path}, "
            f"input={self._input_size}x{self._input_size}, "
            f"threads={self._net.opt.num_threads})",
            flush=True,
        )

    # ── public API ───────────────────────────────────────────────
    def detect(self, frame_bgr: np.ndarray) -> List[dict]:
        import ncnn as _ncnn

        h, w = frame_bgr.shape[:2]
        sz = self._input_size

        # Preprocess: resize to sz×sz, BGR→RGB, normalize to 0–1
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

        # Output shape: (84, N) — 4 bbox + 80 class scores per detection
        # N depends on input size: 8400 @ 640, 2100 @ 320, 1344 @ 256
        out = np.array(mat_out)
        if out.ndim == 1:
            out = out.reshape(84, -1)
        if out.shape[0] == 84:
            out = out.T  # → (N, 84)

        num_dets = out.shape[0]
        cx = out[:, 0]
        cy = out[:, 1]
        bw = out[:, 2]
        bh = out[:, 3]
        scores = out[:, 4:]  # (N, 80)

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

        # Scale from sz×sz back to original frame size
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
        if self.max_person > 0:
            pruned: List[dict] = []
            person_seen = 0
            for obj in objects:
                if obj.get("name") == "person":
                    if person_seen >= self.max_person:
                        continue
                    person_seen += 1
                pruned.append(obj)
            objects = pruned
        return objects

    # ── helpers ──────────────────────────────────────────────────
    @staticmethod
    def _norm_name(name: str) -> str:
        n = name.strip().lower()
        if n in ("phone", "cellphone", "mobile", "mobile phone"):
            return "cell phone"
        return n

    @staticmethod
    def _detect_pi4() -> bool:
        try:
            model_path = "/proc/device-tree/model"
            if os.path.exists(model_path):
                with open(model_path, "rb") as f:
                    model = f.read().decode("utf-8", errors="ignore").lower()
                if "raspberry pi 4" in model:
                    return True
        except Exception:
            pass
        return False
