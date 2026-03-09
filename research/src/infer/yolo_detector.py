"""
YOLO object detector (NCNN) with letterbox preprocessing and full COCO names.

Goals:
- Accuracy: correct letterbox + unpad so boxes map cleanly to the original frame.
- Simplicity: no downstream class filtering surprises; filter is opt-in.
- Pi-aware defaults: tuned for Raspberry Pi CPU inference unless overridden.

Output format auto-detection:
- E2E/NMS-free models: out0 shape (N, 6) = [x1,y1,x2,y2,conf,cls_id]
- Classic models (including current YOLO26n NCNN export): out0 shape (N, 84) = [cx,cy,w,h,class0..79]
"""

import os
from typing import Dict, List, Tuple

import cv2
import numpy as np

# Full 80-class COCO list (index -> name)
COCO_NAMES: Dict[int, str] = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane", 5: "bus",
    6: "train", 7: "truck", 8: "boat", 9: "traffic light", 10: "fire hydrant",
    11: "stop sign", 12: "parking meter", 13: "bench", 14: "bird", 15: "cat",
    16: "dog", 17: "horse", 18: "sheep", 19: "cow", 20: "elephant", 21: "bear",
    22: "zebra", 23: "giraffe", 24: "backpack", 25: "umbrella", 26: "handbag",
    27: "tie", 28: "suitcase", 29: "frisbee", 30: "skis", 31: "snowboard",
    32: "sports ball", 33: "kite", 34: "baseball bat", 35: "baseball glove",
    36: "skateboard", 37: "surfboard", 38: "tennis racket", 39: "bottle",
    40: "wine glass", 41: "cup", 42: "fork", 43: "knife", 44: "spoon", 45: "bowl",
    46: "banana", 47: "apple", 48: "sandwich", 49: "orange", 50: "broccoli",
    51: "carrot", 52: "hot dog", 53: "pizza", 54: "donut", 55: "cake",
    56: "chair", 57: "couch", 58: "potted plant", 59: "bed", 60: "dining table",
    61: "toilet", 62: "tv", 63: "laptop", 64: "mouse", 65: "remote", 66: "keyboard",
    67: "cell phone", 68: "microwave", 69: "oven", 70: "toaster", 71: "sink",
    72: "refrigerator", 73: "book", 74: "clock", 75: "vase", 76: "scissors",
    77: "teddy bear", 78: "hair drier", 79: "toothbrush",
}


def _detect_pi() -> bool:
    """Returns True for any Raspberry Pi (4, 5, or later)."""
    try:
        model_path = "/proc/device-tree/model"
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                model = f.read().decode("utf-8", errors="ignore").lower()
            if "raspberry pi" in model:
                return True
    except Exception:
        pass
    return False


def _detect_pi4() -> bool:
    """Kept for backward compatibility; use _detect_pi() for new code."""
    return _detect_pi()


def _letterbox(
    img_bgr: np.ndarray, new_size: int, pad_color: Tuple[int, int, int] = (114, 114, 114)
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """Legacy module-level letterbox (unused; kept for backward compat)."""
    h, w = img_bgr.shape[:2]
    scale = min(new_size / w, new_size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    pad_w, pad_h = new_size - nw, new_size - nh
    dw, dh = pad_w // 2, pad_h // 2

    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new_size, new_size, 3), pad_color, dtype=np.uint8)
    canvas[dh:dh + nh, dw:dw + nw] = resized
    canvas = canvas[:, :, ::-1]
    return canvas, scale, (dw, dh)


_PHONE_NAMES: frozenset[str] = frozenset(
    {"phone", "cellphone", "mobile", "mobile phone", "cell phone"}
)


class YoloDetector:
    def __init__(self):
        import ncnn as _ncnn

        self.conf = float(os.environ.get("YOLO_CONF", "0.25"))
        self.nms_thresh = float(os.environ.get("YOLO_NMS", "0.45"))
        self.model_path = os.environ.get("YOLO_MODEL", "yolo26n_ncnn_model")
        self.max_person = int(os.environ.get("YOLO_MAX_PERSON", "1"))
        self.max_dets = int(os.environ.get("YOLO_MAX_DETS", "200"))

        filt = os.environ.get("YOLO_FILTER", "").strip()
        self.filter_names = {
            self._norm_name(x)
            for x in (p.strip() for p in filt.split(","))
            if x and x.strip()
        } if filt != "" else set()

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
                pass

        param_path = os.path.join(self.model_path, "model.ncnn.param")
        bin_path = os.path.join(self.model_path, "model.ncnn.bin")
        if not os.path.exists(param_path):
            raise FileNotFoundError(
                f"NCNN model not found at {param_path}. "
                "Run the setup script or set YOLO_MODEL."
            )

        self._net = _ncnn.Net()
        self._net.opt.use_vulkan_compute = False
        self._net.opt.use_packing_layout = True
        self._net.opt.use_fp16_storage = True
        self._net.opt.use_fp16_arithmetic = True
        # Use all cores for YOLO — inference is the bottleneck, not thread contention (env NCNN_THREADS can override).
        self._net.opt.num_threads = 4
        if os.environ.get("NCNN_THREADS") is not None:
            self._net.opt.num_threads = int(os.environ.get("NCNN_THREADS", "4"))
        self._net.load_param(param_path)
        self._net.load_model(bin_path)

        # Ultra-low resolution for maximum FPS (Nano model, close-up camera).
        # 256x256 is ~1.6x faster than 320x320, sufficient for detecting large objects (person, phone).
        default_input_size = "256"
        self._input_size = int(os.environ.get("YOLO_INPUT_SIZE", default_input_size))

        # Pre-allocate letterbox buffers to avoid per-frame allocations.
        sz = self._input_size
        self._canvas = np.full((sz, sz, 3), 114, dtype=np.uint8)
        self._canvas_rgb = np.empty((sz, sz, 3), dtype=np.uint8)

        print(
            f"[infer] YOLO backend: raw ncnn ({self.model_path}, "
            f"input={self._input_size}x{self._input_size}, "
            f"threads={self._net.opt.num_threads})",
            flush=True,
        )

    def _letterbox_fast(self, img_bgr: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """Reuses pre-allocated canvas buffers to avoid per-frame allocations."""
        new_size = self._input_size
        h, w = img_bgr.shape[:2]
        scale = min(new_size / w, new_size / h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        dw, dh = (new_size - nw) // 2, (new_size - nh) // 2

        # Fill only pad strips, not entire canvas
        self._canvas[:dh, :] = 114
        self._canvas[dh + nh:, :] = 114
        self._canvas[dh:dh + nh, :dw] = 114
        self._canvas[dh:dh + nh, dw + nw:] = 114
        # Resize directly into the canvas ROI — eliminates the intermediate `resized` allocation.
        cv2.resize(img_bgr, (nw, nh), dst=self._canvas[dh:dh + nh, dw:dw + nw], interpolation=cv2.INTER_LINEAR)
        cv2.cvtColor(self._canvas, cv2.COLOR_BGR2RGB, dst=self._canvas_rgb)
        return self._canvas_rgb, scale, (dw, dh)

    # ── public API ───────────────────────────────────────────────
    def detect(self, frame_bgr: np.ndarray) -> List[dict]:
        import ncnn as _ncnn

        h, w = frame_bgr.shape[:2]
        sz = self._input_size

        img_rgb, scale, (dw, dh) = self._letterbox_fast(frame_bgr)
        mat_in = _ncnn.Mat.from_pixels(img_rgb, _ncnn.Mat.PixelType.PIXEL_RGB, sz, sz)
        mat_in.substract_mean_normalize([0.0, 0.0, 0.0], [1 / 255.0] * 3)

        ex = self._net.create_extractor()
        ex.input("in0", mat_in)
        _ret, mat_out = ex.extract("out0")

        # Copy away from NCNN-managed buffers before post-processing.
        out = np.asarray(mat_out, dtype=np.float32).copy()
        if not np.isfinite(out).all():
            out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

        if out.size == 0:
            return []

        # ── auto-detect output format ────────────────────────────────────────────────
        # E2E/NMS-free: (N, 6) = [x1, y1, x2, y2, conf, cls_id]
        # Classic:      (84, N) or (N, 84) = [cx,cy,w,h,cls0..79]
        if out.ndim == 2 and out.shape[1] == 6:
            return self._decode_e2e(out, scale, dw, dh, w, h)

        # Classic path: normalise to (N, 84)
        if out.ndim == 1:
            out = out.reshape(84, -1)
        if out.shape[0] == 84:
            out = out.T  # (N, 84)
        return self._decode_classic(out, scale, dw, dh, w, h)

    # ── E2E decoder (for models exported with end-to-end head) ──────────────────────
    def _decode_e2e(
        self,
        out: np.ndarray,
        scale: float,
        dw: int,
        dh: int,
        img_w: int,
        img_h: int,
    ) -> List[dict]:
        """Decode end-to-end NMS-free output: (N, 6) = [x1,y1,x2,y2,conf,cls_id]."""
        confs = out[:, 4]
        mask = confs > self.conf
        if not np.any(mask):
            return []
        out = out[mask]

        x1 = np.clip((out[:, 0] - dw) / scale, 0, img_w - 1)
        y1 = np.clip((out[:, 1] - dh) / scale, 0, img_h - 1)
        x2 = np.clip((out[:, 2] - dw) / scale, 0, img_w - 1)
        y2 = np.clip((out[:, 3] - dh) / scale, 0, img_h - 1)

        # No manual NMS — model ensures one prediction per object.
        return self._build_objects(x1, y1, x2, y2, out[:, 4], out[:, 5].astype(int), img_w, img_h)

    # ── Classic decoder (YOLOv8/11 and current YOLO26n NCNN export) ─────────────────
    def _decode_classic(
        self,
        out: np.ndarray,
        scale: float,
        dw: int,
        dh: int,
        img_w: int,
        img_h: int,
    ) -> List[dict]:
        """Decode classic anchor-free output: (N, 84) = [cx,cy,w,h,class0..79]."""
        cx = out[:, 0]
        cy = out[:, 1]
        bw = out[:, 2]
        bh = out[:, 3]
        scores = out[:, 4:]

        class_ids = np.argmax(scores, axis=1)
        max_scores = np.max(scores, axis=1)

        mask = max_scores > self.conf
        if not np.any(mask):
            return []

        cx, cy, bw, bh = cx[mask], cy[mask], bw[mask], bh[mask]
        class_ids, max_scores = class_ids[mask], max_scores[mask]

        # Undo letterbox
        x1 = (cx - bw / 2 - dw) / scale
        y1 = (cy - bh / 2 - dh) / scale
        x2 = (cx + bw / 2 - dw) / scale
        y2 = (cy + bh / 2 - dh) / scale

        boxes_xywh = list(zip(x1.tolist(), y1.tolist(), (x2 - x1).tolist(), (y2 - y1).tolist()))
        indices = cv2.dnn.NMSBoxes(boxes_xywh, max_scores.tolist(), self.conf, self.nms_thresh)

        if indices is None or len(indices) == 0:
            return []

        idx_list = [int(i) if isinstance(i, (int, np.integer)) else int(i[0]) for i in indices]
        return self._build_objects(
            x1[idx_list], y1[idx_list], x2[idx_list], y2[idx_list],
            max_scores[idx_list], class_ids[idx_list],
            img_w, img_h,
        )

    # ── shared post-processing ───────────────────────────────────────────────────────
    def _build_objects(
        self,
        x1: np.ndarray,
        y1: np.ndarray,
        x2: np.ndarray,
        y2: np.ndarray,
        confs: np.ndarray,
        cls_ids: np.ndarray,
        img_w: int,
        img_h: int,
    ) -> List[dict]:
        """Clip boxes, apply filter, sort, prune by max_person / max_dets."""
        max_x = max(img_w - 1, 0)
        max_y = max(img_h - 1, 0)
        objects: List[dict] = []
        for i in range(len(confs)):
            cls = int(cls_ids[i])
            name = self.names.get(cls, str(cls))
            if self.filter_names and self._norm_name(name) not in self.filter_names:
                continue
            x1i = max(0, min(int(x1[i]), max_x))
            y1i = max(0, min(int(y1[i]), max_y))
            x2i = max(0, min(int(x2[i]), max_x))
            y2i = max(0, min(int(y2[i]), max_y))
            if x2i < x1i:
                x1i, x2i = x2i, x1i
            if y2i < y1i:
                y1i, y2i = y2i, y1i
            objects.append(
                {
                    "cls": cls,
                    "name": name,
                    "conf": float(confs[i]),
                    "bbox": [x1i, y1i, x2i, y2i],
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

        if self.max_dets > 0 and len(objects) > self.max_dets:
            objects = objects[: self.max_dets]

        return objects

    @staticmethod
    def _norm_name(name: str) -> str:
        n = name.strip().lower()
        if n in _PHONE_NAMES:
            return "cell phone"
        return n
