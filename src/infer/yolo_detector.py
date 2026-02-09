"""
YOLO object detector (NCNN) with letterbox preprocessing and full COCO names.

Goals:
- Accuracy: correct letterbox + unpad so boxes map cleanly to the original frame.
- Simplicity: no downstream class filtering surprises; filter is opt-in.
- Pi-aware defaults: smaller input/threads on Pi4 unless overridden.
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


def _letterbox(
    img_bgr: np.ndarray, new_size: int, pad_color: Tuple[int, int, int] = (114, 114, 114)
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    h, w = img_bgr.shape[:2]
    scale = min(new_size / w, new_size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    pad_w, pad_h = new_size - nw, new_size - nh
    dw, dh = pad_w // 2, pad_h // 2

    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new_size, new_size, 3), pad_color, dtype=np.uint8)
    canvas[dh:dh + nh, dw:dw + nw] = resized
    # Convert to RGB for model
    canvas = canvas[:, :, ::-1]
    return canvas, scale, (dw, dh)


class YoloDetector:
    def __init__(self):
        import ncnn as _ncnn

        self.conf = float(os.environ.get("YOLO_CONF", "0.25"))
        self.nms_thresh = float(os.environ.get("YOLO_NMS", "0.45"))
        self.model_path = os.environ.get("YOLO_MODEL", "yolo26s_ncnn_model")
        self.max_person = int(os.environ.get("YOLO_MAX_PERSON", "1"))
        self.max_dets = int(os.environ.get("YOLO_MAX_DETS", "200"))
        self._is_pi4 = _detect_pi4()

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

        img_rgb, scale, (dw, dh) = _letterbox(frame_bgr, sz)
        mat_in = _ncnn.Mat.from_pixels(img_rgb, _ncnn.Mat.PixelType.PIXEL_RGB, sz, sz)
        mat_in.substract_mean_normalize([0.0, 0.0, 0.0], [1 / 255.0] * 3)

        ex = self._net.create_extractor()
        ex.input("in0", mat_in)
        _ret, mat_out = ex.extract("out0")

        out = np.array(mat_out)
        if out.ndim == 1:
            out = out.reshape(84, -1)
        if out.shape[0] == 84:
            out = out.T  # (N, 84)

        if out.size == 0:
            return []

        cx = out[:, 0]
        cy = out[:, 1]
        bw = out[:, 2]
        bh = out[:, 3]
        scores = out[:, 4:]

        class_ids = np.argmax(scores, axis=1)
        max_scores = scores[np.arange(scores.shape[0]), class_ids]

        mask = max_scores > self.conf
        if not np.any(mask):
            return []

        cx = cx[mask]
        cy = cy[mask]
        bw = bw[mask]
        bh = bh[mask]
        class_ids = class_ids[mask]
        max_scores = max_scores[mask]

        # Undo letterbox
        x1 = (cx - bw / 2 - dw) / scale
        y1 = (cy - bh / 2 - dh) / scale
        x2 = (cx + bw / 2 - dw) / scale
        y2 = (cy + bh / 2 - dh) / scale

        boxes_xywh = list(zip(x1.tolist(), y1.tolist(), (x2 - x1).tolist(), (y2 - y1).tolist()))
        confidences = max_scores.tolist()

        indices = cv2.dnn.NMSBoxes(boxes_xywh, confidences, self.conf, self.nms_thresh)

        objects: List[dict] = []
        for i in indices:
            idx = int(i) if isinstance(i, (int, np.integer)) else int(i[0])
            cls = int(class_ids[idx])
            name = self.names.get(cls, str(cls))
            if self.filter_names and self._norm_name(name) not in self.filter_names:
                continue
            bx, by, bwb, bhb = boxes_xywh[idx]
            x1i = max(0, min(int(bx), w - 1))
            y1i = max(0, min(int(by), h - 1))
            x2i = max(0, min(int(bx + bwb), w - 1))
            y2i = max(0, min(int(by + bhb), h - 1))
            objects.append(
                {
                    "cls": cls,
                    "name": name,
                    "conf": float(max_scores[idx]),
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
        if n in ("phone", "cellphone", "mobile", "mobile phone", "cell phone"):
            return "cell phone"
        return n
