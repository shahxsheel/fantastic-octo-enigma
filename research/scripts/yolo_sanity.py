"""
Quick YOLO sanity runner (NCNN) to test detector independently.

Usage:
  python scripts/yolo_sanity.py --image path/to.jpg
  python scripts/yolo_sanity.py --camera   (uses camera_source; HEADLESS=1 supported)

Env knobs (same as runtime):
  YOLO_MODEL, YOLO_INPUT_SIZE, YOLO_CONF, YOLO_NMS, YOLO_FILTER, NCNN_THREADS
"""

import argparse
import os
from typing import Optional

import cv2

from src.infer.yolo_detector import YoloDetector
from src.camera.camera_source import open_camera


def run_on_image(path: str, save: Optional[str]) -> None:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    det = YoloDetector()
    objs = det.detect(img)
    print(f"Detections ({len(objs)}):")
    for o in objs:
        print(o)
    if save:
        for o in objs:
            x1, y1, x2, y2 = o["bbox"]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 165, 255), 2)
            cv2.putText(img, f"{o['name']} {o['conf']:.2f}", (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
        cv2.imwrite(save, img)
        print(f"Saved: {save}")


def run_on_camera(frames: int, save: Optional[str]) -> None:
    cam, cam_name = open_camera(headless=True)
    print(f"[yolo_sanity] camera={cam_name}")
    det = YoloDetector()
    last = None
    for i in range(frames):
        bundle = cam.read()
        last = bundle.main_bgr
        objs = det.detect(bundle.infer_bgr)
        print(f"Frame {i} -> {len(objs)} dets")
        for o in objs:
            print(o)
    if save and last is not None:
        objs = det.detect(last)
        for o in objs:
            x1, y1, x2, y2 = o["bbox"]
            cv2.rectangle(last, (x1, y1), (x2, y2), (0, 165, 255), 2)
            cv2.putText(last, f"{o['name']} {o['conf']:.2f}", (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
        cv2.imwrite(save, last)
        print(f"Saved: {save}")
    cam.release()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, help="Path to image")
    parser.add_argument("--camera", action="store_true", help="Capture from camera")
    parser.add_argument("--frames", type=int, default=3, help="Frames to run in camera mode")
    parser.add_argument("--save", type=str, help="Optional output image path")
    args = parser.parse_args()

    if args.image:
        run_on_image(args.image, args.save)
    else:
        run_on_camera(args.frames, args.save)


if __name__ == "__main__":
    main()
