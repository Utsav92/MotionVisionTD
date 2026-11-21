import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2
import numpy as np
from ai.detection import FootballDetector

def inspect():
    video_path = "assets/videos/21.11.2026_06.43.47_REC.mp4"
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"User recording: {w}x{h} @ {fps:.2f} fps, {total} frames ({total/fps:.2f}s)")

    detector = FootballDetector()
    for i in [0, 50, 100, 150, 200, 250, 300]:
        if i < total:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                dets = detector.detect_frame(frame, i)
                classes = [d['class_name'] for d in dets]
                print(f"  Frame {i:3d}: detected {len(dets)} objects: {classes}")
    cap.release()

if __name__ == "__main__":
    inspect()
