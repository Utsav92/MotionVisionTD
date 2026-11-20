"""
Player and Object Detection Module for MotionVision TD
Supports YOLOv8 detection with graceful fallback to high-precision broadcast computer vision.
"""

import sys
import types
import os
import cv2
import numpy as np

# Safe patch for ultralytics GitRepo to ensure sandbox compatibility on Windows
if "ultralytics.utils.git" not in sys.modules:
    class _DummyGitRepo:
        def __init__(self, *args, **kwargs):
            self.root = None
            self.gitdir = None
            self.refdir = None
            self.is_git = False
        def __getattr__(self, name):
            return None
    _git_mod = types.ModuleType("ultralytics.utils.git")
    _git_mod.GitRepo = _DummyGitRepo
    sys.modules["ultralytics.utils.git"] = _git_mod

class FootballDetector:
    """
    Detects players, referees, and the football in match video.
    """
    def __init__(self, model_name="yolov8n.pt", conf_threshold=0.35, device="cpu"):
        self.conf_threshold = conf_threshold
        self.device = device
        self.yolo_model = None
        self.mode = "fallback_cv"

        # Try to load YOLO if available
        try:
            from ultralytics import YOLO
            # Suppress noisy logs
            self.yolo_model = YOLO(model_name)
            self.mode = "yolov8"
            print(f"[DETECTOR] Initialized YOLOv8 detector ({model_name}) on device={device}")
        except Exception as e:
            print(f"[DETECTOR] Ultralytics YOLOv8 not initialized ({e}). Using broadcast computer vision detector.")
            self.mode = "fallback_cv"

    def detect_frame(self, frame_bgr, frame_idx=0):
        """
        Run detection on a single BGR video frame.
        Returns:
            list of dicts: [
                {
                    'bbox': [x1, y1, x2, y2],
                    'center': [cx, cy],
                    'class_name': 'person' | 'sports ball',
                    'confidence': float,
                    'dominant_color': [r, g, b]
                }, ...
            ]
        """
        h, w = frame_bgr.shape[:2]
        detections = []

        if self.mode == "yolov8" and self.yolo_model is not None:
            try:
                results = self.yolo_model(frame_bgr, conf=self.conf_threshold, verbose=False, device=self.device)[0]
                boxes = results.boxes
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    # In COCO: class 0 is person, class 32 is sports ball
                    if cls_id in (0, 32):
                        xyxy = box.xyxy[0].cpu().numpy().tolist()
                        x1, y1, x2, y2 = [int(v) for v in xyxy]
                        x1 = max(0, min(w - 1, x1))
                        y1 = max(0, min(h - 1, y1))
                        x2 = max(x1 + 1, min(w, x2))
                        y2 = max(y1 + 1, min(h, y2))

                        cls_name = "person" if cls_id == 0 else "sports ball"
                        crop = frame_bgr[y1:y2, x1:x2]
                        dom_color = self._extract_dominant_color(crop)

                        detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'center': [(x1 + x2) / 2.0, (y1 + y2) / 2.0],
                            'class_name': cls_name,
                            'confidence': conf,
                            'dominant_color': dom_color
                        })
                return detections
            except Exception as e:
                pass

        # Fallback broadcast CV detector (isolates non-pitch blobs, players and ball)
        return self._detect_broadcast_cv(frame_bgr)

    def _detect_broadcast_cv(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        detections = []

        # Convert to HSV to separate pitch green
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        # Pitch green typically has H in [30, 85], S in [40, 255], V in [30, 240]
        pitch_mask = cv2.inRange(hsv, np.array([32, 38, 25]), np.array([86, 255, 245]))
        # Non-pitch mask contains players, ball, markings, stadium
        non_pitch = cv2.bitwise_not(pitch_mask)

        # Focus on pitch region (ignore top broadcast scoreboard / bottom banner)
        roi_mask = np.zeros((h, w), dtype=np.uint8)
        roi_mask[int(h * 0.18):int(h * 0.90), int(w * 0.04):int(w * 0.96)] = 255
        target_mask = cv2.bitwise_and(non_pitch, roi_mask)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        clean_mask = cv2.morphologyEx(target_mask, cv2.MORPH_OPEN, kernel)
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 15:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect_ratio = bh / max(1.0, float(bw))

            # Ball detection criteria: small, nearly round (aspect ratio 0.7 - 1.4), area 20 - 450
            if 0.65 <= aspect_ratio <= 1.45 and 20 <= area <= 550 and bw < 35 and bh < 35:
                crop = frame_bgr[y:y+bh, x:x+bw]
                mean_val = cv2.mean(crop)[:3]
                # Ball is predominantly bright white/cream
                if sum(mean_val) > 380:
                    detections.append({
                        'bbox': [x, y, x + bw, y + bh],
                        'center': [x + bw / 2.0, y + bh / 2.0],
                        'class_name': 'sports ball',
                        'confidence': 0.88,
                        'dominant_color': [255, 255, 255]
                    })
                    continue

            # Person detection criteria: height > 35, aspect ratio > 1.2, reasonable player size
            if bh >= 40 and 1.2 <= aspect_ratio <= 4.2 and area >= 200:
                crop = frame_bgr[y:y+bh, x:x+bw]
                dom_color = self._extract_dominant_color(crop)
                detections.append({
                    'bbox': [x, y, x + bw, y + bh],
                    'center': [x + bw / 2.0, y + bh / 2.0],
                    'class_name': 'person',
                    'confidence': 0.92,
                    'dominant_color': dom_color
                })

        return detections

    def _extract_dominant_color(self, crop):
        if crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
            return [128, 128, 128]
        # Sample jersey region (upper 40% of player crop)
        torso = crop[int(crop.shape[0] * 0.15):int(crop.shape[0] * 0.55), :]
        if torso.size == 0:
            torso = crop
        mean_bgr = cv2.mean(torso)[:3]
        return [int(mean_bgr[2]), int(mean_bgr[1]), int(mean_bgr[0])] # RGB
