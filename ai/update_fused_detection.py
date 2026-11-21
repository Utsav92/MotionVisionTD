"""
Update and Refine Neural Detection Artifacts:
- Fuses Keypoint R-CNN head & neck topology with LRASPP person segmentation mask.
- Refines ball tracking with shoe-exclusion and velocity smoothing.
- Saves updated data/tracking/user_rec_real_detection.npz and .json.
"""

import os
import sys
import json
import math
import cv2
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def main():
    video_path = os.path.join(PROJECT_ROOT, "assets", "videos", "21.11.2026_06.43.47_REC.mp4")
    npz_path = os.path.join(PROJECT_ROOT, "data", "tracking", "user_rec_real_detection.npz")
    json_path = os.path.join(PROJECT_ROOT, "data", "tracking", "user_rec_real_detection.json")

    print("[1/4] Loading source video and existing neural tracking...")
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, f = cap.read()
        if not ret: break
        frames.append(f)
    cap.release()

    n_frames = len(frames)
    h, w = frames[0].shape[:2]

    data = np.load(npz_path)
    raw_masks = data["masks"].copy()
    raw_kps = data["keypoints"].copy()
    raw_bboxes = data["bboxes"].copy()

    print(f"[2/4] Fusing head & neck geometry on all {n_frames} frames...")
    fused_masks = np.zeros_like(raw_masks)
    updated_bboxes = np.zeros_like(raw_bboxes)

    for i in range(n_frames):
        m = raw_masks[i].copy()
        kp = raw_kps[i]

        # Head & Hair & Backwards Cap coverage
        hx = int((kp[6, 0] + kp[0, 0]) / 2.0 + 4)
        hy = int((min(kp[3, 1], kp[4, 1]) + kp[0, 1]) / 2.0 - 10)
        rx = 34
        ry = 34
        cv2.ellipse(m, (hx, hy), (rx, ry), 0, 0, 360, 1, -1)

        # Neck polygon connecting shoulders and head
        neck_poly = np.array([
            kp[6, :2],
            [hx - rx, hy + ry // 2],
            [hx + rx, hy + ry // 2],
            kp[5, :2]
        ], dtype=np.int32)
        cv2.fillPoly(m, [neck_poly], 1)

        # Left Foot & Shoe (cleats coverage)
        ax_l, ay_l = int(kp[15, 0]), int(kp[15, 1])
        cv2.ellipse(m, (ax_l - 6, ay_l + 12), (34, 16), 18, 0, 360, 1, -1)

        # Right Foot & Shoe (cleats coverage)
        ax_r, ay_r = int(kp[16, 0]), int(kp[16, 1])
        cv2.ellipse(m, (ax_r + 10, ay_r + 12), (34, 16), -15, 0, 360, 1, -1)

        # Morphological smoothing
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)

        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            clean = np.zeros_like(m)
            cv2.drawContours(clean, [c], -1, 1, -1)
            fused_masks[i] = clean
            bx, by, bw, bh = cv2.boundingRect(c)
            updated_bboxes[i] = [bx, by, bx + bw, by + bh]
        else:
            fused_masks[i] = m
            updated_bboxes[i] = raw_bboxes[i]

    print(f"[3/4] Refined ball tracking with shoe rejection across {n_frames} frames...")
    ball_pos = np.zeros((n_frames, 2), dtype=np.float32)
    ball_radii = np.zeros(n_frames, dtype=np.float32)

    curr_pos = np.array([195.0, 238.0], dtype=np.float32)
    curr_r = 18.0
    velocity = np.array([0.0, 0.0], dtype=np.float32)

    for i in range(n_frames):
        img = frames[i]
        kp = raw_kps[i]
        l_ankle = kp[15, :2]
        r_ankle = kp[16, :2]

        pred_pos = curr_pos + velocity * 0.4

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(gray, 185, 255, cv2.THRESH_BINARY)
        cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        for c in cnts:
            area = cv2.contourArea(c)
            if 80 <= area <= 2800:
                bx, by, bw, bh = cv2.boundingRect(c)
                asp = bw / max(1.0, float(bh))
                if 0.65 <= asp <= 1.50:
                    cx = bx + bw / 2.0
                    cy = by + bh / 2.0
                    # Reject shoes (white cleats)
                    d_l = np.linalg.norm([cx - l_ankle[0], cy - l_ankle[1]])
                    d_r = np.linalg.norm([cx - r_ankle[0], cy - r_ankle[1]])
                    if d_l < 38.0 or d_r < 38.0:
                        continue
                    # Reject far field background balls
                    body_cx = (updated_bboxes[i, 0] + updated_bboxes[i, 2]) / 2.0
                    if abs(cx - body_cx) > 130:
                        continue
                    perim = cv2.arcLength(c, True)
                    circ = 4 * math.pi * (area / (perim * perim + 1e-4))
                    dist_to_pred = np.linalg.norm([cx - pred_pos[0], cy - pred_pos[1]])
                    score = circ - (dist_to_pred / 220.0)
                    candidates.append((score, cx, cy, (bw + bh) / 4.0))

        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            best = candidates[0]
            new_pos = np.array([best[1], best[2]], dtype=np.float32)
            curr_pos = curr_pos * 0.40 + new_pos * 0.60
            curr_r = best[3]
            velocity = new_pos - curr_pos
        else:
            curr_pos += velocity * 0.6
            velocity[1] += 0.5 # gravity bias

        ball_pos[i] = curr_pos
        ball_radii[i] = curr_r

    print("[4/4] Saving updated NPZ and JSON files...")
    np.savez_compressed(
        npz_path,
        masks=fused_masks,
        keypoints=raw_kps,
        bboxes=updated_bboxes,
        ball_pos=ball_pos,
        ball_radii=ball_radii
    )

    json_records = []
    for i in range(n_frames):
        json_records.append({
            "frame_idx": int(i),
            "timestamp_sec": round(float(i / 25.0), 3),
            "bbox_xyxy": [round(float(v), 2) for v in updated_bboxes[i]],
            "mask_pixel_count": int(fused_masks[i].sum()),
            "ball_pos_xy": [round(float(ball_pos[i, 0]), 2), round(float(ball_pos[i, 1]), 2)],
            "ball_radius": round(float(ball_radii[i]), 2),
            "keypoints": raw_kps[i].tolist()
        })

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_records, f, indent=2)

    print(f"[SUCCESS] Updated {npz_path} and {json_path} successfully!")

if __name__ == "__main__":
    main()
