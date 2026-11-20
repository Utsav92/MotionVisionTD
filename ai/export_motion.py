"""
Master Motion Export Pipeline for MotionVision TD
Orchestrates detection, tracking, ball tracking, 3D SMPL reconstruction,
camera estimation, and biomechanics export for TouchDesigner consumption.
"""

import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import cv2
import numpy as np

from ai.detection import FootballDetector
from ai.tracking import MultiObjectTracker
from ai.player_identity import PlayerIdentityManager, ROLE_RONALDINHO, ROLE_MADRID, ROLE_BARCELONA, ROLE_REFEREE, ROLE_BALL
from ai.ball_tracking import BallTracker
from ai.mesh_reconstruction import SMPLReconstructor
from ai.camera_estimation import CameraEstimator
from ai.motion_metrics import MotionMetricsEngine

def run_pipeline(
    video_path="assets/videos/ronaldinho_bernabeu_2005.mp4",
    output_base="data",
    max_frames=1800,
    fps=30
):
    print("=" * 70)
    print("MOTIONVISION TD — RONALDINHO: THE ART OF MOTION")
    print("Master AI Inference & 3D Reconstruction Pipeline")
    print(f"Input Video: {video_path}")
    print(f"Max Frames: {max_frames} (target @ {fps} fps)")
    print("=" * 70)

    if not os.path.exists(video_path):
        fallback = "assets/videos/synthetic_ronaldinho_test.mp4"
        if os.path.exists(fallback):
            print(f"[PIPELINE WARNING] {video_path} not found. Using validated test clip: {fallback}")
            video_path = fallback
        else:
            raise FileNotFoundError(f"Neither {video_path} nor {fallback} exists. Run create_test_footage.py first.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file: {video_path}")

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[PIPELINE] Video loaded: {width}x{height} @ {video_fps:.1f} fps ({total_video_frames} frames available)")

    frames_to_process = min(max_frames, total_video_frames)

    # Initialize modules
    detector = FootballDetector()
    tracker = MultiObjectTracker()
    identity_mgr = PlayerIdentityManager()
    ball_tracker = BallTracker()
    camera_estimator = CameraEstimator(width=width, height=height)
    biomech_engine = MotionMetricsEngine(fps=fps)
    smpl_reconstructor = SMPLReconstructor()

    tracking_results = []
    camera_results = []
    frame_idx = 0

    print("[PIPELINE] Starting detection and multi-person tracking loop...")

    while frame_idx < frames_to_process:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = round(frame_idx / float(fps), 3)

        # 1. Detection
        detections = detector.detect_frame(frame, frame_idx)

        # Separate ball detection from player detections
        ball_det = next((d for d in detections if d['class_name'] == 'sports ball'), None)
        player_dets = [d for d in detections if d['class_name'] == 'person']

        # 2. Register Ball
        ball_tracker.register_detection(frame_idx, timestamp, ball_det)

        # 3. Update Multi-Player Tracker
        active_tracks = tracker.update(player_dets)

        # 4. Classify Player Identities
        current_frame_players = []
        ball_pos_approx = ball_det['center'] if ball_det else [width * 0.5, height * 0.5]

        for trk in active_tracks:
            role, conf = identity_mgr.classify_track(trk, frame_idx, ball_pos=ball_pos_approx)
            current_frame_players.append({
                'track_id': trk.track_id,
                'role': role,
                'bbox': trk.bbox,
                'center': [(trk.bbox[0] + trk.bbox[2]) / 2.0, (trk.bbox[1] + trk.bbox[3]) / 2.0],
                'confidence': round(conf, 3),
                'dominant_color': trk.dominant_color
            })

        frame_record = {
            'frame_idx': frame_idx,
            'timestamp': timestamp,
            'players': current_frame_players
        }
        tracking_results.append(frame_record)

        # 5. Estimate Camera
        cam_rec = camera_estimator.estimate_camera_frame(frame_idx, timestamp, current_frame_players)
        camera_results.append(cam_rec)

        frame_idx += 1
        if frame_idx % 300 == 0 or frame_idx == frames_to_process:
            print(f"  Processed frame {frame_idx}/{frames_to_process} ({frame_idx/frames_to_process*100:.1f}%)")

    cap.release()

    # 6. Finalize Ball Trajectory
    print("[PIPELINE] Finalizing ball trajectory with occlusion recovery...")
    ball_trajectory = ball_tracker.finalize_trajectory()

    # 7. SMPL 3D Reconstruction
    print("[PIPELINE] Reconstructing 3D articulated SMPL motion...")
    motion_dataset = smpl_reconstructor.reconstruct_tracks(video_path, tracking_results)

    # 8. Biomechanics & Tactical Interactions
    print("[PIPELINE] Calculating biomechanics and tactical analytics...")
    ron_track_id = None
    for pid, pdata in motion_dataset['people'].items():
        if pdata['role'] == ROLE_RONALDINHO:
            ron_track_id = pid
            break
    if ron_track_id is None and len(motion_dataset['people']) > 0:
        ron_track_id = list(motion_dataset['people'].keys())[0]

    biomech_data = []
    if ron_track_id is not None:
        ron_pdata = motion_dataset['people'][ron_track_id]
        frames_records = []
        for i in range(len(ron_pdata['frames'])):
            frames_records.append({
                'frame_idx': ron_pdata['frames'][i],
                'timestamp': ron_pdata['timestamps'][i],
                'joints_3d': ron_pdata['joints_3d'][i],
                'root_trans': ron_pdata['root_trans'][i]
            })
        biomech_data = biomech_engine.compute_player_biomechanics(frames_records)

    # 9. Save all artifacts to disk
    os.makedirs(os.path.join(output_base, "tracking"), exist_ok=True)
    os.makedirs(os.path.join(output_base, "poses"), exist_ok=True)
    os.makedirs(os.path.join(output_base, "metadata"), exist_ok=True)

    tracking_file = os.path.join(output_base, "tracking", "tracking_data.json")
    with open(tracking_file, "w") as f:
        json.dump({'frames': tracking_results}, f, indent=2)

    ball_file = os.path.join(output_base, "tracking", "ball_trajectory.json")
    with open(ball_file, "w") as f:
        json.dump({'trajectory': ball_trajectory}, f, indent=2)

    camera_file = os.path.join(output_base, "metadata", "camera_data.json")
    with open(camera_file, "w") as f:
        json.dump({'camera_frames': camera_results}, f, indent=2)

    biomech_file = os.path.join(output_base, "metadata", "biomechanics.json")
    with open(biomech_file, "w") as f:
        json.dump({'ronaldinho_biomechanics': biomech_data}, f, indent=2)

    # Save NPZ for TouchDesigner
    npz_file = os.path.join(output_base, "poses", "motion_data.npz")
    summary_file = os.path.join(output_base, "metadata", "scene_manifest.json")
    smpl_reconstructor.save_motion_data(motion_dataset, npz_file, summary_file)

    print("=" * 70)
    print("AI MOTION EXPORT COMPLETE!")
    print(f"Tracking Data: {tracking_file}")
    print(f"Ball Trajectory: {ball_file}")
    print(f"Camera Parameters: {camera_file}")
    print(f"Biomechanics Metrics: {biomech_file}")
    print(f"SMPL Motion NPZ: {npz_file}")
    print("=" * 70)
    return {
        'total_frames': frames_to_process,
        'tracking_file': tracking_file,
        'ball_file': ball_file,
        'camera_file': camera_file,
        'biomech_file': biomech_file,
        'npz_file': npz_file
    }

if __name__ == "__main__":
    vid = sys.argv[1] if len(sys.argv) > 1 else "assets/videos/ronaldinho_bernabeu_2005.mp4"
    frames = int(sys.argv[2]) if len(sys.argv) > 2 else 1800
    run_pipeline(video_path=vid, max_frames=frames)
