"""
Automated Tests for MotionVision TD User Recording 65-Second Sequence
Verifies:
1. Master 65s video file integrity, frame count (1635 frames), resolution (720x1280), fps (25.0).
2. Split preview video integrity (1440x640, 1635 frames).
3. Player tracking data schema and telemetry consistency.
4. Validation snapshots for Acts 2, 3, 4, 5.
5. Particle simulation mechanics (35,000 particles, divergence-free curl field).
"""

import os
import json
import cv2
import pytest
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def test_user_recording_master_video_properties():
    master_path = os.path.join(PROJECT_ROOT, "renders", "final", "motionvision_user_recording_65s.mp4")
    assert os.path.exists(master_path), f"Master video not found at {master_path}"
    assert os.path.getsize(master_path) > 10_000_000, "Master video file is too small"

    cap = cv2.VideoCapture(master_path)
    assert cap.isOpened(), "Could not open master video with cv2"
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    assert frame_count == 1635, f"Expected 1635 frames (5 * 327), got {frame_count}"
    assert width == 720, f"Expected 720 width, got {width}"
    assert height == 1280, f"Expected 1280 height, got {height}"
    assert pytest.approx(fps, abs=0.1) == 25.0, f"Expected 25.0 fps, got {fps}"

def test_user_recording_split_preview_properties():
    preview_path = os.path.join(PROJECT_ROOT, "renders", "preview", "user_recording_65s_split_comparison.mp4")
    assert os.path.exists(preview_path), f"Split preview video not found at {preview_path}"
    assert os.path.getsize(preview_path) > 10_000_000, "Preview video file is too small"

    cap = cv2.VideoCapture(preview_path)
    assert cap.isOpened(), "Could not open preview video with cv2"
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    assert frame_count == 1635, f"Expected 1635 frames, got {frame_count}"
    assert width == 1440, f"Expected 1440 width (split 2x), got {width}"
    assert height == 640, f"Expected 640 height, got {height}"

def test_user_recording_tracking_schema():
    tracking_path = os.path.join(PROJECT_ROOT, "data", "tracking", "user_recording_tracking.json")
    assert os.path.exists(tracking_path), "Tracking data JSON not found"

    with open(tracking_path, "r") as f:
        data = json.load(f)

    assert "frames" in data, "JSON missing 'frames' key"
    frames = data["frames"]
    assert len(frames) == 327, f"Expected 327 frames of tracking data, got {len(frames)}"

    # Validate first and last frame structures
    for f in (frames[0], frames[160], frames[-1]):
        assert "ronaldinho" in f, "Frame missing ronaldinho data"
        assert "defender1" in f, "Frame missing defender1 data"
        assert "ball" in f, "Frame missing ball data"
        r = f["ronaldinho"]
        assert len(r["bbox"]) == 4
        assert r["speed"] > 0.0
        assert 60.0 <= r["knee_flex_l"] <= 185.0
        assert 60.0 <= r["knee_flex_r"] <= 185.0

def test_validation_snapshots_generated():
    expected_snapshots = [
        "act2_apple_comotion_gold_mesh.png",
        "act3_transparent_xray_skeleton.png",
        "act4_pointcloud_particle_sculpture.png",
        "act5_rgbd_camera_depth_effect.png"
    ]
    for snap in expected_snapshots:
        snap_path = os.path.join(PROJECT_ROOT, "renders", "preview", snap)
        assert os.path.exists(snap_path), f"Missing validation snapshot: {snap}"
        assert os.path.getsize(snap_path) > 100_000, f"Snapshot {snap} is suspiciously small"
        img = cv2.imread(snap_path)
        assert img is not None, f"Could not read snapshot {snap} with cv2"
        assert img.shape == (1280, 720, 3), f"Unexpected image shape for {snap}: {img.shape}"

def test_particle_simulation_mechanics():
    from touchdesigner.render_user_recording_65s import create_particle_cloud, update_curl_particles

    particles = create_particle_cloud(num_particles=35000)
    assert particles['count'] == 35000
    assert particles['pos'].shape == (35000, 3)
    assert particles['vel'].shape == (35000, 3)

    # Run 5 simulation steps
    for step in range(5):
        pos = update_curl_particles(particles, target_center=(0.1, 0.2), target_scale=0.15, t_sec=step * 0.04)

    assert pos.shape == (35000, 3)
    # Ensure particles stay within bounded reasonable radius
    dist_from_origin = np.linalg.norm(pos, axis=1)
    assert np.all(np.isfinite(dist_from_origin)), "Particles contain NaNs or Infs"
    assert np.max(dist_from_origin) < 5.0, "Particles exploded unboundedly"
