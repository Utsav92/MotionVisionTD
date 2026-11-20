"""
Tests for Sequence 2: 60-Second 5-Act Cinematic Sequence
"""

import os
import sys
import json
import cv2
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def test_sequence2_storyboard_manifest():
    scenes_path = "cinematics/scenes_sequence2.json"
    assert os.path.exists(scenes_path)

    with open(scenes_path, "r") as f:
        data = json.load(f)

    assert data["total_duration_sec"] == 60
    assert data["total_frames"] == 1800
    assert data["fps"] == 30
    assert len(data["acts"]) == 5

    acts = data["acts"]
    # Act 1: 0 - 10s
    assert acts[0]["start_sec"] == 0.0 and acts[0]["end_sec"] == 10.0
    assert acts[0]["start_frame"] == 0 and acts[0]["end_frame"] == 300
    # Act 2: 10 - 25s
    assert acts[1]["start_sec"] == 10.0 and acts[1]["end_sec"] == 25.0
    assert acts[1]["start_frame"] == 300 and acts[1]["end_frame"] == 750
    # Act 3: 25 - 40s
    assert acts[2]["start_sec"] == 25.0 and acts[2]["end_sec"] == 40.0
    assert acts[2]["start_frame"] == 750 and acts[2]["end_frame"] == 1200
    # Act 4: 40 - 50s
    assert acts[3]["start_sec"] == 40.0 and acts[3]["end_sec"] == 50.0
    assert acts[3]["start_frame"] == 1200 and acts[3]["end_frame"] == 1500
    # Act 5: 50 - 60s
    assert acts[4]["start_sec"] == 50.0 and acts[4]["end_sec"] == 60.0
    assert acts[4]["start_frame"] == 1500 and acts[4]["end_frame"] == 1800

def test_sequence2_timeline_keyframes():
    tl_path = "cinematics/timeline_sequence2.json"
    assert os.path.exists(tl_path)
    with open(tl_path, "r") as f:
        data = json.load(f)
    keyframes = data["keyframes"]
    assert len(keyframes) >= 5
    # Verify initial and terminal keyframes
    assert keyframes[0]["frame"] == 0
    assert keyframes[-1]["frame"] == 1800

def test_sequence2_rendered_video_integrity():
    vid_path = "renders/final/motionvision_ronaldinho_sequence2_60s.mp4"
    if os.path.exists(vid_path):
        cap = cv2.VideoCapture(vid_path)
        assert cap.isOpened()
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        assert w == 1920
        assert h == 1080
        assert fps == pytest.approx(30.0, abs=1.0)
        assert count == 1800
