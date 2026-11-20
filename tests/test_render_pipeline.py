"""
Tests for Rendering Pipeline, Video Integrity & Exported Artifacts
"""

import os
import sys
import cv2
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def test_source_video_integrity():
    vid_path = "assets/videos/ronaldinho_bernabeu_2005.mp4"
    assert os.path.exists(vid_path), f"Video missing: {vid_path}"

    cap = cv2.VideoCapture(vid_path)
    assert cap.isOpened(), "Could not open video file"
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    assert w == 1920
    assert h == 1080
    assert fps == pytest.approx(30.0, abs=1.0)
    assert count >= 300

def test_validation_frames_and_rendered_outputs():
    # Check that validation frame images exist
    preview_dir = "renders/preview"
    assert os.path.isdir(preview_dir)

    val_images = [f for f in os.listdir(preview_dir) if f.startswith("validation_") and f.endswith(".png")]
    assert len(val_images) > 0, "No validation images found in renders/preview"

    for img_name in val_images:
        img_p = os.path.join(preview_dir, img_name)
        mat = cv2.imread(img_p)
        assert mat is not None, f"Corrupted image: {img_p}"
        assert mat.shape[0] == 1080
        assert mat.shape[1] == 1920

def test_master_rendered_video_playable():
    video_path = "renders/final/motionvision_ronaldinho_60s.mp4"
    if not os.path.exists(video_path):
        video_path = "renders/preview/motionvision_ronaldinho_15s_social.mp4"

    if os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
        assert cap.isOpened(), f"Cannot open rendered video: {video_path}"
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        assert w == 1920
        assert h == 1080
        assert frame_count > 0
