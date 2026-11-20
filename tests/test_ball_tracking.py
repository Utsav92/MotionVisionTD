"""
Tests for Ball Tracking & Occlusion Interpolation
"""

import os
import sys
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ai.ball_tracking import BallTracker

def test_ball_occlusion_interpolation():
    tracker = BallTracker(max_interpolation_gap=10)

    # Frame 0: detected at (100, 100)
    tracker.register_detection(0, 0.0, {'center': [100.0, 100.0], 'confidence': 0.95})

    # Frames 1-3: occluded by player legs
    for f in range(1, 4):
        tracker.register_detection(f, f / 30.0, None)

    # Frame 4: detected at (140, 120)
    tracker.register_detection(4, 4 / 30.0, {'center': [140.0, 120.0], 'confidence': 0.90})

    trajectory = tracker.finalize_trajectory()
    assert len(trajectory) == 5

    # Check key frame 0
    assert trajectory[0]['detected'] is True
    assert trajectory[0]['interpolated'] is False
    assert trajectory[0]['pos_2d'] == [100.0, 100.0]

    # Check interpolated frame 2 (mid-gap)
    assert trajectory[2]['detected'] is False
    assert trajectory[2]['interpolated'] is True
    assert trajectory[2]['pos_2d'][0] == pytest.approx(120.0)
    assert trajectory[2]['pos_2d'][1] == pytest.approx(110.0)
    assert trajectory[2]['confidence'] < 0.90 # Uncertainty penalty applied

    # Check key frame 4
    assert trajectory[4]['detected'] is True
    assert trajectory[4]['interpolated'] is False
