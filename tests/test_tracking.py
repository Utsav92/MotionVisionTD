"""
Tests for Multi-Object Tracking & Player Identity Management
"""

import os
import sys
import pytest
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ai.tracking import MultiObjectTracker, Track, compute_iou
from ai.player_identity import PlayerIdentityManager, ROLE_RONALDINHO, ROLE_MADRID, ROLE_BARCELONA, ROLE_REFEREE

def test_compute_iou():
    box1 = [100, 100, 200, 200]
    box2 = [100, 100, 200, 200]
    assert compute_iou(box1, box2) == pytest.approx(1.0)

    box3 = [150, 150, 250, 250]
    iou = compute_iou(box1, box3)
    assert 0.14 < iou < 0.15

    box_disjoint = [300, 300, 400, 400]
    assert compute_iou(box1, box_disjoint) == 0.0

def test_tracker_lifecycle_and_velocity():
    tracker = MultiObjectTracker(max_lost_frames=5)
    
    # Frame 1: Initial detection
    det1 = [{'bbox': [100, 100, 150, 200], 'class_name': 'person', 'confidence': 0.9, 'dominant_color': [240, 240, 240]}]
    active = tracker.update(det1)
    assert len(active) == 1
    t1 = active[0]
    assert t1.track_id == 1

    # Frame 2: Moving right by 10 pixels
    det2 = [{'bbox': [110, 100, 160, 200], 'class_name': 'person', 'confidence': 0.9, 'dominant_color': [240, 240, 240]}]
    active2 = tracker.update(det2)
    assert len(active2) == 1
    assert active2[0].track_id == 1
    assert active2[0].velocity[0] > 0 # Positive X velocity

def test_player_identity_classification():
    mgr = PlayerIdentityManager(overrides_path="config/test_overrides.json")

    # Real Madrid jersey: bright white
    track_madrid = Track(1, [100, 100, 150, 200], 'person', 0.95, [245, 245, 245])
    role_m, conf_m = mgr.classify_track(track_madrid)
    assert role_m == ROLE_MADRID

    # Referee: dark uniform
    track_ref = Track(2, [100, 100, 150, 200], 'person', 0.90, [30, 20, 35])
    role_r, conf_r = mgr.classify_track(track_ref)
    assert role_r == ROLE_REFEREE

    # Ronaldinho: hero ball carrier
    track_ron = Track(3, [500, 500, 550, 600], 'person', 0.95, [140, 25, 35])
    role_ron, conf_ron = mgr.classify_track(track_ron, ball_pos=[520, 550])
    assert role_ron == ROLE_RONALDINHO

    # Manual confirmation override
    mgr.confirm_identity(99, ROLE_RONALDINHO)
    track_override = Track(99, [200, 200, 250, 300], 'person', 0.8, [100, 100, 100])
    role_ov, _ = mgr.classify_track(track_override)
    assert role_ov == ROLE_RONALDINHO

    if os.path.exists("config/test_overrides.json"):
        os.remove("config/test_overrides.json")
