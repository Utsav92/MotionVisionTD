"""
Tests for Frame Synchronization & Timeline Integrity
"""

import os
import sys
import json
import pytest
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def test_frame_index_synchronization():
    tracking_path = "data/tracking/tracking_data.json"
    ball_path = "data/tracking/ball_trajectory.json"
    camera_path = "data/metadata/camera_data.json"
    biomech_path = "data/metadata/biomechanics.json"
    npz_path = "data/poses/motion_data.npz"

    # All files must exist
    for p in (tracking_path, ball_path, camera_path, biomech_path, npz_path):
        assert os.path.exists(p), f"Missing data file: {p}"

    with open(tracking_path, "r") as f:
        t_data = json.load(f)['frames']
    with open(camera_path, "r") as f:
        c_data = json.load(f)['camera_frames']
    with open(biomech_path, "r") as f:
        b_data = json.load(f)['ronaldinho_biomechanics']

    # Frame count parity
    assert len(t_data) == len(c_data)
    assert len(b_data) == len(t_data)

    # Check timestamps across 50 randomly sampled frames
    fps = 30.0
    for idx in range(0, len(t_data), 35):
        expected_t = round(idx / fps, 3)
        assert t_data[idx]['frame_idx'] == idx
        assert t_data[idx]['timestamp'] == pytest.approx(expected_t, abs=0.01)
        assert c_data[idx]['frame_idx'] == idx
        assert c_data[idx]['timestamp'] == pytest.approx(expected_t, abs=0.01)
        assert b_data[idx]['frame_idx'] == idx
        assert b_data[idx]['timestamp'] == pytest.approx(expected_t, abs=0.01)

def test_npz_binary_data_consistency():
    npz_path = "data/poses/motion_data.npz"
    npz_data = np.load(npz_path)

    assert 'total_frames' in npz_data
    assert 'joint_names' in npz_data
    assert len(npz_data['joint_names']) == 24

    # Check person 1 arrays
    if 'person_1_root_trans' in npz_data:
        root_trans = npz_data['person_1_root_trans']
        joints_3d = npz_data['person_1_joints_3d']
        assert root_trans.ndim == 2
        assert root_trans.shape[1] == 3
        assert joints_3d.ndim == 3
        assert joints_3d.shape[1] == 24
        assert joints_3d.shape[2] == 3
