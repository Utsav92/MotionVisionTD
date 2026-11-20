"""
Tests for 3D Geometry, SMPL Kinematic Hierarchy & Biomechanical Metrics
"""

import os
import sys
import pytest
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ai.mesh_reconstruction import SMPLReconstructor, SMPL_JOINT_NAMES, SMPL_PARENTS
from ai.camera_estimation import CameraEstimator
from ai.motion_metrics import calculate_angle_3d, MotionMetricsEngine

def test_smpl_hierarchy():
    assert len(SMPL_JOINT_NAMES) == 24
    assert len(SMPL_PARENTS) == 24
    assert SMPL_JOINT_NAMES[0] == "Pelvis"
    assert SMPL_PARENTS[0] == -1 # Root joint has no parent

    # Verify children have valid parent indices
    for i in range(1, 24):
        assert 0 <= SMPL_PARENTS[i] < i

def test_template_mesh_generation(tmp_path):
    test_obj = os.path.join(tmp_path, "test_template.obj")
    reconstructor = SMPLReconstructor(template_obj_path=test_obj)
    assert os.path.exists(test_obj)

    with open(test_obj, "r") as f:
        lines = f.readlines()
    v_lines = [l for l in lines if l.startswith("v ")]
    f_lines = [l for l in lines if l.startswith("f ")]
    assert len(v_lines) > 500
    assert len(f_lines) > 1000

def test_camera_projection():
    estimator = CameraEstimator(width=1920, height=1080, default_fov_deg=35.0)
    cam_rec = estimator.estimate_camera_frame(0, 0.0, [])

    # Point along optical axis [0, 0, 10.0] should project to optical center (960, 540)
    pt_center = [0.0, 0.0, 10.0]
    proj = estimator.project_3d_to_2d(pt_center, cam_rec)
    assert proj[0] == pytest.approx(960.0, abs=1.0)
    assert proj[1] == pytest.approx(540.0, abs=1.0)

def test_biomechanics_angle_calculation():
    # 90-degree right angle (e.g. Hip at (0, 1, 0), Knee at (0, 0, 0), Ankle at (1, 0, 0))
    p_hip = [0.0, 1.0, 0.0]
    p_knee = [0.0, 0.0, 0.0]
    p_ankle = [1.0, 0.0, 0.0]
    angle = calculate_angle_3d(p_hip, p_knee, p_ankle)
    assert angle == pytest.approx(90.0, abs=0.1)

    # 180-degree straight leg
    p_ankle_straight = [0.0, -1.0, 0.0]
    angle_straight = calculate_angle_3d(p_hip, p_knee, p_ankle_straight)
    assert angle_straight == pytest.approx(180.0, abs=0.1)
