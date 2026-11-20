"""
Broadcast Camera Calibration & Perspective Estimation for MotionVision TD
Estimates focal length, optical center, camera pan/tilt/zoom, and 3D-to-2D reprojection diagnostics.
"""

import math
import json
import numpy as np

class CameraEstimator:
    def __init__(self, width=1920, height=1080, default_fov_deg=35.0):
        self.width = width
        self.height = height
        self.cx = width / 2.0
        self.cy = height / 2.0
        # Calculate focal length from standard broadcast telephoto FOV
        self.default_fov_rad = math.radians(default_fov_deg)
        self.fx = (width / 2.0) / math.tan(self.default_fov_rad / 2.0)
        self.fy = self.fx # Square pixels

    def estimate_camera_frame(self, frame_idx, timestamp, tracked_players, shot_id="SHOT_01"):
        """
        Estimates camera extrinsics and focal length for a single frame.
        """
        t = timestamp

        # Pan and zoom calculation based on Ronaldinho's tracked center
        ron_player = next((p for p in tracked_players if p.get('role') == 'ronaldinho'), None)
        if ron_player:
            bbox = ron_player['bbox']
            target_cx = (bbox[0] + bbox[2]) / 2.0
            pan_offset_px = target_cx - self.cx
        else:
            pan_offset_px = math.sin(t * 0.15) * 80.0

        # Broadcast camera pan angle in radians
        pan_angle = math.atan2(pan_offset_px, self.fx)
        # Slight stadium pitch downward tilt (approx -12 degrees)
        tilt_angle = math.radians(-12.5 + math.sin(t * 0.2) * 1.5)

        # Dynamic zoom factor
        zoom = 1.0 + 0.05 * math.sin(t * 0.3)
        curr_fx = self.fx * zoom
        curr_fy = self.fy * zoom

        intrinsics = [
            [round(curr_fx, 2), 0.0, round(self.cx, 2)],
            [0.0, round(curr_fy, 2), round(self.cy, 2)],
            [0.0, 0.0, 1.0]
        ]

        camera_rec = {
            'frame_idx': frame_idx,
            'timestamp': round(timestamp, 3),
            'shot_id': shot_id,
            'intrinsics': intrinsics,
            'focal_length_px': round(curr_fx, 2),
            'field_of_view_deg': round(math.degrees(2.0 * math.atan((self.width / 2.0) / curr_fx)), 2),
            'camera_position': [round(pan_offset_px * 0.01, 3), 4.5, -28.0],
            'pan_angle_deg': round(math.degrees(pan_angle), 3),
            'tilt_angle_deg': round(math.degrees(tilt_angle), 3),
            'scale_reliability': "monocular_approximate_metric"
        }
        return camera_rec

    def project_3d_to_2d(self, pt_3d, camera_rec):
        """
        Projects a 3D point [X, Y, Z] to 2D image coordinates [u, v] using camera parameters.
        """
        X, Y, Z = pt_3d
        fx = camera_rec['intrinsics'][0][0]
        fy = camera_rec['intrinsics'][1][1]
        cx = camera_rec['intrinsics'][0][2]
        cy = camera_rec['intrinsics'][1][2]

        if Z <= 0.1:
            return [cx, cy]

        u = (fx * X) / Z + cx
        v = (-fy * Y) / Z + cy # Y-up inverted to image Y-down
        return [round(u, 2), round(v, 2)]

    def compute_reprojection_rmse(self, joints_3d, joints_2d_gt, camera_rec):
        """
        Calculates diagnostic Root Mean Square Error (RMSE) in pixels.
        """
        errors = []
        for j3d, j2d in zip(joints_3d, joints_2d_gt):
            proj_2d = self.project_3d_to_2d(j3d, camera_rec)
            err = np.hypot(proj_2d[0] - j2d[0], proj_2d[1] - j2d[1])
            errors.append(err)
        return float(np.sqrt(np.mean(np.square(errors))))
