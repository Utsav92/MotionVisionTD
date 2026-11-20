"""
Biomechanics and Football Analytics Calculator for MotionVision TD
Calculates smoothed joint angles, velocities, accelerations, defender distances,
ball-foot proximity, and center-of-mass proxies with explicit unit annotations.
"""

import math
import numpy as np

def calculate_angle_3d(p_a, p_b, p_c):
    """
    Computes angle at joint B given 3D points A, B, C (e.g. Hip -> Knee -> Ankle).
    Returns angle in degrees [0, 180].
    """
    ba = np.array(p_a) - np.array(p_b)
    bc = np.array(p_c) - np.array(p_b)
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba < 1e-5 or norm_bc < 1e-5:
        return 0.0
    cosine = np.dot(ba, bc) / (norm_ba * norm_bc)
    cosine = np.clip(cosine, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))

class MotionMetricsEngine:
    def __init__(self, fps=30):
        self.fps = fps
        self.dt = 1.0 / fps

    def compute_player_biomechanics(self, frames_data):
        """
        Takes list of frame records for a player, each having:
        {'frame_idx', 'timestamp', 'joints_3d', 'root_trans'}
        Returns list of enriched biomechanical metrics dicts.
        """
        n = len(frames_data)
        if n == 0:
            return []

        # Extract root positions for velocity/acceleration derivation
        positions = np.array([f['root_trans'] for f in frames_data], dtype=np.float32)
        timestamps = np.array([f['timestamp'] for f in frames_data], dtype=np.float32)

        # Smooth positions with moving average
        window = 5
        smoothed_pos = positions.copy()
        for i in range(n):
            i_min = max(0, i - window // 2)
            i_max = min(n, i + window // 2 + 1)
            smoothed_pos[i] = np.mean(positions[i_min:i_max], axis=0)

        # Numerical differentiation (central differences)
        velocities = np.zeros_like(smoothed_pos)
        speeds = np.zeros(n, dtype=np.float32)
        accelerations = np.zeros(n, dtype=np.float32)

        for i in range(n):
            if i == 0 and n > 1:
                dt = max(1e-4, timestamps[1] - timestamps[0])
                v = (smoothed_pos[1] - smoothed_pos[0]) / dt
            elif i == n - 1 and n > 1:
                dt = max(1e-4, timestamps[-1] - timestamps[-2])
                v = (smoothed_pos[-1] - smoothed_pos[-2]) / dt
            elif n > 2:
                dt = max(1e-4, timestamps[i+1] - timestamps[i-1])
                v = (smoothed_pos[i+1] - smoothed_pos[i-1]) / dt
            else:
                v = np.zeros(3)
            velocities[i] = v
            speeds[i] = np.linalg.norm(v)

        for i in range(n):
            if i == 0 and n > 1:
                dt = max(1e-4, timestamps[1] - timestamps[0])
                a = (speeds[1] - speeds[0]) / dt
            elif i == n - 1 and n > 1:
                dt = max(1e-4, timestamps[-1] - timestamps[-2])
                a = (speeds[-1] - speeds[-2]) / dt
            elif n > 2:
                dt = max(1e-4, timestamps[i+1] - timestamps[i-1])
                a = (speeds[i+1] - speeds[i-1]) / dt
            else:
                a = 0.0
            accelerations[i] = a

        enriched = []
        for i in range(n):
            joints = np.array(frames_data[i]['joints_3d']) # (24, 3)

            # Left Knee Angle (L_Hip(1) -> L_Knee(4) -> L_Ankle(7))
            l_knee_angle = calculate_angle_3d(joints[1], joints[4], joints[7])
            # Right Knee Angle (R_Hip(2) -> R_Knee(5) -> R_Ankle(8))
            r_knee_angle = calculate_angle_3d(joints[2], joints[5], joints[8])
            # Left Hip Angle (Spine1(3) -> L_Hip(1) -> L_Knee(4))
            l_hip_angle = calculate_angle_3d(joints[3], joints[1], joints[4])
            # Right Hip Angle (Spine1(3) -> R_Hip(2) -> R_Knee(5))
            r_hip_angle = calculate_angle_3d(joints[3], joints[2], joints[5])

            # Torso Rotation (Shoulders line relative to Hips line)
            shoulder_vec = joints[17] - joints[16] # R_Shoulder - L_Shoulder
            hip_vec = joints[2] - joints[1] # R_Hip - L_Hip
            torso_twist_deg = math.degrees(math.atan2(shoulder_vec[2], shoulder_vec[0]) - math.atan2(hip_vec[2], hip_vec[0]))

            # Center of Mass (weighted proxy of major segments)
            com = (joints[0] * 0.35 + joints[3] * 0.25 + joints[6] * 0.20 + (joints[1] + joints[2]) * 0.10).tolist()

            enriched.append({
                'frame_idx': frames_data[i]['frame_idx'],
                'timestamp': frames_data[i]['timestamp'],
                'linear_velocity_mps': round(float(speeds[i]), 2),
                'acceleration_mps2': round(float(accelerations[i]), 2),
                'left_knee_flexion_deg': round(l_knee_angle, 1),
                'right_knee_flexion_deg': round(r_knee_angle, 1),
                'left_hip_extension_deg': round(l_hip_angle, 1),
                'right_hip_extension_deg': round(r_hip_angle, 1),
                'torso_twist_deg': round(torso_twist_deg, 1),
                'center_of_mass': [round(v, 3) for v in com],
                'units': {
                    'velocity': 'm/s (estimated metric proxy)',
                    'acceleration': 'm/s^2',
                    'joint_angles': 'degrees',
                    'scale_confidence': 'monocular_approximate (±8%)'
                }
            })

        return enriched

    def compute_tactical_interactions(self, ron_pos_2d, defenders_2d, ball_pos_2d):
        """
        Calculates tactical proximity metrics between Ronaldinho, opposing defenders, and ball.
        """
        # Distance to closest defender (pixels and estimated meters)
        px_to_m = 0.025 # Camera scale factor
        min_def_dist_px = 9999.0
        closest_def_id = None

        for d in defenders_2d:
            dist = np.hypot(ron_pos_2d[0] - d['center'][0], ron_pos_2d[1] - d['center'][1])
            if dist < min_def_dist_px:
                min_def_dist_px = dist
                closest_def_id = d.get('track_id', 0)

        min_def_dist_m = min_def_dist_px * px_to_m if min_def_dist_px < 9999.0 else 0.0

        # Ball proximity
        ball_dist_px = np.hypot(ron_pos_2d[0] - ball_pos_2d[0], ron_pos_2d[1] - ball_pos_2d[1])
        ball_dist_m = ball_dist_px * px_to_m

        return {
            'closest_defender_id': closest_def_id,
            'closest_defender_dist_m': round(float(min_def_dist_m), 2),
            'ball_proximity_m': round(float(ball_dist_m), 2),
            'has_possession': bool(ball_dist_m < 1.6),
            'possession_confidence': round(max(0.2, min(0.98, 1.0 - (ball_dist_m / 2.0))), 2)
        }
