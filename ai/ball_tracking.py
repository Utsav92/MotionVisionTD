"""
Football Tracking & Trajectory Synthesis Module for MotionVision TD
Specialized tracking for small fast-moving football with occlusion recovery and interpolation flags.
"""

import math
import numpy as np

class BallTracker:
    def __init__(self, max_interpolation_gap=25, smoothing_window=5):
        self.max_gap = max_interpolation_gap
        self.smoothing_window = smoothing_window
        # Raw records: list of dicts {frame_idx, timestamp, pos: [x,y], confidence, detected: bool}
        self.records = []
        # Trajectory history for fast lookup
        self.smoothed_trajectory = {} # frame_idx -> dict

    def register_detection(self, frame_idx, timestamp, ball_detection=None):
        """
        Registers a detection for a given frame.
        ball_detection can be dict with 'center', 'confidence' or None if missing.
        """
        if ball_detection is not None:
            pos = [float(ball_detection['center'][0]), float(ball_detection['center'][1])]
            conf = float(ball_detection.get('confidence', 0.9))
            rec = {
                'frame_idx': frame_idx,
                'timestamp': timestamp,
                'pos': pos,
                'confidence': conf,
                'detected': True,
                'interpolated': False
            }
        else:
            rec = {
                'frame_idx': frame_idx,
                'timestamp': timestamp,
                'pos': None,
                'confidence': 0.0,
                'detected': False,
                'interpolated': False
            }
        self.records.append(rec)

    def finalize_trajectory(self):
        """
        Runs occlusion filling and trajectory smoothing across the entire recorded sequence.
        Marks interpolated frames explicitly.
        """
        n = len(self.records)
        if n == 0:
            return {}

        # Step 1: Linear / spline interpolation for missing gaps
        filled_positions = []
        for i in range(n):
            if self.records[i]['detected']:
                filled_positions.append((i, self.records[i]['pos'], self.records[i]['confidence']))

        if len(filled_positions) == 0:
            # Fallback: create default trajectory if ball was never detected
            return {}

        # Interpolate between detected keypoints
        final_dict = {}
        for k in range(len(filled_positions) - 1):
            f_start, p_start, c_start = filled_positions[k]
            f_end, p_end, c_end = filled_positions[k + 1]
            gap = f_end - f_start

            # Register start point
            final_dict[f_start] = {
                'frame_idx': f_start,
                'timestamp': self.records[f_start]['timestamp'],
                'pos_2d': p_start,
                'confidence': c_start,
                'detected': True,
                'interpolated': False
            }

            if 1 < gap <= self.max_gap:
                for f in range(f_start + 1, f_end):
                    alpha = (f - f_start) / float(gap)
                    interp_x = (1 - alpha) * p_start[0] + alpha * p_end[0]
                    interp_y = (1 - alpha) * p_start[1] + alpha * p_end[1]
                    # Uncertainty increases mid-gap
                    uncertainty_factor = math.sin(alpha * math.pi)
                    conf = max(0.2, (1 - alpha) * c_start + alpha * c_end - 0.3 * uncertainty_factor)

                    final_dict[f] = {
                        'frame_idx': f,
                        'timestamp': self.records[f]['timestamp'],
                        'pos_2d': [interp_x, interp_y],
                        'confidence': round(conf, 3),
                        'detected': False,
                        'interpolated': True
                    }

        # Last point
        f_last, p_last, c_last = filled_positions[-1]
        final_dict[f_last] = {
            'frame_idx': f_last,
            'timestamp': self.records[f_last]['timestamp'],
            'pos_2d': p_last,
            'confidence': c_last,
            'detected': True,
            'interpolated': False
        }

        # Step 2: Temporal moving-average smoothing on continuous segments
        sorted_frames = sorted(final_dict.keys())
        for idx, f in enumerate(sorted_frames):
            window = [
                final_dict[sorted_frames[j]]['pos_2d']
                for j in range(max(0, idx - self.smoothing_window // 2),
                               min(len(sorted_frames), idx + self.smoothing_window // 2 + 1))
            ]
            avg_x = sum(p[0] for p in window) / len(window)
            avg_y = sum(p[1] for p in window) / len(window)
            final_dict[f]['smoothed_pos_2d'] = [round(avg_x, 2), round(avg_y, 2)]

        self.smoothed_trajectory = final_dict
        return final_dict
