"""
Player Identity & Role Classification Module for MotionVision TD
Categorizes tracked players into Ronaldinho (Gold), Real Madrid (Red/Pink),
Barcelona (Blue), and Referee (Purple), with manual confirmation and override tools.
"""

import json
import numpy as np

ROLE_RONALDINHO = "ronaldinho"
ROLE_MADRID = "real_madrid"
ROLE_BARCELONA = "barcelona"
ROLE_REFEREE = "referee"
ROLE_BALL = "ball"

# Defined visual theme colors RGBA
ROLE_COLORS = {
    ROLE_RONALDINHO: [1.0, 0.843, 0.0, 1.0],      # Electric Gold (#FFD700)
    ROLE_MADRID: [1.0, 0.22, 0.38, 1.0],         # Neon Crimson/Pink (#FF3860)
    ROLE_BARCELONA: [0.0, 0.48, 1.0, 1.0],        # Electric Blue (#007BFF)
    ROLE_REFEREE: [0.65, 0.25, 0.95, 1.0],        # Referee Purple (#A640F3)
    ROLE_BALL: [1.0, 1.0, 1.0, 1.0]               # Pure White (#FFFFFF)
}

class PlayerIdentityManager:
    def __init__(self, overrides_path="config/identity_overrides.json"):
        self.overrides_path = overrides_path
        self.manual_overrides = {} # {track_id: role_string}
        self.track_stats = {} # {track_id: {'role': str, 'confidence': float, 'appearances': int, 'avg_color': []}}
        self._load_overrides()

    def _load_overrides(self):
        try:
            with open(self.overrides_path, "r") as f:
                self.manual_overrides = json.load(f)
                print(f"[IDENTITY] Loaded {len(self.manual_overrides)} manual identity overrides from {self.overrides_path}")
        except (FileNotFoundError, json.JSONDecodeError):
            self.manual_overrides = {}

    def save_overrides(self):
        with open(self.overrides_path, "w") as f:
            json.dump(self.manual_overrides, f, indent=2)
        print(f"[IDENTITY] Saved manual overrides to {self.overrides_path}")

    def confirm_identity(self, track_id, role, start_frame=0, end_frame=None):
        """
        Manually assign or confirm a player's identity.
        """
        self.manual_overrides[str(track_id)] = {
            "role": role,
            "start_frame": start_frame,
            "end_frame": end_frame
        }
        self.save_overrides()

    def classify_track(self, track, frame_idx=0, ball_pos=None):
        """
        Classify a Track object into its tactical/visual role.
        """
        track_id_str = str(track.track_id)

        # Check manual override first
        if track_id_str in self.manual_overrides:
            override = self.manual_overrides[track_id_str]
            if isinstance(override, dict):
                start = override.get("start_frame", 0)
                end = override.get("end_frame", None)
                if frame_idx >= start and (end is None or frame_idx <= end):
                    return override["role"], 1.0
            elif isinstance(override, str):
                return override, 1.0

        if track.cls_name == "sports ball":
            return ROLE_BALL, track.confidence

        # Analyze RGB dominant color
        r, g, b = track.dominant_color
        total_intensity = r + g + b
        brightness = total_intensity / 3.0

        # Referee detection: very low brightness (dark kit) or purplish
        if brightness < 65 or (r > 30 and b > 40 and g < 30):
            role = ROLE_REFEREE
            confidence = 0.85
        # Real Madrid: high brightness (white jersey, high R, G, B with low color saturation)
        elif brightness > 175 and abs(r - g) < 35 and abs(g - b) < 35:
            role = ROLE_MADRID
            confidence = 0.90
        # FC Barcelona (Ronaldinho vs teammates)
        else:
            # Check proximity to ball and velocity characteristics
            # Ronaldinho is typically close to the ball, moving with high acceleration
            is_ball_carrier = False
            if ball_pos is not None:
                cx = (track.bbox[0] + track.bbox[2]) / 2.0
                cy = (track.bbox[1] + track.bbox[3]) / 2.0
                dist_to_ball = np.hypot(cx - ball_pos[0], cy - ball_pos[1])
                if dist_to_ball < 85.0:
                    is_ball_carrier = True

            # If track was previously confirmed as Ronaldinho or has hero characteristics
            if track.track_id in self.track_stats and self.track_stats[track.track_id]['role'] == ROLE_RONALDINHO:
                role = ROLE_RONALDINHO
                confidence = 0.94
            elif is_ball_carrier or (track.track_id == 1 and brightness < 170):
                role = ROLE_RONALDINHO
                confidence = 0.92
            else:
                role = ROLE_BARCELONA
                confidence = 0.88

        # Cache stats
        if track.track_id not in self.track_stats:
            self.track_stats[track.track_id] = {
                'role': role,
                'confidence': confidence,
                'appearances': 1,
                'dominant_color': track.dominant_color
            }
        else:
            self.track_stats[track.track_id]['appearances'] += 1

        return role, confidence
