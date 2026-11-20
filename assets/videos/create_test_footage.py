"""
Synthetic Match Footage Generator for MotionVision TD
Generates broadcast-quality test footage simulating Ronaldinho's 2005 Bernabeu performance
with realistic players, jerseys, ball dynamics, pitch markings, and broadcast camera pans.
"""

import os
import sys
import math
import cv2
import numpy as np

def create_synthetic_match_clip(
    output_path="assets/videos/synthetic_ronaldinho_test.mp4",
    duration_sec=60,
    fps=30,
    width=1920,
    height=1080
):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    total_frames = int(duration_sec * fps)
    print(f"[TEST FOOTAGE] Generating {total_frames} frames ({duration_sec}s @ {fps}fps, {width}x{height}) -> {output_path}")

    # Use MP4V or avc1 codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not out.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Seed for deterministic generation
    np.random.seed(20051119)

    # Base pitch surface
    pitch_base = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        # Vertical gradient for grass depth
        v_factor = 0.65 + 0.35 * (y / height)
        # Horizontal subtle grass stripe pattern
        stripe = 1.0 + 0.05 * math.sin(y * 0.08)
        r = int(18 * v_factor * stripe)
        g = int(95 * v_factor * stripe)
        b = int(32 * v_factor * stripe)
        pitch_base[y, :] = [b, g, r]

    # Add stadium floodlight ambient vignette
    y_coords, x_coords = np.ogrid[:height, :width]
    center_y, center_x = height * 0.45, width * 0.5
    dist_from_center = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
    max_dist = math.sqrt(center_x**2 + center_y**2)
    vignette = 1.0 - 0.28 * (dist_from_center / max_dist)**1.5
    vignette = np.clip(vignette, 0.4, 1.0)[:, :, np.newaxis]
    pitch_base = np.clip(pitch_base * vignette, 0, 255).astype(np.uint8)

    # Camera motion simulation (panning with Ronaldinho)
    for frame_idx in range(total_frames):
        t = frame_idx / fps
        frame = pitch_base.copy()

        # Phase / Scene breakdown
        # Scene 1: 0 - 8s: Midfield drive
        # Scene 2: 8 - 18s: Confrontation & stepover setup
        # Scene 3: 18 - 30s: Explosive cut past Ramos
        # Scene 4: 30 - 40s: Penalty box penetration
        # Scene 5: 40 - 50s: Near-post strike
        # Scene 6: 50 - 60s: Standing ovation celebration

        # Virtual camera pan
        cam_x = math.sin(t * 0.15) * 80.0
        cam_zoom = 1.0 + 0.05 * math.sin(t * 0.3)

        # Field lines in perspective
        # Touchline
        cv2.line(frame, (0, int(height * 0.22)), (width, int(height * 0.22)), (230, 235, 230), 3, cv2.LINE_AA)
        # Penalty area line
        pen_x = int(width * 0.75 + cam_x * 0.5)
        cv2.line(frame, (pen_x, int(height * 0.22)), (pen_x, height), (220, 230, 220), 4, cv2.LINE_AA)
        # Penalty spot
        spot_x = int(width * 0.82 + cam_x * 0.5)
        spot_y = int(height * 0.65)
        cv2.circle(frame, (spot_x, spot_y), 5, (240, 240, 240), -1, cv2.LINE_AA)

        # Ronaldinho motion trajectory
        if t < 18.0:
            # Steady forward progression with slight rhythm
            ron_x = width * 0.25 + (t / 18.0) * (width * 0.20) + math.sin(t * 5.0) * 15.0
            ron_y = height * 0.62 + math.cos(t * 4.0) * 18.0
            action_desc = "DRIBBLE & STEPOVER SETUP"
        elif t < 30.0:
            # Explosive burst and cut inside
            sub_t = (t - 18.0) / 12.0
            ron_x = width * 0.45 + sub_t * (width * 0.22) + math.sin(sub_t * math.pi * 3) * 35.0
            ron_y = height * 0.62 - sub_t * 60.0 + math.cos(sub_t * math.pi * 2) * 20.0
            action_desc = "ACCELERATION & DIRECTION CHANGE"
        elif t < 40.0:
            # Penalty area entry
            sub_t = (t - 30.0) / 10.0
            ron_x = width * 0.67 + sub_t * (width * 0.12)
            ron_y = height * 0.56 + sub_t * 20.0 + math.sin(sub_t * 8.0) * 10.0
            action_desc = "PENALTY BOX PENETRATION"
        elif t < 50.0:
            # Shot execution & follow-through
            sub_t = (t - 40.0) / 10.0
            ron_x = width * 0.79 + sub_t * (width * 0.04)
            ron_y = height * 0.58 + math.sin(sub_t * math.pi) * 12.0
            action_desc = "GOAL STRIKE EXECUTION"
        else:
            # Celebration samba dance & Bernabeu ovation
            sub_t = (t - 50.0) / 10.0
            ron_x = width * 0.82 + math.sin(sub_t * 6.0) * 12.0
            ron_y = height * 0.58 + math.cos(sub_t * 8.0) * 8.0
            action_desc = "SAMBA CELEBRATION & STANDING OVATION"

        ron_x += cam_x
        ron_scale = 1.05 + 0.1 * ((ron_y - height * 0.5) / (height * 0.5))

        # Ball trajectory relative to Ronaldinho
        if t < 41.5:
            # Ball at Ronaldinho's right foot with rhythmic touches
            ball_offset_x = 26.0 * ron_scale + math.sin(t * 7.0) * (8.0 * ron_scale)
            ball_offset_y = 38.0 * ron_scale + math.cos(t * 6.0) * (5.0 * ron_scale)
            ball_x = ron_x + ball_offset_x
            ball_y = ron_y + ball_offset_y
        elif t < 45.0:
            # Ball shot towards goal (upper-near corner)
            sub_t = (t - 41.5) / 3.5
            ball_x = (ron_x + 30.0) + sub_t * (width * 0.94 - (ron_x + 30.0))
            ball_y = (ron_y + 35.0) - math.sin(sub_t * math.pi) * 90.0 - sub_t * 110.0
        else:
            # Ball in back of net vibrating
            ball_x = width * 0.94 + math.sin(t * 20.0) * 2.0
            ball_y = height * 0.42 + math.cos(t * 18.0) * 2.0

        # Draw Defender 1: Sergio Ramos (White Kit #4)
        if t < 35.0:
            ramos_x = ron_x + (85.0 - (t / 35.0) * 30.0)
            ramos_y = ron_y - 25.0 + math.sin(t * 3.5) * 15.0
            _draw_player(frame, int(ramos_x), int(ramos_y), scale=ron_scale * 0.98,
                         kit_color=(245, 245, 245), shorts_color=(235, 235, 235),
                         skin_color=(150, 190, 225), number="4", name="RAMOS", label_color=(200, 200, 255))

        # Draw Defender 2: Ivan Helguera (White Kit #6)
        if 15.0 < t < 48.0:
            helg_x = ron_x + 130.0 - ((t - 15.0) / 33.0) * 40.0
            helg_y = ron_y + 40.0 - math.cos(t * 2.5) * 20.0
            _draw_player(frame, int(helg_x), int(helg_y), scale=ron_scale * 0.97,
                         kit_color=(240, 240, 240), shorts_color=(230, 230, 230),
                         skin_color=(140, 185, 220), number="6", name="HELGUERA", label_color=(200, 200, 255))

        # Draw Teammate: Samuel Eto'o (Blaugrana #9)
        etoo_x = ron_x - 170.0 + (t / 60.0) * 50.0
        etoo_y = ron_y - 80.0 + math.sin(t * 2.0) * 12.0
        _draw_player(frame, int(etoo_x), int(etoo_y), scale=ron_scale * 0.95,
                     kit_color=(140, 30, 30), shorts_color=(120, 20, 20),
                     skin_color=(50, 75, 110), number="9", name="ETO'O", label_color=(255, 180, 100))

        # Draw Referee: Iturralde (Black/Purple kit)
        ref_x = ron_x - 240.0
        ref_y = ron_y + 110.0 + math.sin(t * 1.5) * 10.0
        _draw_player(frame, int(ref_x), int(ref_y), scale=ron_scale * 0.92,
                     kit_color=(40, 20, 50), shorts_color=(30, 15, 40),
                     skin_color=(145, 185, 225), number="REF", name="REFEREE", label_color=(220, 160, 255))

        # Draw Goalkeeper: Casillas (Green/Grey kit #1) near goal
        goal_gk_x = int(width * 0.92 + cam_x * 0.3)
        goal_gk_y = int(height * 0.44 + math.sin(t * 4.0) * 25.0)
        _draw_player(frame, goal_gk_x, goal_gk_y, scale=ron_scale * 0.93,
                     kit_color=(30, 110, 45), shorts_color=(25, 30, 35),
                     skin_color=(145, 185, 225), number="1", name="CASILLAS", label_color=(150, 255, 180))

        # Draw Ronaldinho (Iconic #10 - Blaugrana / Gold Trim)
        _draw_player(frame, int(ron_x), int(ron_y), scale=ron_scale,
                     kit_color=(145, 25, 35), shorts_color=(125, 20, 25),
                     skin_color=(60, 95, 140), number="10", name="RONALDINHO",
                     label_color=(0, 215, 255), is_hero=True, t=t)

        # Draw Football
        _draw_football(frame, int(ball_x), int(ball_y), scale=ron_scale)

        # Broadcast Graphics HUD
        _draw_broadcast_hud(frame, t, action_desc, width, height)

        out.write(frame)

        if (frame_idx + 1) % 300 == 0 or frame_idx == total_frames - 1:
            print(f"  Processed {frame_idx + 1}/{total_frames} frames ({((frame_idx + 1)/total_frames)*100:.1f}%)")

    out.release()
    print(f"[TEST FOOTAGE] Video generation complete: {output_path}")
    return output_path

def _draw_player(frame, cx, cy, scale=1.0, kit_color=(120, 20, 20), shorts_color=(100, 15, 15),
                 skin_color=(60, 95, 140), number="10", name="RONALDINHO", label_color=(0, 215, 255),
                 is_hero=False, t=0.0):
    # Dynamic posture offsets
    h = int(110 * scale)
    w = int(42 * scale)

    # Shadow on grass
    shadow_axes = (int(w * 0.9), int(h * 0.16))
    cv2.ellipse(frame, (cx, cy + int(h * 0.48)), shadow_axes, 0, 0, 360, (10, 45, 15), -1, cv2.LINE_AA)

    # Legs & Boots
    leg_l_x = cx - int(w * 0.3)
    leg_r_x = cx + int(w * 0.3)
    leg_stride = math.sin(t * 12.0) * (14.0 * scale) if is_hero else math.sin(t * 6.0) * (8.0 * scale)

    # Left leg
    cv2.line(frame, (leg_l_x, cy + int(h * 0.1)), (int(leg_l_x - leg_stride), cy + int(h * 0.46)), skin_color, int(8 * scale), cv2.LINE_AA)
    # Right leg
    cv2.line(frame, (leg_r_x, cy + int(h * 0.1)), (int(leg_r_x + leg_stride), cy + int(h * 0.46)), skin_color, int(8 * scale), cv2.LINE_AA)
    # Boots
    boot_color = (0, 215, 255) if is_hero else (30, 30, 30) # Ronaldinho iconic gold Tiempo boots!
    cv2.circle(frame, (int(leg_l_x - leg_stride), cy + int(h * 0.46)), int(5 * scale), boot_color, -1, cv2.LINE_AA)
    cv2.circle(frame, (int(leg_r_x + leg_stride), cy + int(h * 0.46)), int(5 * scale), boot_color, -1, cv2.LINE_AA)

    # Shorts
    cv2.rectangle(frame, (cx - int(w * 0.42), cy - int(h * 0.02)), (cx + int(w * 0.42), cy + int(h * 0.16)), shorts_color, -1, cv2.LINE_AA)

    # Torso (Jersey)
    cv2.rectangle(frame, (cx - int(w * 0.45), cy - int(h * 0.32)), (cx + int(w * 0.45), cy), kit_color, -1, cv2.LINE_AA)
    # Barcelona blaugrana stripes or Madrid trim
    if is_hero:
        # Gold trim & blue stripe
        cv2.rectangle(frame, (cx - int(w * 0.15), cy - int(h * 0.32)), (cx + int(w * 0.15), cy), (180, 50, 40), -1, cv2.LINE_AA)

    # Head & Hair
    head_y = cy - int(h * 0.44)
    cv2.circle(frame, (cx, head_y), int(14 * scale), skin_color, -1, cv2.LINE_AA)
    if is_hero:
        # Ronaldinho's iconic black headband & flowing ponytail
        cv2.line(frame, (cx - int(12 * scale), head_y - int(3 * scale)), (cx + int(12 * scale), head_y - int(3 * scale)), (245, 245, 245), int(3 * scale), cv2.LINE_AA)
        # Ponytail
        cv2.circle(frame, (cx - int(14 * scale), head_y + int(4 * scale)), int(7 * scale), (20, 20, 20), -1, cv2.LINE_AA)

    # Arms
    arm_swing = math.cos(t * 12.0) * (18.0 * scale) if is_hero else math.cos(t * 6.0) * (10.0 * scale)
    cv2.line(frame, (cx - int(w * 0.45), cy - int(h * 0.28)), (int(cx - w * 0.6 + arm_swing), cy - int(h * 0.05)), skin_color, int(6 * scale), cv2.LINE_AA)
    cv2.line(frame, (cx + int(w * 0.45), cy - int(h * 0.28)), (int(cx + w * 0.6 - arm_swing), cy - int(h * 0.05)), skin_color, int(6 * scale), cv2.LINE_AA)

    # Player Label / Number
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.42 * scale
    cv2.putText(frame, f"{name} #{number}", (cx - int(w * 0.9), cy - int(h * 0.54)),
                font, font_scale, (15, 15, 15), 3, cv2.LINE_AA)
    cv2.putText(frame, f"{name} #{number}", (cx - int(w * 0.9), cy - int(h * 0.54)),
                font, font_scale, label_color, 1, cv2.LINE_AA)

def _draw_football(frame, bx, by, scale=1.0):
    r = int(9 * scale)
    # Shadow
    cv2.ellipse(frame, (bx, by + int(6 * scale)), (r + 2, int(4 * scale)), 0, 0, 360, (15, 45, 20), -1, cv2.LINE_AA)
    # Ball body
    cv2.circle(frame, (bx, by), r, (250, 250, 252), -1, cv2.LINE_AA)
    # Pentagons (Adidas Teamgeist / Fevernova pattern)
    cv2.circle(frame, (bx, by), int(r * 0.45), (35, 35, 40), -1, cv2.LINE_AA)
    cv2.circle(frame, (bx + int(r * 0.5), by - int(r * 0.3)), int(r * 0.3), (35, 35, 40), -1, cv2.LINE_AA)
    cv2.circle(frame, (bx - int(r * 0.5), by + int(r * 0.3)), int(r * 0.3), (35, 35, 40), -1, cv2.LINE_AA)

def _draw_broadcast_hud(frame, t, action_desc, width, height):
    # Top-left match scorebug
    cv2.rectangle(frame, (45, 35), (420, 85), (20, 22, 25), -1)
    cv2.rectangle(frame, (45, 35), (420, 85), (60, 65, 70), 1, cv2.LINE_AA)
    # Colors bar
    cv2.rectangle(frame, (45, 35), (55, 85), (0, 215, 255), -1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    match_min = int(58 + (t / 60.0) * 1.5)
    match_sec = int((t * 2.5) % 60)
    score_text = f"RMA 0 - 2 FCB  {match_min}:{match_sec:02d}"
    cv2.putText(frame, score_text, (68, 68), font, 0.72, (245, 245, 245), 2, cv2.LINE_AA)

    # Bottom tracking bar
    cv2.rectangle(frame, (45, height - 75), (width - 45, height - 35), (15, 18, 22), -1)
    cv2.rectangle(frame, (45, height - 75), (width - 45, height - 35), (45, 55, 65), 1, cv2.LINE_AA)

    status_text = f"MOTIONVISION TD | 2005 SANTIAGO BERNABEU | TARGET: RONALDINHO (#10) | {action_desc} | T+{t:05.2f}s"
    cv2.putText(frame, status_text, (65, height - 48), font, 0.52, (0, 215, 255), 1, cv2.LINE_AA)

if __name__ == "__main__":
    out_file = sys.argv[1] if len(sys.argv) > 1 else "assets/videos/synthetic_ronaldinho_test.mp4"
    duration = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    create_synthetic_match_clip(output_path=out_file, duration_sec=duration)
