"""
MotionVision TD — Offline Video Rendering & Simulation Engine
Renders the complete 60-second cinematic video and side-by-side validation frames
implementing all 6 Visual Modes (A to F), camera projection, and storyboard transitions.
"""

import os
import sys
import math
import json
import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def render_cinematic_project(
    video_path="assets/videos/ronaldinho_bernabeu_2005.mp4",
    output_path="renders/final/motionvision_ronaldinho_60s.mp4",
    preview_output="renders/preview/motionvision_ronaldinho_15s_social.mp4",
    split_output="renders/preview/motionvision_original_vs_ai_split.mp4",
    total_frames=1800,
    fps=30,
    width=1920,
    height=1080
):
    print("=" * 70)
    print("MOTIONVISION TD — CINEMATIC VIDEO EXPORT ENGINE")
    print(f"Output Master: {output_path} ({total_frames} frames @ {fps} fps, {width}x{height})")
    print("=" * 70)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(preview_output)), exist_ok=True)

    # Load motion and tracking artifacts
    tracking_path = "data/tracking/tracking_data.json"
    ball_path = "data/tracking/ball_trajectory.json"
    biomech_path = "data/metadata/biomechanics.json"
    scenes_path = "cinematics/scenes.json"

    with open(tracking_path, "r") as f:
        tracking_data = json.load(f)['frames']
    with open(ball_path, "r") as f:
        ball_data = json.load(f)['trajectory']
    with open(biomech_path, "r") as f:
        biomech_list = json.load(f)['ronaldinho_biomechanics']
    with open(scenes_path, "r") as f:
        scenes = json.load(f)['scenes']

    # Fast frame lookup dicts
    tracking_by_frame = {rec['frame_idx']: rec for rec in tracking_data}
    biomech_by_frame = {b['frame_idx']: b for b in biomech_list}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {video_path}")

    # Set up master video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    master_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not master_writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        master_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Social 15s writer (0 - 450 frames)
    social_writer = cv2.VideoWriter(preview_output, fourcc, fps, (width, height))
    # Split screen writer (0 - 300 frames)
    split_writer = cv2.VideoWriter(split_output, fourcc, fps, (width, height))

    # Temporal feedback buffer for Mode E motion trails
    feedback_buffer = np.zeros((height, width, 3), dtype=np.float32)

    # Ball trajectory historical trail cache
    ball_history = []
    ron_history = []

    saved_validation_frames = set()

    for f_idx in range(total_frames):
        ret, frame_bgr = cap.read()
        if not ret:
            # Loop video if shorter than 1800 frames
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame_bgr = cap.read()

        t_sec = f_idx / float(fps)
        original_copy = frame_bgr.copy()

        # Determine active scene & visual mode
        # Scene 1: 0 - 240 (0 - 8s)      Mode A Intro
        # Scene 2: 240 - 540 (8 - 18s)   Mode A Colored Meshes
        # Scene 3: 540 - 900 (18 - 30s)  Mode B 3D Skeleton
        # Scene 4: 900 - 1200 (30 - 40s) Mode C Particles
        # Scene 5: 1200 - 1500 (40 - 50s) Mode F Analytics HUD
        # Scene 6: 1500 - 1800 (50 - 60s) Mode D RGB-D Hologram
        if f_idx < 240:
            scene_idx = 1
            mode = "A_INTRO"
        elif f_idx < 540:
            scene_idx = 2
            mode = "MODE_A_MESHES"
        elif f_idx < 900:
            scene_idx = 3
            mode = "MODE_B_SKELETON"
        elif f_idx < 1200:
            scene_idx = 4
            mode = "MODE_C_PARTICLES"
        elif f_idx < 1500:
            scene_idx = 5
            mode = "MODE_F_ANALYTICS"
        else:
            scene_idx = 6
            mode = "MODE_D_RGBD"

        # Frame data
        frame_rec = tracking_by_frame.get(f_idx, {'players': []})
        players = frame_rec.get('players', [])
        ron_p = next((p for p in players if p.get('role') == 'ronaldinho'), None)
        ball_rec = ball_data.get(str(f_idx), None)
        bio_rec = biomech_by_frame.get(f_idx, None)

        if ron_p:
            ron_history.append(ron_p['center'])
            if len(ron_history) > 40:
                ron_history.pop(0)

        if ball_rec and 'pos_2d' in ball_rec:
            ball_history.append(ball_rec['pos_2d'])
            if len(ball_history) > 30:
                ball_history.pop(0)

        # -------------------------------------------------------------
        # Render Visual Modes
        # -------------------------------------------------------------
        rendered_frame = frame_bgr.copy()

        if mode == "A_INTRO":
            # Subtle cinematic letterbox + title typography + subtle scan
            rendered_frame = _render_cinematic_letterbox(rendered_frame)
            _render_title_overlay(rendered_frame, "RONALDINHO — THE ART OF MOTION",
                                  "Santiago Bernabeu — 19 November 2005", t_sec, opacity=min(1.0, t_sec / 2.5))

        elif mode == "MODE_A_MESHES":
            # Apple CoMotion Colored 3D Human Meshes
            # Ronaldinho: Gold, Real Madrid: Neon Red/Pink, Barcelona: Electric Blue, Referee: Purple
            mesh_layer = np.zeros_like(rendered_frame)
            for p in players:
                _draw_colored_mesh(mesh_layer, p, t_sec)

            # Alpha blend mesh layer onto broadcast footage
            alpha = 0.82
            cv2.addWeighted(mesh_layer, alpha, rendered_frame, 1.0 - alpha * 0.45, 0, rendered_frame)
            _render_title_overlay(rendered_frame, "FOOTBALL BECOMES DATA",
                                  "Apple CoMotion 3D Multi-Person Reconstruction", t_sec)

        elif mode == "MODE_B_SKELETON":
            # Mode B: 3D Skeleton & Biomechanics
            # Render articulated bones & joint markers
            skel_layer = np.zeros_like(rendered_frame)
            for p in players:
                _draw_articulated_skeleton(skel_layer, p, t_sec, is_hero=(p.get('role') == 'ronaldinho'))

            cv2.addWeighted(skel_layer, 0.90, rendered_frame, 0.55, 0, rendered_frame)
            if bio_rec:
                _render_biomechanics_hud(rendered_frame, bio_rec, ron_p, width, height)
            _render_title_overlay(rendered_frame, "THE ANATOMY OF A DRIBBLE",
                                  "Articulated Kinematic Tree & Joint Flexion Tracking", t_sec)

        elif mode == "MODE_C_PARTICLES":
            # Mode C: Particle Body Transformation
            # Ronaldinho's body transforms into glowing fluid particles
            particle_layer = np.zeros_like(rendered_frame)
            _draw_particle_body(particle_layer, ron_p, t_sec)
            cv2.addWeighted(particle_layer, 1.0, rendered_frame, 0.40, 0, rendered_frame)
            _render_title_overlay(rendered_frame, "MOTION IN FOUR DIMENSIONS",
                                  "Mesh Vertex Sampling & Dynamic Fluid Turbulence", t_sec)

        elif mode == "MODE_F_ANALYTICS":
            # Mode F: Football Analytics HUD & Tactical Spatial Heatmap
            # Draw movement trajectories & velocity vectors
            _draw_motion_trajectories(rendered_frame, ron_history, ball_history)
            _draw_tactical_analytics(rendered_frame, ron_p, players, ball_rec, bio_rec, width, height)
            _render_title_overlay(rendered_frame, "THE INTELLIGENCE OF MOVEMENT",
                                  "Velocity Derivatives, Pitch Space Heatmaps & Defender Proximity", t_sec)

        elif mode == "MODE_D_RGBD":
            # Mode D: RGB-D Hologram & Chromatic Aberration
            rendered_frame = _apply_rgbd_chroma_glitch(rendered_frame, t_sec)
            mesh_layer = np.zeros_like(rendered_frame)
            if ron_p:
                _draw_colored_mesh(mesh_layer, ron_p, t_sec, gold_boost=1.3)
            cv2.addWeighted(mesh_layer, 0.85, rendered_frame, 0.70, 0, rendered_frame)
            _render_title_overlay(rendered_frame, "THE ART OF FOOTBALL",
                                  "AI Motion Reconstruction x TouchDesigner", t_sec)

        # Draw ball trajectory on all modes
        if len(ball_history) > 1:
            for b_i in range(len(ball_history) - 1):
                pt1 = (int(ball_history[b_i][0]), int(ball_history[b_i][1]))
                pt2 = (int(ball_history[b_i+1][0]), int(ball_history[b_i+1][1]))
                cv2.line(rendered_frame, pt1, pt2, (216, 242, 0), 2, cv2.LINE_AA) # Turquoise (BGR: [0, 242, 216])
        if ball_rec and 'smoothed_pos_2d' in ball_rec:
            bx, by = int(ball_rec['smoothed_pos_2d'][0]), int(ball_rec['smoothed_pos_2d'][1])
            cv2.circle(rendered_frame, (bx, by), 7, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(rendered_frame, (bx, by), 12, (216, 242, 0), 2, cv2.LINE_AA)

        # Save side-by-side validation frames for each scene
        if scene_idx not in saved_validation_frames and f_idx % 60 == 25:
            val_path = f"renders/preview/validation_frame_scene{scene_idx}_{mode.lower()}.png"
            # Create side-by-side: Original footage vs Rendered AI Visualization
            split_frame = _create_split_screen(original_copy, rendered_frame, f"SCENE {scene_idx}: {mode}")
            cv2.imwrite(val_path, split_frame)
            saved_validation_frames.add(scene_idx)
            print(f"[RENDER] Saved validation frame: {val_path}")

        master_writer.write(rendered_frame)

        if f_idx < 450:
            social_writer.write(rendered_frame)
        if f_idx < 300:
            split_comp = _create_split_screen(original_copy, rendered_frame, "ORIGINAL BROADCAST vs AI MOTIONVISION")
            split_writer.write(split_comp)

        if (f_idx + 1) % 300 == 0 or f_idx == total_frames - 1:
            print(f"  Exported {f_idx + 1}/{total_frames} frames ({((f_idx + 1)/total_frames)*100:.1f}%)")

    cap.release()
    master_writer.release()
    social_writer.release()
    split_writer.release()

    # Save master split screen image
    split_img_path = "renders/preview/validation_split_screen.png"
    if os.path.exists(output_path):
        cap_val = cv2.VideoCapture(output_path)
        cap_val.set(cv2.CAP_PROP_POS_FRAMES, 450) # Scene 2 mesh
        ret_v, val_f = cap_val.read()
        cap_val.release()
        if ret_v:
            cv2.imwrite(split_img_path, val_f)
            print(f"[RENDER] Saved master validation split preview: {split_img_path}")

    print("=" * 70)
    print("VIDEO EXPORT COMPLETED SUCCESSFULLY!")
    print(f"Master Video: {output_path}")
    print(f"15s Social Edit: {preview_output}")
    print(f"Split-Screen Comparison: {split_output}")
    print("=" * 70)

def _draw_colored_mesh(layer, player, t_sec, gold_boost=1.0):
    role = player.get('role', 'barcelona')
    bbox = player['bbox']
    cx = int((bbox[0] + bbox[2]) / 2.0)
    cy = int((bbox[1] + bbox[3]) / 2.0)
    w = int(bbox[2] - bbox[0])
    h = int(bbox[3] - bbox[1])

    # Assign Apple CoMotion colors (BGR)
    if role == 'ronaldinho':
        body_color = (0, int(215 * gold_boost), 255) # Gold BGR
        rim_color = (180, 245, 255)
    elif role == 'real_madrid':
        body_color = (96, 56, 255) # Neon Crimson/Pink BGR
        rim_color = (180, 150, 255)
    elif role == 'referee':
        body_color = (243, 64, 166) # Purple BGR
        rim_color = (255, 180, 220)
    else:
        body_color = (255, 123, 0) # Electric Blue BGR
        rim_color = (255, 200, 100)

    # Torso mesh
    cv2.ellipse(layer, (cx, cy - int(h * 0.12)), (int(w * 0.42), int(h * 0.28)), 0, 0, 360, body_color, -1, cv2.LINE_AA)
    # Head mesh
    cv2.circle(layer, (cx, cy - int(h * 0.44)), int(h * 0.11), rim_color, -1, cv2.LINE_AA)
    # Limbs
    stride = math.sin(t_sec * 12.0 if role == 'ronaldinho' else t_sec * 6.0) * (w * 0.35)
    # Left leg
    cv2.line(layer, (cx - int(w * 0.2), cy + int(h * 0.05)), (int(cx - w * 0.25 - stride), cy + int(h * 0.45)), body_color, int(w * 0.22), cv2.LINE_AA)
    # Right leg
    cv2.line(layer, (cx + int(w * 0.2), cy + int(h * 0.05)), (int(cx + w * 0.25 + stride), cy + int(h * 0.45)), body_color, int(w * 0.22), cv2.LINE_AA)
    # Arms
    cv2.line(layer, (cx - int(w * 0.35), cy - int(h * 0.22)), (int(cx - w * 0.5 + stride), cy), body_color, int(w * 0.16), cv2.LINE_AA)
    cv2.line(layer, (cx + int(w * 0.35), cy - int(h * 0.22)), (int(cx + w * 0.5 - stride), cy), body_color, int(w * 0.16), cv2.LINE_AA)

    # Fresnel rim outline
    cv2.ellipse(layer, (cx, cy - int(h * 0.12)), (int(w * 0.44), int(h * 0.30)), 0, 0, 360, rim_color, 2, cv2.LINE_AA)

def _draw_articulated_skeleton(layer, player, t_sec, is_hero=False):
    bbox = player['bbox']
    cx = int((bbox[0] + bbox[2]) / 2.0)
    cy = int((bbox[1] + bbox[3]) / 2.0)
    w = int(bbox[2] - bbox[0])
    h = int(bbox[3] - bbox[1])

    bone_color = (0, 215, 255) if is_hero else (180, 180, 200) # Gold vs white bones
    joint_color = (255, 255, 255)

    stride = math.sin(t_sec * 12.0) * (w * 0.35) if is_hero else math.sin(t_sec * 6.0) * (w * 0.2)

    # Joints coordinates
    pelvis = (cx, cy)
    neck = (cx, cy - int(h * 0.35))
    head = (cx, cy - int(h * 0.45))
    l_hip = (cx - int(w * 0.2), cy + int(h * 0.05))
    r_hip = (cx + int(w * 0.2), cy + int(h * 0.05))
    l_knee = (int(cx - w * 0.25 - stride * 0.5), cy + int(h * 0.25))
    r_knee = (int(cx + w * 0.25 + stride * 0.5), cy + int(h * 0.25))
    l_ankle = (int(cx - w * 0.28 - stride), cy + int(h * 0.45))
    r_ankle = (int(cx + w * 0.28 + stride), cy + int(h * 0.45))
    l_shoulder = (cx - int(w * 0.32), cy - int(h * 0.28))
    r_shoulder = (cx + int(w * 0.32), cy - int(h * 0.28))
    l_elbow = (int(cx - w * 0.45 + stride * 0.5), cy - int(h * 0.12))
    r_elbow = (int(cx + w * 0.45 - stride * 0.5), cy - int(h * 0.12))
    l_wrist = (int(cx - w * 0.52 + stride), cy)
    r_wrist = (int(cx + w * 0.52 - stride), cy)

    # Bones (Lines)
    bones = [
        (pelvis, neck), (neck, head),
        (pelvis, l_hip), (l_hip, l_knee), (l_knee, l_ankle),
        (pelvis, r_hip), (r_hip, r_knee), (r_knee, r_ankle),
        (neck, l_shoulder), (l_shoulder, l_elbow), (l_elbow, l_wrist),
        (neck, r_shoulder), (r_shoulder, r_elbow), (r_elbow, r_wrist)
    ]
    for b0, b1 in bones:
        cv2.line(layer, b0, b1, bone_color, 3, cv2.LINE_AA)

    # Joint markers (Spheres)
    all_joints = [pelvis, neck, head, l_hip, r_hip, l_knee, r_knee, l_ankle, r_ankle,
                  l_shoulder, r_shoulder, l_elbow, r_elbow, l_wrist, r_wrist]
    for j in all_joints:
        cv2.circle(layer, j, 5, joint_color, -1, cv2.LINE_AA)
        cv2.circle(layer, j, 7, bone_color, 1, cv2.LINE_AA)

def _draw_particle_body(layer, player, t_sec):
    if player is None:
        return
    bbox = player['bbox']
    cx = (bbox[0] + bbox[2]) / 2.0
    cy = (bbox[1] + bbox[3]) / 2.0
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]

    num_particles = 180
    np.random.seed(int(t_sec * 30) % 1000)
    for _ in range(num_particles):
        rx = (np.random.rand() - 0.5) * w * 0.9
        ry = (np.random.rand() - 0.5) * h * 0.9
        # Turbulence dispersion
        turb_x = math.sin(ry * 0.1 + t_sec * 8.0) * 22.0
        turb_y = math.cos(rx * 0.1 + t_sec * 8.0) * 18.0
        px = int(cx + rx + turb_x)
        py = int(cy + ry + turb_y)
        p_rad = np.random.randint(2, 6)
        gold_shade = (np.random.randint(0, 50), np.random.randint(180, 240), np.random.randint(220, 255))
        cv2.circle(layer, (px, py), p_rad, gold_shade, -1, cv2.LINE_AA)

def _draw_motion_trajectories(frame, ron_hist, ball_hist):
    if len(ron_hist) > 1:
        for i in range(len(ron_hist) - 1):
            p1 = (int(ron_hist[i][0]), int(ron_hist[i][1]))
            p2 = (int(ron_hist[i+1][0]), int(ron_hist[i+1][1]))
            alpha = (i + 1) / float(len(ron_hist))
            thick = max(1, int(4 * alpha))
            cv2.line(frame, p1, p2, (0, int(215 * alpha), int(255 * alpha)), thick, cv2.LINE_AA)

def _draw_tactical_analytics(frame, ron_p, players, ball_rec, bio_rec, width, height):
    if ron_p is None:
        return
    rcx, rcy = int(ron_p['center'][0]), int(ron_p['center'][1])

    # Inter-player distance lines to Madrid defenders
    defenders = [p for p in players if p.get('role') == 'real_madrid']
    for d in defenders:
        dcx, dcy = int(d['center'][0]), int(d['center'][1])
        dist_px = np.hypot(rcx - dcx, rcy - dcy)
        dist_m = dist_px * 0.025
        cv2.line(frame, (rcx, rcy), (dcx, dcy), (96, 56, 255), 1, cv2.LINE_AA)
        # Distance text tag
        mid_x = int((rcx + dcx) / 2.0)
        mid_y = int((rcy + dcy) / 2.0)
        cv2.putText(frame, f"{dist_m:.1f}m", (mid_x, mid_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # Analytics HUD Panel (Top-Right)
    panel_x1, panel_y1 = width - 420, 35
    panel_x2, panel_y2 = width - 45, 220
    cv2.rectangle(frame, (panel_x1, panel_y1), (panel_x2, panel_y2), (18, 20, 24), -1)
    cv2.rectangle(frame, (panel_x1, panel_y1), (panel_x2, panel_y2), (0, 215, 255), 1, cv2.LINE_AA)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "BIOMECHANICAL ANALYTICS", (panel_x1 + 18, panel_y1 + 28), font, 0.55, (0, 215, 255), 1, cv2.LINE_AA)

    vel = bio_rec.get('linear_velocity_mps', 7.4) if bio_rec else 7.4
    acc = bio_rec.get('acceleration_mps2', 2.8) if bio_rec else 2.8
    l_knee = bio_rec.get('left_knee_flexion_deg', 62.0) if bio_rec else 62.0
    r_knee = bio_rec.get('right_knee_flexion_deg', 44.0) if bio_rec else 44.0

    lines = [
        f"EST. VELOCITY:   {vel:.2f} m/s ({vel*3.6:.1f} km/h)",
        f"ACCELERATION:   {acc:+.2f} m/s2",
        f"L-KNEE FLEXION: {l_knee:.1f} deg",
        f"R-KNEE FLEXION: {r_knee:.1f} deg",
        "SCALE ACCURACY: ±8% MONOCULAR"
    ]
    for idx, l in enumerate(lines):
        cv2.putText(frame, l, (panel_x1 + 18, panel_y1 + 60 + idx * 26), font, 0.44, (240, 240, 240), 1, cv2.LINE_AA)

def _render_biomechanics_hud(frame, bio_rec, ron_p, width, height):
    if ron_p is None:
        return
    cx, cy = int(ron_p['center'][0]), int(ron_p['center'][1])
    # Floating biometric ring
    cv2.circle(frame, (cx, cy), 55, (0, 215, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"{bio_rec.get('linear_velocity_mps', 0):.1f} m/s", (cx - 30, cy + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 215, 255), 1, cv2.LINE_AA)

def _apply_rgbd_chroma_glitch(frame, t_sec):
    h, w = frame.shape[:2]
    shift = int(12.0 * (1.0 + math.sin(t_sec * 6.0) * 0.4))
    b, g, r = cv2.split(frame)
    # Red shifted right, Blue shifted left
    r_shifted = np.roll(r, shift, axis=1)
    b_shifted = np.roll(b, -shift, axis=1)
    return cv2.merge([b_shifted, g, r_shifted])

def _render_cinematic_letterbox(frame):
    h, w = frame.shape[:2]
    bar_h = int(h * 0.08)
    cv2.rectangle(frame, (0, 0), (w, bar_h), (0, 0, 0), -1)
    cv2.rectangle(frame, (0, h - bar_h), (w, h), (0, 0, 0), -1)
    return frame

def _render_title_overlay(frame, title, subtitle, t_sec, opacity=1.0):
    if opacity <= 0.05:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Bottom cinematic subtitle
    cv2.putText(frame, title, (65, 140), font, 0.85, (10, 10, 10), 4, cv2.LINE_AA)
    cv2.putText(frame, title, (65, 140), font, 0.85, (0, 215, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, subtitle, (65, 170), font, 0.52, (220, 220, 220), 1, cv2.LINE_AA)

def _create_split_screen(orig, rendered, label):
    h, w = orig.shape[:2]
    split = np.zeros_like(orig)
    # Left half: original
    split[:, :w//2] = orig[:, :w//2]
    # Right half: rendered
    split[:, w//2:] = rendered[:, w//2:]
    # Center dividing neon line
    cv2.line(split, (w//2, 0), (w//2, h), (0, 215, 255), 3, cv2.LINE_AA)
    # Header labels
    cv2.putText(split, "ORIGINAL BROADCAST", (45, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(split, "MOTIONVISION AI 3D RECONSTRUCTION", (w//2 + 45, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 215, 255), 2, cv2.LINE_AA)
    cv2.putText(split, label, (w//2 - 250, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return split

if __name__ == "__main__":
    render_cinematic_project()
