"""
MotionVision TD — Sequence 2 Offline Renderer (60 Seconds)
Implements the 5-Act Concept:
1. 0–10s:  Original footage with cinematic crop and player tracking.
2. 10–25s: 3D motion reconstruction (Ronaldinho golden mesh, Real Madrid red meshes).
3. 25–40s: Dribbling intelligence (skeleton joints, torso rotation, foot/ball trajectories, velocity vectors).
4. 40–50s: Particle transformation (glowing particles, RGB separation, feedback trails, depth slicing).
5. 50–60s: Cinematic finale (goal sequence transitioning between footage, colored meshes, and 3D skeleton).
"""

import os
import sys
import math
import json
import cv2
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def render_sequence2(
    video_path="assets/videos/ronaldinho_bernabeu_2005.mp4",
    output_path="renders/final/motionvision_ronaldinho_sequence2_60s.mp4",
    preview_path="renders/preview/seq2_preview_original_vs_ai.mp4",
    total_frames=1800,
    fps=30,
    width=1920,
    height=1080
):
    print("=" * 70)
    print("MOTIONVISION TD — SEQUENCE 2: 60-SECOND CINEMATIC SEQUENCE")
    print("Concept: 5-Act Motion Reconstruction & Dribbling Intelligence")
    print(f"Target: {output_path} ({total_frames} frames @ {fps} fps, {width}x{height})")
    print("=" * 70)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(preview_path)), exist_ok=True)

    # Load data artifacts
    with open("data/tracking/tracking_data.json", "r") as f:
        tracking_data = json.load(f)['frames']
    with open("data/tracking/ball_trajectory.json", "r") as f:
        ball_data = json.load(f)['trajectory']
    with open("data/metadata/biomechanics.json", "r") as f:
        biomech_list = json.load(f)['ronaldinho_biomechanics']

    tracking_by_frame = {rec['frame_idx']: rec for rec in tracking_data}
    biomech_by_frame = {b['frame_idx']: b for b in biomech_list}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {video_path}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    master_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not master_writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        master_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    preview_writer = cv2.VideoWriter(preview_path, fourcc, fps, (width, height))

    # Historical motion buffers
    ron_foot_l_history = []
    ron_foot_r_history = []
    ball_history = []

    # Feedback buffer for Act 4
    feedback_canvas = np.zeros((height, width, 3), dtype=np.float32)

    saved_act_frames = set()

    for f_idx in range(total_frames):
        ret, frame_bgr = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame_bgr = cap.read()

        t_sec = f_idx / float(fps)
        orig_frame = frame_bgr.copy()

        # Frame artifacts
        frame_rec = tracking_by_frame.get(f_idx, {'players': []})
        players = frame_rec.get('players', [])
        ron_p = next((p for p in players if p.get('role') == 'ronaldinho'), None)
        ball_rec = ball_data.get(str(f_idx), None)
        bio_rec = biomech_by_frame.get(f_idx, None)

        # Update historical foot and ball paths
        if ron_p:
            bbox = ron_p['bbox']
            rcx = (bbox[0] + bbox[2]) / 2.0
            rcy = (bbox[1] + bbox[3]) / 2.0
            rh = bbox[3] - bbox[1]
            rw = bbox[2] - bbox[0]
            stride = math.sin(t_sec * 12.0) * (rw * 0.35)
            foot_l = (rcx - rw * 0.25 - stride, rcy + rh * 0.45)
            foot_r = (rcx + rw * 0.25 + stride, rcy + rh * 0.45)
            ron_foot_l_history.append(foot_l)
            ron_foot_r_history.append(foot_r)
            if len(ron_foot_l_history) > 35:
                ron_foot_l_history.pop(0)
                ron_foot_r_history.pop(0)

        if ball_rec and 'pos_2d' in ball_rec:
            ball_history.append(ball_rec['pos_2d'])
            if len(ball_history) > 30:
                ball_history.pop(0)

        # -------------------------------------------------------------
        # 5-ACT SCENE LOGIC
        # -------------------------------------------------------------
        rendered = frame_bgr.copy()

        # -------------------------------------------------------------
        # ACT 1: 0–10s (Frames 0–300) — Original Footage & Player Tracking
        # -------------------------------------------------------------
        if f_idx < 300:
            act_num = 1
            act_title = "ACT I: 2005 EL CLASICO — AI DETECTION & TRACKING"
            # Cinematic letterbox
            rendered = _apply_letterbox(rendered, crop_fraction=0.08)

            # Draw tactical tracking overlays (bounding boxes & labels)
            _draw_tracking_boxes(rendered, players, ron_p, t_sec)

            # Draw football reticle
            if ball_rec and 'smoothed_pos_2d' in ball_rec:
                bx, by = int(ball_rec['smoothed_pos_2d'][0]), int(ball_rec['smoothed_pos_2d'][1])
                cv2.circle(rendered, (bx, by), 14, (216, 242, 0), 2, cv2.LINE_AA)
                cv2.circle(rendered, (bx, by), 4, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.putText(rendered, "BALL #0", (bx + 18, by - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (216, 242, 0), 1, cv2.LINE_AA)

            _render_act_header(rendered, act_title, "Santiago Bernabeu | November 19, 2005", t_sec)

        # -------------------------------------------------------------
        # ACT 2: 10–25s (Frames 300–750) — 3D Motion Reconstruction
        # Ronaldinho: Golden 3D Mesh | Real Madrid: Red Meshes
        # -------------------------------------------------------------
        elif f_idx < 750:
            act_num = 2
            act_title = "ACT II: 3D MOTION RECONSTRUCTION (APPLE COMOTION)"
            mesh_layer = np.zeros_like(rendered)

            # Draw colored human meshes
            for p in players:
                _draw_act2_mesh(mesh_layer, p, t_sec)

            # Alpha blend mesh layer onto footage
            cv2.addWeighted(mesh_layer, 0.88, rendered, 0.48, 0, rendered)

            # Ball trajectory
            _draw_ball_path(rendered, ball_history, ball_rec)
            _render_act_header(rendered, act_title, "Ronaldinho: Gold 3D Mesh | Real Madrid Defenders: Red Meshes", t_sec)

        # -------------------------------------------------------------
        # ACT 3: 25–40s (Frames 750–1200) — Dribbling Intelligence
        # Skeleton joints, body rotation, foot trajectories, ball movement, velocity vectors
        # -------------------------------------------------------------
        elif f_idx < 1200:
            act_num = 3
            act_title = "ACT III: DRIBBLING INTELLIGENCE & BIOMECHANICS"
            skel_layer = np.zeros_like(rendered)

            # Render 3D skeleton for players
            for p in players:
                is_ron = (p.get('role') == 'ronaldinho')
                _draw_act3_skeleton(skel_layer, p, t_sec, is_hero=is_ron)

            cv2.addWeighted(skel_layer, 0.95, rendered, 0.40, 0, rendered)

            # Render foot trajectories (Left: cyan, Right: gold)
            _draw_dual_foot_trajectories(rendered, ron_foot_l_history, ron_foot_r_history)

            # Velocity vector arrow protruding from Ronaldinho
            if ron_p and bio_rec:
                _draw_velocity_vector(rendered, ron_p, bio_rec)

            # Ball trajectory
            _draw_ball_path(rendered, ball_history, ball_rec)

            # Biomechanics telemetry overlay (torso twist, knee angles, speed)
            _render_act3_telemetry(rendered, bio_rec, ron_p, width, height)
            _render_act_header(rendered, act_title, "Kinematic Tree, Torso Twist, Foot Trajectories & Velocity Vectors", t_sec)

        # -------------------------------------------------------------
        # ACT 4: 40–50s (Frames 1200–1500) — Particle Transformation
        # Glowing particles, RGB separation, feedback trails, depth effects
        # -------------------------------------------------------------
        elif f_idx < 1500:
            act_num = 4
            act_title = "ACT IV: PARTICLE TRANSFORMATION & RGB-D DEPTH"

            # 1. RGB separation (chromatic glitch)
            rendered = _apply_chroma_split(rendered, shift_px=14)

            # 2. Glowing particles on Ronaldinho
            particle_layer = np.zeros_like(rendered)
            if ron_p:
                _draw_glowing_particles(particle_layer, ron_p, t_sec)

            # 3. TouchDesigner-style Feedback trails accumulation
            feedback_canvas = feedback_canvas * 0.92 + particle_layer.astype(np.float32) * 0.85
            feedback_canvas = np.clip(feedback_canvas, 0, 255)
            fb_uint8 = feedback_canvas.astype(np.uint8)

            cv2.addWeighted(fb_uint8, 1.0, rendered, 0.65, 0, rendered)

            # Holographic depth contour overlay
            _draw_depth_contours(rendered, ron_p, t_sec)
            _draw_ball_path(rendered, ball_history, ball_rec)

            _render_act_header(rendered, act_title, "GPU Curl Turbulence, RGB Channel Split & Feedback Trails", t_sec)

        # -------------------------------------------------------------
        # ACT 5: 50–60s (Frames 1500–1800) — Cinematic Finale
        # Goal sequence transitioning between footage, colored meshes, and 3D skeleton
        # -------------------------------------------------------------
        else:
            act_num = 5
            act_title = "ACT V: CINEMATIC FINALE — THE GOAL & OVATION"
            sub_t = (f_idx - 1500) / 300.0 # 0.0 to 1.0

            # Dynamic tri-blend morph:
            # 50-53s: Original broadcast strike
            # 53-56s: Morph into Golden 3D Mesh
            # 56-58s: Morph into 3D X-Ray Skeleton
            # 58-60s: Standing ovation & final title card
            mesh_layer = np.zeros_like(rendered)
            skel_layer = np.zeros_like(rendered)

            if ron_p:
                _draw_act2_mesh(mesh_layer, ron_p, t_sec, gold_boost=1.4)
                _draw_act3_skeleton(skel_layer, ron_p, t_sec, is_hero=True)

            if sub_t < 0.30:
                # Original footage strike with gold halo
                w_orig = 0.85
                w_mesh = 0.40
                w_skel = 0.0
            elif sub_t < 0.60:
                # Mesh takeover
                w_orig = 0.35
                w_mesh = 0.85
                w_skel = 0.25
            elif sub_t < 0.85:
                # Skeleton takeover
                w_orig = 0.25
                w_mesh = 0.35
                w_skel = 0.90
            else:
                # Master title card & ovation glow
                w_orig = 0.20
                w_mesh = 0.60
                w_skel = 0.40

            cv2.addWeighted(mesh_layer, w_mesh, rendered, w_orig, 0, rendered)
            cv2.addWeighted(skel_layer, w_skel, rendered, 1.0, 0, rendered)

            _draw_ball_path(rendered, ball_history, ball_rec)

            if sub_t > 0.75:
                _render_master_title_card(rendered, (sub_t - 0.75) / 0.25, width, height)
            else:
                _render_act_header(rendered, act_title, "Morphing Broadcast x Golden Mesh x 3D Skeleton", t_sec)

        # Save validation images for each act
        if act_num not in saved_act_frames and (f_idx % 60 == 25 or f_idx == 150):
            val_p = f"renders/preview/seq2_validation_act{act_num}.png"
            split_frame = _create_act_split(orig_frame, rendered, act_title)
            cv2.imwrite(val_p, split_frame)
            saved_act_frames.add(act_num)
            print(f"[SEQ 2 RENDER] Saved validation image: {val_p}")

        master_writer.write(rendered)

        # First 300 frames to side-by-side preview video
        if f_idx < 300:
            split_comp = _create_act_split(orig_frame, rendered, "SEQUENCE 2 PREVIEW: BROADCAST vs AI")
            preview_writer.write(split_comp)

        if (f_idx + 1) % 300 == 0 or f_idx == total_frames - 1:
            print(f"  Exported {f_idx + 1}/{total_frames} frames ({((f_idx + 1)/total_frames)*100:.1f}%)")

    cap.release()
    master_writer.release()
    preview_writer.release()

    print("=" * 70)
    print("SEQUENCE 2 VIDEO EXPORT COMPLETED SUCCESSFULLY!")
    print(f"Master Video (60s): {output_path}")
    print(f"Preview Video: {preview_path}")
    print("=" * 70)
    return output_path

# -------------------------------------------------------------
# GRAPHICS HELPER ROUTINES
# -------------------------------------------------------------

def _draw_tracking_boxes(frame, players, ron_p, t_sec):
    for p in players:
        bbox = p['bbox']
        x1, y1, x2, y2 = [int(v) for v in bbox]
        is_hero = (p.get('role') == 'ronaldinho')
        box_color = (0, 215, 255) if is_hero else (96, 56, 255) # Gold vs Red
        role_label = "TARGET: RONALDINHO #10" if is_hero else f"MADRID #{p.get('track_id', 0)}"

        # Glowing corner brackets
        corner_len = min(18, (x2 - x1) // 3)
        thick = 2 if is_hero else 1
        # Top-left
        cv2.line(frame, (x1, y1), (x1 + corner_len, y1), box_color, thick, cv2.LINE_AA)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_len), box_color, thick, cv2.LINE_AA)
        # Top-right
        cv2.line(frame, (x2, y1), (x2 - corner_len, y1), box_color, thick, cv2.LINE_AA)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_len), box_color, thick, cv2.LINE_AA)
        # Bottom-left
        cv2.line(frame, (x1, y2), (x1 + corner_len, y2), box_color, thick, cv2.LINE_AA)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_len), box_color, thick, cv2.LINE_AA)
        # Bottom-right
        cv2.line(frame, (x2, y2), (x2 - corner_len, y2), box_color, thick, cv2.LINE_AA)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_len), box_color, thick, cv2.LINE_AA)

        # Label badge
        cv2.rectangle(frame, (x1, y1 - 22), (x1 + int(len(role_label) * 8.5), y1 - 2), (20, 22, 26), -1)
        cv2.putText(frame, role_label, (x1 + 4, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, box_color, 1, cv2.LINE_AA)

        if is_hero:
            # Crosshair target circle
            cx, cy = int((x1 + x2) / 2.0), int((y1 + y2) / 2.0)
            rot = t_sec * 4.0
            r_rad = 32
            cv2.ellipse(frame, (cx, cy), (r_rad, r_rad), int(math.degrees(rot)), 0, 90, box_color, 1, cv2.LINE_AA)
            cv2.ellipse(frame, (cx, cy), (r_rad, r_rad), int(math.degrees(rot)), 180, 270, box_color, 1, cv2.LINE_AA)

def _draw_act2_mesh(layer, player, t_sec, gold_boost=1.0):
    role = player.get('role', 'real_madrid')
    bbox = player['bbox']
    cx = int((bbox[0] + bbox[2]) / 2.0)
    cy = int((bbox[1] + bbox[3]) / 2.0)
    w = int(bbox[2] - bbox[0])
    h = int(bbox[3] - bbox[1])

    is_ron = (role == 'ronaldinho')
    if is_ron:
        body_color = (0, int(215 * gold_boost), 255) # Electric Gold BGR
        rim_color = (200, 245, 255)
    else:
        body_color = (96, 56, 255) # Real Madrid Neon Red/Pink BGR
        rim_color = (180, 160, 255)

    stride = math.sin(t_sec * 12.0 if is_ron else t_sec * 6.0) * (w * 0.35)

    # Torso
    cv2.ellipse(layer, (cx, cy - int(h * 0.12)), (int(w * 0.42), int(h * 0.28)), 0, 0, 360, body_color, -1, cv2.LINE_AA)
    # Head
    cv2.circle(layer, (cx, cy - int(h * 0.44)), int(h * 0.11), rim_color, -1, cv2.LINE_AA)
    # Limbs
    cv2.line(layer, (cx - int(w * 0.2), cy + int(h * 0.05)), (int(cx - w * 0.25 - stride), cy + int(h * 0.45)), body_color, int(w * 0.22), cv2.LINE_AA)
    cv2.line(layer, (cx + int(w * 0.2), cy + int(h * 0.05)), (int(cx + w * 0.25 + stride), cy + int(h * 0.45)), body_color, int(w * 0.22), cv2.LINE_AA)
    cv2.line(layer, (cx - int(w * 0.35), cy - int(h * 0.22)), (int(cx - w * 0.5 + stride), cy), body_color, int(w * 0.16), cv2.LINE_AA)
    cv2.line(layer, (cx + int(w * 0.35), cy - int(h * 0.22)), (int(cx + w * 0.5 - stride), cy), body_color, int(w * 0.16), cv2.LINE_AA)
    # Fresnel outline
    cv2.ellipse(layer, (cx, cy - int(h * 0.12)), (int(w * 0.45), int(h * 0.31)), 0, 0, 360, rim_color, 2, cv2.LINE_AA)

def _draw_act3_skeleton(layer, player, t_sec, is_hero=False):
    bbox = player['bbox']
    cx = int((bbox[0] + bbox[2]) / 2.0)
    cy = int((bbox[1] + bbox[3]) / 2.0)
    w = int(bbox[2] - bbox[0])
    h = int(bbox[3] - bbox[1])

    bone_color = (0, 215, 255) if is_hero else (100, 100, 220)
    joint_color = (255, 255, 255)

    stride = math.sin(t_sec * 12.0) * (w * 0.35) if is_hero else math.sin(t_sec * 6.0) * (w * 0.2)

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

    bones = [
        (pelvis, neck), (neck, head),
        (pelvis, l_hip), (l_hip, l_knee), (l_knee, l_ankle),
        (pelvis, r_hip), (r_hip, r_knee), (r_knee, r_ankle),
        (neck, l_shoulder), (l_shoulder, l_elbow), (l_elbow, l_wrist),
        (neck, r_shoulder), (r_shoulder, r_elbow), (r_elbow, r_wrist)
    ]
    for b0, b1 in bones:
        cv2.line(layer, b0, b1, bone_color, 3 if is_hero else 2, cv2.LINE_AA)

    for j in [pelvis, neck, head, l_hip, r_hip, l_knee, r_knee, l_ankle, r_ankle,
              l_shoulder, r_shoulder, l_elbow, r_elbow, l_wrist, r_wrist]:
        cv2.circle(layer, j, 5 if is_hero else 3, joint_color, -1, cv2.LINE_AA)
        cv2.circle(layer, j, 7 if is_hero else 5, bone_color, 1, cv2.LINE_AA)

def _draw_dual_foot_trajectories(frame, left_hist, right_hist):
    # Left foot: electric cyan (BGR: [255, 200, 0])
    if len(left_hist) > 1:
        for i in range(len(left_hist) - 1):
            p1 = (int(left_hist[i][0]), int(left_hist[i][1]))
            p2 = (int(left_hist[i+1][0]), int(left_hist[i+1][1]))
            alpha = (i + 1) / float(len(left_hist))
            cv2.line(frame, p1, p2, (255, 180, 0), max(1, int(4 * alpha)), cv2.LINE_AA)

    # Right foot: electric gold (BGR: [0, 215, 255])
    if len(right_hist) > 1:
        for i in range(len(right_hist) - 1):
            p1 = (int(right_hist[i][0]), int(right_hist[i][1]))
            p2 = (int(right_hist[i+1][0]), int(right_hist[i+1][1]))
            alpha = (i + 1) / float(len(right_hist))
            cv2.line(frame, p1, p2, (0, 215, 255), max(1, int(4 * alpha)), cv2.LINE_AA)

def _draw_velocity_vector(frame, ron_p, bio_rec):
    cx = int(ron_p['center'][0])
    cy = int(ron_p['center'][1])
    vel = bio_rec.get('linear_velocity_mps', 7.4)
    # Velocity vector length
    vec_len = int(vel * 8.0)
    angle_rad = math.radians(-15.0)
    end_x = int(cx + vec_len * math.cos(angle_rad))
    end_y = int(cy + vec_len * math.sin(angle_rad))

    cv2.arrowedLine(frame, (cx, cy), (end_x, end_y), (0, 255, 180), 3, cv2.LINE_AA, tipLength=0.25)
    cv2.putText(frame, f"v = {vel:.1f} m/s", (end_x + 10, end_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 180), 1, cv2.LINE_AA)

def _draw_glowing_particles(layer, player, t_sec):
    bbox = player['bbox']
    cx = (bbox[0] + bbox[2]) / 2.0
    cy = (bbox[1] + bbox[3]) / 2.0
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]

    np.random.seed(int(t_sec * 30) % 2000)
    for _ in range(250):
        rx = (np.random.rand() - 0.5) * w * 1.1
        ry = (np.random.rand() - 0.5) * h * 1.1
        turb = math.sin(ry * 0.08 + t_sec * 9.0) * 28.0
        px = int(cx + rx + turb)
        py = int(cy + ry + math.cos(rx * 0.08 + t_sec * 8.0) * 22.0)
        p_rad = np.random.randint(2, 6)
        gold = (np.random.randint(0, 40), np.random.randint(190, 240), np.random.randint(220, 255))
        cv2.circle(layer, (px, py), p_rad, gold, -1, cv2.LINE_AA)

def _draw_depth_contours(frame, ron_p, t_sec):
    if ron_p is None:
        return
    cx, cy = int(ron_p['center'][0]), int(ron_p['center'][1])
    for r in range(20, 90, 18):
        wave = int(math.sin(t_sec * 6.0 + r * 0.1) * 6.0)
        cv2.circle(frame, (cx, cy), r + wave, (255, 200, 0), 1, cv2.LINE_AA)

def _draw_ball_path(frame, ball_hist, ball_rec):
    if len(ball_hist) > 1:
        for b_i in range(len(ball_hist) - 1):
            pt1 = (int(ball_hist[b_i][0]), int(ball_hist[b_i][1]))
            pt2 = (int(ball_hist[b_i+1][0]), int(ball_hist[b_i+1][1]))
            cv2.line(frame, pt1, pt2, (216, 242, 0), 2, cv2.LINE_AA)
    if ball_rec and 'smoothed_pos_2d' in ball_rec:
        bx, by = int(ball_rec['smoothed_pos_2d'][0]), int(ball_rec['smoothed_pos_2d'][1])
        cv2.circle(frame, (bx, by), 7, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, (bx, by), 12, (216, 242, 0), 2, cv2.LINE_AA)

def _apply_chroma_split(frame, shift_px=14):
    b, g, r = cv2.split(frame)
    r_shifted = np.roll(r, shift_px, axis=1)
    b_shifted = np.roll(b, -shift_px, axis=1)
    return cv2.merge([b_shifted, g, r_shifted])

def _apply_letterbox(frame, crop_fraction=0.08):
    h, w = frame.shape[:2]
    bar_h = int(h * crop_fraction)
    cv2.rectangle(frame, (0, 0), (w, bar_h), (0, 0, 0), -1)
    cv2.rectangle(frame, (0, h - bar_h), (w, h), (0, 0, 0), -1)
    return frame

def _render_act_header(frame, title, subtitle, t_sec):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, title, (65, 130), font, 0.78, (15, 15, 15), 4, cv2.LINE_AA)
    cv2.putText(frame, title, (65, 130), font, 0.78, (0, 215, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, subtitle, (65, 160), font, 0.48, (230, 230, 230), 1, cv2.LINE_AA)

def _render_act3_telemetry(frame, bio_rec, ron_p, width, height):
    # HUD Box in top right
    bx1, by1 = width - 430, 40
    bx2, by2 = width - 40, 245
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (18, 20, 24), -1)
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 215, 255), 1, cv2.LINE_AA)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "DRIBBLING TELEMETRY", (bx1 + 18, by1 + 28), font, 0.52, (0, 215, 255), 1, cv2.LINE_AA)

    l_knee = bio_rec.get('left_knee_flexion_deg', 62.0) if bio_rec else 62.0
    r_knee = bio_rec.get('right_knee_flexion_deg', 44.0) if bio_rec else 44.0
    torso = bio_rec.get('torso_twist_deg', 18.5) if bio_rec else 18.5
    vel = bio_rec.get('linear_velocity_mps', 7.4) if bio_rec else 7.4
    acc = bio_rec.get('acceleration_mps2', 2.8) if bio_rec else 2.8

    stats = [
        f"TORSO ROTATION:  {torso:+.1f} deg",
        f"L-KNEE FLEXION:  {l_knee:.1f} deg",
        f"R-KNEE FLEXION:  {r_knee:.1f} deg",
        f"SPRINT VELOCITY: {vel:.2f} m/s ({vel*3.6:.1f} km/h)",
        f"ACCELERATION:    {acc:+.2f} m/s2",
        "BALL PROXIMITY:  0.42 m (CONTROLLED)"
    ]
    for i, s in enumerate(stats):
        cv2.putText(frame, s, (bx1 + 18, by1 + 58 + i * 26), font, 0.42, (240, 240, 240), 1, cv2.LINE_AA)

def _render_master_title_card(frame, alpha, width, height):
    # Overlay dark fade
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, height), (8, 10, 14), -1)
    cv2.addWeighted(overlay, alpha * 0.75, frame, 1.0 - alpha * 0.75, 0, frame)

    font = cv2.FONT_HERSHEY_SIMPLEX
    title = "RONALDINHO — THE ART OF MOTION"
    sub1 = "AI Motion Reconstruction x TouchDesigner"
    sub2 = "Santiago Bernabeu — 19 November 2005"

    t_size = cv2.getTextSize(title, font, 1.25, 3)[0]
    tx = (width - t_size[0]) // 2
    ty = height // 2 - 20

    cv2.putText(frame, title, (tx, ty), font, 1.25, (0, 215, 255), 3, cv2.LINE_AA)
    s1_size = cv2.getTextSize(sub1, font, 0.68, 1)[0]
    cv2.putText(frame, sub1, ((width - s1_size[0]) // 2, ty + 50), font, 0.68, (240, 240, 240), 1, cv2.LINE_AA)
    s2_size = cv2.getTextSize(sub2, font, 0.52, 1)[0]
    cv2.putText(frame, sub2, ((width - s2_size[0]) // 2, ty + 90), font, 0.52, (180, 180, 180), 1, cv2.LINE_AA)

def _create_act_split(orig, rendered, title):
    h, w = orig.shape[:2]
    split = np.zeros_like(orig)
    split[:, :w//2] = orig[:, :w//2]
    split[:, w//2:] = rendered[:, w//2:]
    cv2.line(split, (w//2, 0), (w//2, h), (0, 215, 255), 3, cv2.LINE_AA)
    cv2.putText(split, "ORIGINAL MATCH FOOTAGE", (45, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(split, "AI MOTION RECONSTRUCTION", (w//2 + 45, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 215, 255), 2, cv2.LINE_AA)
    cv2.putText(split, title, (w//2 - 280, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2, cv2.LINE_AA)
    return split

if __name__ == "__main__":
    render_sequence2()
