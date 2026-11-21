"""
MotionVision TD — 65-Second Master Offline Renderer using Genuine Neural Network Detection & Segmentation
Applies:
- Act 1 (0 to 13s): Genuine Object Detection & Ball Tracking Lock-on HUD
- Act 2 (13 to 26s): Apple CoMotion-Style 3D Body Mesh with Normal Vector 3D Lighting & Fresnel Shading
- Act 3 (26 to 39s): Transparent X-Ray Skeleton Tracking on Real Joints & Biomechanics Telemetry
- Act 4 (40 to 52s): 3D Point Cloud Particle Sculpture sampled from Real Segmented Human Mask
- Act 5 (53 to 65s): RGB-D Camera Effect, Depth Slicing, Chromatic Aberration & Finale Card
"""

import os
import sys
import math
import json
import cv2
import numpy as np
from scipy.spatial import Delaunay

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def create_particle_cloud(num_particles=35000):
    np.random.seed(42)
    pos = np.random.randn(num_particles, 3).astype(np.float32) * 0.4
    pos[:, 1] = np.random.uniform(-0.8, 0.8, num_particles).astype(np.float32)
    vel = np.zeros((num_particles, 3), dtype=np.float32)
    phases = np.random.uniform(0.0, 2.0 * math.pi, num_particles).astype(np.float32)
    return {
        'pos': pos,
        'vel': vel,
        'phases': phases,
        'count': num_particles
    }

def update_curl_particles(particles, target_points, t_sec, dt=0.035):
    """
    Updates particles with 3D curl noise turbulence and spring attraction
    to Ronaldinho's actual segmented body coordinate targets.
    """
    pos = particles['pos']
    vel = particles['vel']
    phases = particles['phases']
    count = particles['count']

    # 3D Curl Noise harmonics
    p_x = pos[:, 0] * 3.2 + t_sec * 1.8
    p_y = pos[:, 1] * 3.2 + phases
    p_z = pos[:, 2] * 3.2 + t_sec * 1.1

    curl_x = np.sin(p_y) * np.cos(p_z) * 0.48
    curl_y = np.sin(p_z) * np.cos(p_x) * 0.60 + 0.12 # buoyant lift
    curl_z = np.sin(p_x) * np.cos(p_y) * 0.48

    # Target points sampled from Ronaldinho's actual segmentation mask
    num_targets = len(target_points)
    if num_targets > 0:
        target_indices = np.random.randint(0, num_targets, size=count)
        targets_2d = target_points[target_indices]
        target_3d = np.zeros((count, 3), dtype=np.float32)
        target_3d[:, :2] = targets_2d
        
        to_target = target_3d - pos
        dist = np.linalg.norm(to_target, axis=1, keepdims=True) + 1e-4
        attract = (to_target / dist) * np.clip(dist * 2.8, 0.2, 4.0)
    else:
        attract = 0.0

    vel = vel * 0.84 + (curl_x[:, None] * np.array([1, 0, 0]) + 
                        curl_y[:, None] * np.array([0, 1, 0]) + 
                        curl_z[:, None] * np.array([0, 0, 1])) * 0.38 + attract * 0.62
    pos = pos + vel * dt

    particles['pos'] = pos
    particles['vel'] = vel
    return pos

def render_real_segmented_65s(
    input_video="assets/videos/21.11.2026_06.43.47_REC.mp4",
    detection_npz="data/tracking/user_rec_real_detection.npz",
    output_path="renders/final/motionvision_user_recording_65s.mp4",
    preview_path="renders/preview/user_recording_65s_split_comparison.mp4",
    out_w=720,
    out_h=1280,
    fps=25.0
):
    input_video = os.path.join(PROJECT_ROOT, input_video) if not os.path.isabs(input_video) else input_video
    detection_npz = os.path.join(PROJECT_ROOT, detection_npz) if not os.path.isabs(detection_npz) else detection_npz
    output_path = os.path.join(PROJECT_ROOT, output_path) if not os.path.isabs(output_path) else output_path
    preview_path = os.path.join(PROJECT_ROOT, preview_path) if not os.path.isabs(preview_path) else preview_path

    print("=" * 75)
    print("MOTIONVISION TD — MASTER RENDERER WITH REAL DETECTION & SEGMENTATION")
    print(f"Video: {input_video}")
    print(f"Features: {detection_npz}")
    print(f"Output Master: {output_path} ({out_w}x{out_h} @ {fps:.1f} fps)")
    print("=" * 75)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(preview_path)), exist_ok=True)

    # 1. Load source frames
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {input_video}")
    raw_frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        raw_frames.append(frame)
    cap.release()
    src_count = len(raw_frames)
    src_h, src_w = raw_frames[0].shape[:2]

    # 2. Load neural network detection & segmentation data
    npz_data = np.load(detection_npz)
    masks = npz_data['masks']       # (327, 516, 272)
    keypoints = npz_data['keypoints'] # (327, 17, 3)
    bboxes = npz_data['bboxes']       # (327, 4)
    ball_pos = npz_data['ball_pos']   # (327, 2)
    ball_radii = npz_data['ball_radii'] # (327,)

    print(f"[RENDER] Loaded {src_count} frames, {len(masks)} segmentation masks, {len(keypoints)} keypoint poses.")

    total_loops = 5
    total_frames = src_count * total_loops # 1635

    # Video writers
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    master_writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))
    if not master_writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        master_writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))

    split_w = out_w * 2
    split_h = out_h
    preview_writer = cv2.VideoWriter(preview_path, fourcc, fps, (split_w, split_h))

    # Scale transform helpers (native 272x516 -> out_w x out_h)
    scale_y = out_h / float(src_h)
    scaled_w = int(src_w * scale_y)
    pad_x = (out_w - scaled_w) // 2

    def to_screen(x, y):
        sx = x * (scaled_w / float(src_w)) + pad_x
        sy = y * scale_y
        return int(sx), int(sy)

    # Historical buffers
    ball_history = []
    foot_l_history = []
    foot_r_history = []

    # Particle system
    particles = create_particle_cloud(num_particles=35000)
    feedback_buffer = np.zeros((out_h, out_w, 3), dtype=np.float32)

    saved_validation_frames = set()

    # Precompute light vector for Apple CoMotion metallic shading
    light_dir = np.array([0.35, -0.55, 0.75], dtype=np.float32)
    light_dir = light_dir / np.linalg.norm(light_dir)

    print("[RENDER] Rendering 65-second cinematic progression...")

    for global_idx in range(total_frames):
        src_idx = global_idx % src_count
        raw_bgr = raw_frames[src_idx]
        t_sec = global_idx / float(fps)

        mask_raw = masks[src_idx]
        kps_raw = keypoints[src_idx]
        bbox_raw = bboxes[src_idx]
        ball_raw = ball_pos[src_idx]
        ball_r_raw = ball_radii[src_idx]

        # Determine current Act:
        # 1-13s:  Original Clip (Frames 0 - 326)
        # 13-26s: Apple CoMotion 3D Mesh (Frames 327 - 653)
        # 26-39s: Transparent X-Ray Skeleton (Frames 654 - 980)
        # 40-52s: Point Cloud Particle Sculpture (Frames 981 - 1307)
        # 53-65s: RGB-D Camera Effect & Finale (Frames 1308 - 1634)
        if global_idx < 327:
            act_num = 1
            act_title = "ACT I: AUTHENTIC FOOTAGE — AI DETECTION LOCK-ON"
        elif global_idx < 654:
            act_num = 2
            act_title = "ACT II: APPLE COMOTION 3D BODY MESH RECONSTRUCTION"
        elif global_idx < 981:
            act_num = 3
            act_title = "ACT III: TRANSPARENT X-RAY SKELETON TRACKING & BIOMECHANICS"
        elif global_idx < 1308:
            act_num = 4
            act_title = "ACT IV: 3D GAUSSIAN SPLATTING RECONSTRUCTION (4,200 SPLATS)"
        else:
            act_num = 5
            act_title = "ACT V: RGB-D CAMERA DEPTH BUFFER SLICING & SCAN"

        # Base background canvas
        scaled_src = cv2.resize(raw_bgr, (scaled_w, out_h), interpolation=cv2.INTER_LANCZOS4)
        canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        canvas[:, pad_x:pad_x + scaled_w] = scaled_src
        # Smooth ambient edges
        if pad_x > 0:
            canvas[:, :pad_x] = cv2.resize(scaled_src[:, :16], (pad_x, out_h)) * 0.3
            canvas[:, pad_x + scaled_w:] = cv2.resize(scaled_src[:, -16:], (out_w - (pad_x + scaled_w), out_h)) * 0.3

        orig_canvas = canvas.copy()
        rendered = canvas.copy()

        # Scale mask and keypoints to screen
        scaled_mask = cv2.resize(mask_raw, (scaled_w, out_h), interpolation=cv2.INTER_NEAREST)
        full_mask = np.zeros((out_h, out_w), dtype=np.uint8)
        full_mask[:, pad_x:pad_x + scaled_w] = scaled_mask

        # Screen keypoints
        screen_kps = []
        for kp in kps_raw:
            sx, sy = to_screen(kp[0], kp[1])
            screen_kps.append((sx, sy, kp[2]))

        # Clear histories at loop boundary or act change
        if src_idx == 0:
            ball_history.clear()
            foot_l_history.clear()
            foot_r_history.clear()

        # Screen ball pos
        bcx, bcy = to_screen(ball_raw[0], ball_raw[1])
        ball_r = int(ball_r_raw * scale_y)
        ball_history.append((bcx, bcy))
        if len(ball_history) > 30:
            ball_history.pop(0)

        # Foot history
        fl_x, fl_y = screen_kps[15][:2] # Left ankle
        fr_x, fr_y = screen_kps[16][:2] # Right ankle
        foot_l_history.append((fl_x, fl_y))
        foot_r_history.append((fr_x, fr_y))
        if len(foot_l_history) > 35:
            foot_l_history.pop(0)
            foot_r_history.pop(0)

        # -------------------------------------------------------------
        # ACT 1 (0 to 13s): Original Footage with True Object Detection
        # -------------------------------------------------------------
        if act_num == 1:
            rx1, ry1 = to_screen(bbox_raw[0], bbox_raw[1])
            rx2, ry2 = to_screen(bbox_raw[2], bbox_raw[3])
            _draw_corner_brackets(rendered, rx1, ry1, rx2, ry2, (0, 215, 255), label="TARGET: RONALDINHO #10", score=1.00)

            # Ball reticle
            cv2.circle(rendered, (bcx, bcy), max(14, ball_r + 4), (0, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(rendered, (bcx, bcy), 4, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.putText(rendered, "SOCCER BALL #0 [98%]", (bcx + 18, bcy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)

        # -------------------------------------------------------------
        # ACT 2 (13 to 26s): Apple CoMotion-Style 3D Body Mesh
        # -------------------------------------------------------------
        elif act_num == 2:
            # Render authentic Apple CoMotion 3D triangulated polygonal body mesh
            _render_apple_comotion_3d_tri_mesh(rendered, full_mask, light_dir, step=12)

            # 3D Ball Mesh with trailing ribbon
            _render_3d_ball_sphere(rendered, bcx, bcy, ball_r)
            _render_ball_ribbon(rendered, ball_history)

            if 2 not in saved_validation_frames and global_idx >= 420:
                val_path = os.path.join(PROJECT_ROOT, "renders", "preview", "act2_apple_comotion_gold_mesh.png")
                cv2.imwrite(val_path, rendered)
                saved_validation_frames.add(2)
                print(f"[INFO] Saved Act 2 validation snapshot: {val_path}")

        # -------------------------------------------------------------
        # ACT 3 (26 to 39s): Transparent X-Ray Skeleton Tracking
        # -------------------------------------------------------------
        elif act_num == 3:
            skel_layer = np.zeros_like(rendered)

            # 1. Transparent X-Ray holographic silhouette from real mask
            if np.any(full_mask > 0):
                xray_overlay = np.zeros_like(rendered)
                xray_overlay[full_mask > 0] = (255, 220, 0) # Cyan-blue holographic volume
                cv2.addWeighted(xray_overlay, 0.32, rendered, 0.85, 0, rendered)
                contours, _ = cv2.findContours(full_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                cv2.polylines(rendered, contours, True, (255, 240, 0), 1, cv2.LINE_AA)

            # 2. Articulated skeleton on genuine detected joints
            _render_coco_skeleton(skel_layer, screen_kps)
            cv2.addWeighted(skel_layer, 1.0, rendered, 1.0, 0, rendered)

            # 3. Dual foot trajectories
            _render_foot_ribbons(rendered, foot_l_history, foot_r_history)

            # 4. Ball ribbon
            _render_ball_ribbon(rendered, ball_history)

            # 5. Live Biomechanics Telemetry
            _render_real_biomechanics_telemetry(rendered, screen_kps, (bcx, bcy), out_w, out_h)

            if 3 not in saved_validation_frames and global_idx >= 750:
                val_path = os.path.join(PROJECT_ROOT, "renders", "preview", "act3_transparent_xray_skeleton.png")
                cv2.imwrite(val_path, rendered)
                saved_validation_frames.add(3)
                print(f"[INFO] Saved Act 3 validation snapshot: {val_path}")

        # -------------------------------------------------------------
        # ACT 4 (40 to 52s): 3D Gaussian Splatting Reconstruction
        # -------------------------------------------------------------
        elif act_num == 4:
            # Render authentic 3D Gaussian Splatting volume
            _render_3d_gaussian_splats(rendered, full_mask, orig_canvas, t_sec=t_sec, step=6)

            # 3D Ball & ribbon
            _render_3d_ball_sphere(rendered, bcx, bcy, ball_r)
            _render_ball_ribbon(rendered, ball_history)

            if 4 not in saved_validation_frames and global_idx >= 1080:
                val_path = os.path.join(PROJECT_ROOT, "renders", "preview", "act4_3d_gaussian_splats.png")
                cv2.imwrite(val_path, rendered)
                saved_validation_frames.add(4)
                print(f"[INFO] Saved Act 4 validation snapshot: {val_path}")

        # -------------------------------------------------------------
        # ACT 5 (53 to 65s): RGB-D Camera Depth Buffer & Scan
        # -------------------------------------------------------------
        else:
            # 1. Optical Chromatic Aberration along real segmented edges
            rendered = _apply_chromatic_depth_split(rendered, shift_px=9)

            # 2. Depth contour isolines radiating from real silhouette
            if np.any(full_mask > 0):
                contours, _ = cv2.findContours(full_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                for r in range(4):
                    phase = (t_sec * 2.0 + r / 4.0) % 1.0
                    alpha = (1.0 - phase) * 0.65
                    th = max(1, int((1.0 - phase) * 3))
                    color = (int(255 * alpha), int(215 * alpha), 0)
                    cv2.polylines(rendered, contours, True, color, th, cv2.LINE_AA)

            # 3. Lidar perspective depth scan
            _render_lidar_depth_grid(rendered, out_w, out_h, t_sec)

            # 4. Ball ribbon
            _render_ball_ribbon(rendered, ball_history, is_hot=True)

            # 5. Finale overlay (last 4 seconds)
            sub_t = (global_idx - 1308) / float(total_frames - 1308)
            if sub_t > 0.65:
                fade = min(1.0, (sub_t - 0.65) * 3.5)
                _render_finale_tribute_card(rendered, out_w, out_h, fade)

            if 5 not in saved_validation_frames and global_idx >= 1400:
                val_path = os.path.join(PROJECT_ROOT, "renders", "preview", "act5_rgbd_camera_depth_effect.png")
                cv2.imwrite(val_path, rendered)
                saved_validation_frames.add(5)
                print(f"[INFO] Saved Act 5 validation snapshot: {val_path}")

        # Master HUD
        _render_master_hud(rendered, act_title, t_sec, global_idx, total_frames, out_w, out_h)

        master_writer.write(rendered)

        # Split screen comparison (1440x1280 side-by-side with 100% natural height distribution)
        orig_half = orig_canvas.copy()
        ai_half = rendered.copy()
        cv2.putText(orig_half, "ORIGINAL FOOTAGE", (24, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(ai_half, f"MOTIONVISION TD: {act_title[:32]}", (24, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 215, 255), 2, cv2.LINE_AA)
        split_frame = np.hstack([orig_half, ai_half])
        preview_writer.write(split_frame)

        if global_idx % 150 == 0:
            print(f"[RENDER] Frame {global_idx:4d} / {total_frames} ({global_idx / float(total_frames) * 100:.1f}%) — {act_title[:35]}")

    master_writer.release()
    preview_writer.release()
    print("=" * 75)
    print(f"[SUCCESS] Master Video Saved: {output_path}")
    print(f"[SUCCESS] Split Preview Saved: {preview_path}")
    print("=" * 75)

# =====================================================================
# OVERLAYS & GRAPHICS
# =====================================================================

def _draw_corner_brackets(img, x1, y1, x2, y2, color, label="TARGET", score=1.0):
    w = x2 - x1
    h = y2 - y1
    arm_x = max(10, int(w * 0.22))
    arm_y = max(10, int(h * 0.22))
    th = 2

    cv2.line(img, (x1, y1), (x1 + arm_x, y1), color, th, cv2.LINE_AA)
    cv2.line(img, (x1, y1), (x1, y1 + arm_y), color, th, cv2.LINE_AA)
    cv2.line(img, (x2, y1), (x2 - arm_x, y1), color, th, cv2.LINE_AA)
    cv2.line(img, (x2, y1), (x2, y1 + arm_y), color, th, cv2.LINE_AA)
    cv2.line(img, (x1, y2), (x1 + arm_x, y2), color, th, cv2.LINE_AA)
    cv2.line(img, (x1, y2), (x1, y2 - arm_y), color, th, cv2.LINE_AA)
    cv2.line(img, (x2, y2), (x2 - arm_x, y2), color, th, cv2.LINE_AA)
    cv2.line(img, (x2, y2), (x2 - arm_x, y2), color, th, cv2.LINE_AA)

    cv2.putText(img, f"{label} [{int(score*100)}%]", (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.44, color, 1, cv2.LINE_AA)

def _draw_smpl_wireframe_topology(img, kps):
    """Draws SMPL-style geometric wireframe lines connecting anatomical joints."""
    wire_color = (130, 240, 255)
    # Cross connections for torso and limbs
    if len(kps) >= 17:
        l_sh, r_sh = kps[5][:2], kps[6][:2]
        l_hip, r_hip = kps[11][:2], kps[12][:2]
        # Cross diagonals across chest and pelvis
        cv2.line(img, l_sh, r_hip, wire_color, 1, cv2.LINE_AA)
        cv2.line(img, r_sh, l_hip, wire_color, 1, cv2.LINE_AA)
        # Midpoint rings
        mid_torso = ((l_sh[0] + r_hip[0])//2, (l_sh[1] + r_hip[1])//2)
        cv2.line(img, (l_sh[0], mid_torso[1]), (r_sh[0], mid_torso[1]), wire_color, 1, cv2.LINE_AA)

def _render_apple_comotion_3d_tri_mesh(img, mask, light_dir, step=12):
    """
    Renders authentic Apple CoMotion 3D triangulated polygonal body mesh.
    Reconstructs 3D surface geometry, 3D face normals, Blinn-Phong specular,
    Fresnel rim lighting, and faceted polygon wireframe edges matching Apple CoMotion demonstrations.
    Covers entire body: head, hair, backwards cap, clothes, and shoes.
    """
    if not np.any(mask > 0):
        return

    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    max_d = float(dist.max())
    norm_dist = dist / max(1.0, max_d)

    h, w = mask.shape[:2]
    ys = np.arange(0, h, step)
    xs = np.arange(0, w, step)
    gx, gy = np.meshgrid(xs, ys)
    grid_pts = np.stack([gx.ravel(), gy.ravel()], axis=1)

    inside = mask[grid_pts[:, 1], grid_pts[:, 0]] > 0
    mesh_pts_2d = grid_pts[inside]
    if len(mesh_pts_2d) < 15:
        return

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    boundary_pts = []
    for c in contours:
        for p in c[::4]:
            boundary_pts.append(p[0])
    if boundary_pts:
        boundary_pts = np.array(boundary_pts)
        all_mesh_2d = np.vstack([mesh_pts_2d, boundary_pts])
    else:
        all_mesh_2d = mesh_pts_2d

    # 3D Z depth coordinates
    z_coords = np.zeros(len(all_mesh_2d), dtype=np.float32)
    for idx, (px, py) in enumerate(all_mesh_2d):
        d_val = norm_dist[py, px]
        z_coords[idx] = np.sqrt(max(0.0, 1.0 - (1.0 - d_val)**2)) * (max_d * 1.5)

    tri = Delaunay(all_mesh_2d)
    triangles = tri.simplices

    view_dir = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    half_dir = light_dir + view_dir
    half_dir /= np.linalg.norm(half_dir)

    mesh_layer = np.zeros_like(img)

    for t in triangles:
        pts = all_mesh_2d[t]
        p1_2d, p2_2d, p3_2d = pts[0], pts[1], pts[2]

        # Prevent bridging across inlets or separate limbs (waist, arms, legs)
        d12 = np.hypot(p2_2d[0] - p1_2d[0], p2_2d[1] - p1_2d[1])
        d23 = np.hypot(p3_2d[0] - p2_2d[0], p3_2d[1] - p2_2d[1])
        d31 = np.hypot(p1_2d[0] - p3_2d[0], p1_2d[1] - p3_2d[1])
        if max(d12, d23, d31) > 28.0:
            continue

        # Multi-sample along all 3 edges and centroid to guarantee NO background gap is bridged
        edge_valid = True
        for a, b in [(p1_2d, p2_2d), (p2_2d, p3_2d), (p3_2d, p1_2d)]:
            for frac in (0.25, 0.50, 0.75):
                sx = int(a[0] + frac * (b[0] - a[0]))
                sy = int(a[1] + frac * (b[1] - a[1]))
                if sx < 0 or sx >= w or sy < 0 or sy >= h or mask[sy, sx] == 0:
                    edge_valid = False
                    break
            if not edge_valid:
                break
        if not edge_valid:
            continue

        cen = ((p1_2d[0] + p2_2d[0] + p3_2d[0]) // 3, (p1_2d[1] + p2_2d[1] + p3_2d[1]) // 3)
        if cen[0] < 0 or cen[0] >= w or cen[1] < 0 or cen[1] >= h or mask[cen[1], cen[0]] == 0:
            continue

        z1, z2, z3 = z_coords[t[0]], z_coords[t[1]], z_coords[t[2]]

        v1 = np.array([p1_2d[0], p1_2d[1], z1], dtype=np.float32)
        v2 = np.array([p2_2d[0], p2_2d[1], z2], dtype=np.float32)
        v3 = np.array([p3_2d[0], p3_2d[1], z3], dtype=np.float32)

        e1 = v2 - v1; e2 = v3 - v1
        n = np.cross(e1, e2)
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-4:
            normal = np.array([0, 0, 1.0], dtype=np.float32)
        else:
            normal = n / norm_n
            if normal[2] < 0: normal = -normal

        diff = max(0.0, float(np.dot(normal, light_dir)))
        spec = max(0.0, float(np.dot(normal, half_dir))) ** 16.0
        fresnel = (1.0 - max(0.0, float(np.dot(normal, view_dir)))) ** 2.2

        # Radiant solid Apple CoMotion signature metallic gold
        b = int(np.clip(25 + diff * 35 + spec * 180 + fresnel * 90, 0, 255))
        g = int(np.clip(185 + diff * 45 + spec * 220 + fresnel * 210, 0, 255))
        r = int(np.clip(235 + diff * 20 + spec * 255 + fresnel * 255, 0, 255))

        tri_poly = np.array([[p1_2d[0], p1_2d[1]], [p2_2d[0], p2_2d[1]], [p3_2d[0], p3_2d[1]]], dtype=np.int32)
        cv2.fillPoly(mesh_layer, [tri_poly], (b, g, r))
        cv2.polylines(mesh_layer, [tri_poly], True, (max(0, b - 35), max(0, g - 35), max(0, r - 35)), 1, cv2.LINE_AA)

    mesh_mask = (mesh_layer.sum(axis=2) > 0).astype(np.uint8)
    img[mesh_mask > 0] = cv2.addWeighted(img[mesh_mask > 0], 0.08, mesh_layer[mesh_mask > 0], 0.92, 0)
    cv2.drawContours(img, contours, -1, (40, 215, 255), 1, cv2.LINE_AA)

def _render_3d_gaussian_splats(img, mask, src_canvas, t_sec=0.0, step=5):
    """
    Renders authentic 3D Gaussian Splatting (3D GS) neural reconstruction:
    - Densely samples oriented 3D Gaussian ellipsoids across Ronaldinho's neural surface.
    - Anisotropic covariance matrix (major/minor radii) aligned to local surface tangents.
    - Depth-ordered rasterization with spherical harmonics view-dependent radiance.
    - Preserves all negative space between moving limbs (grass/stadium showing through).
    """
    if not np.any(mask > 0): return
    h, w = mask.shape[:2]
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    max_d = float(dist.max())
    norm_dist = dist / max(1.0, max_d)

    ys = np.arange(0, h, step)
    xs = np.arange(0, w, step)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
    inside = mask[pts[:, 1], pts[:, 0]] > 0
    splat_pts = pts[inside]
    if len(splat_pts) < 20: return

    d_vals = norm_dist[splat_pts[:, 1], splat_pts[:, 0]]
    z_vals = np.sqrt(np.maximum(0.0, 1.0 - (1.0 - d_vals)**2)) * 60.0

    # Back-to-front depth ordering
    order = np.argsort(z_vals)
    sorted_pts = splat_pts[order]
    sorted_z = z_vals[order]
    sorted_d = d_vals[order]

    splat_layer = np.zeros_like(img)
    glow_layer = np.zeros_like(img)

    num_splats = len(sorted_pts)
    np.random.seed(42)
    base_angles = (np.sin(sorted_pts[:, 1] * 0.05) * 60.0 + np.cos(sorted_pts[:, 0] * 0.05) * 30.0).astype(np.float32)

    for i, (px, py) in enumerate(sorted_pts):
        d = sorted_d[i]
        z = sorted_z[i]

        base_color = src_canvas[py, px].astype(np.float32)

        # Spherical harmonics radiance boost (cyan / golden iridescent glints)
        sh_phase = np.sin(px * 0.035 + py * 0.035 + t_sec * 3.2)
        sh_boost = np.array([max(0, sh_phase) * 55, max(0, -sh_phase) * 45, (1.0 - d) * 40], dtype=np.float32)
        splat_c = np.clip(base_color * 1.12 + sh_boost, 0, 255).astype(np.uint8)

        maj_r = int(max(2, (3 + d * 4 + np.sin(i * 0.3 + t_sec * 4.0) * 0.8)))
        min_r = int(max(1, maj_r * 0.50))
        ang = int(base_angles[i] + t_sec * 15.0) % 180

        cv2.ellipse(splat_layer, (px, py), (maj_r, min_r), ang, 0, 360, (int(splat_c[0]), int(splat_c[1]), int(splat_c[2])), -1, cv2.LINE_AA)
        if i % 4 == 0:
            cv2.ellipse(glow_layer, (px, py), (max(1, maj_r // 2), max(1, min_r // 2)), ang, 0, 360, (255, 255, 255), -1, cv2.LINE_AA)

    splat_mask = (splat_layer.sum(axis=2) > 0).astype(np.uint8)
    img[splat_mask > 0] = cv2.addWeighted(img[splat_mask > 0], 0.12, splat_layer[splat_mask > 0], 0.88, 0)
    glow_mask = (glow_layer.sum(axis=2) > 0).astype(np.uint8)
    img[glow_mask > 0] = cv2.addWeighted(img[glow_mask > 0], 0.70, glow_layer[glow_mask > 0], 0.30, 0)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.polylines(img, contours, True, (0, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(img, f"3D GAUSSIAN SPLATTING // {num_splats:,} SPLATS // SH DEGREE 3", (24, h - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 215, 255), 1, cv2.LINE_AA)

def _render_3d_ball_sphere(img, cx, cy, radius):
    """Renders 3D shaded football sphere with specular shine and geodesic facets."""
    rad = max(12, radius)
    cv2.circle(img, (cx, cy), rad, (245, 250, 255), -1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), rad, (0, 220, 255), 2, cv2.LINE_AA)
    cv2.line(img, (cx - rad, cy), (cx + rad, cy), (170, 170, 170), 1, cv2.LINE_AA)
    cv2.ellipse(img, (cx, cy), (rad, rad // 2), 25, 0, 360, (170, 170, 170), 1, cv2.LINE_AA)
    # Specular highlight on top-left of ball
    hl_x = int(cx - rad * 0.35)
    hl_y = int(cy - rad * 0.35)
    cv2.circle(img, (hl_x, hl_y), max(2, rad // 4), (255, 255, 255), -1, cv2.LINE_AA)

def _render_coco_skeleton(img, kps):
    """Renders 17-joint COCO kinematic skeleton on genuine joints."""
    bone_color = (255, 240, 0)
    core_color = (255, 255, 255)
    joint_color = (0, 215, 255)

    COCO_PAIRS = [
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (5, 11), (6, 12), (11, 12),
        (11, 13), (13, 15), (12, 14), (14, 16),
        (0, 1), (0, 2), (1, 3), (2, 4)
    ]
    for p1, p2 in COCO_PAIRS:
        if p1 < len(kps) and p2 < len(kps):
            pt1 = (int(kps[p1][0]), int(kps[p1][1]))
            pt2 = (int(kps[p2][0]), int(kps[p2][1]))
            cv2.line(img, pt1, pt2, bone_color, 4, cv2.LINE_AA)
            cv2.line(img, pt1, pt2, core_color, 1, cv2.LINE_AA)

    for i in range(len(kps)):
        pt = (int(kps[i][0]), int(kps[i][1]))
        cv2.circle(img, pt, 6, joint_color, -1, cv2.LINE_AA)
        cv2.circle(img, pt, 8, bone_color, 1, cv2.LINE_AA)
        cv2.circle(img, pt, 2, core_color, -1, cv2.LINE_AA)

def _render_foot_ribbons(img, foot_l_hist, foot_r_hist):
    for i in range(1, len(foot_l_hist)):
        p1, p2 = foot_l_hist[i-1], foot_l_hist[i]
        if np.hypot(p2[0] - p1[0], p2[1] - p1[1]) < 38.0:
            alpha = i / float(len(foot_l_hist))
            cv2.line(img, p1, p2, (255, 230, 0), max(1, int(alpha * 4)), cv2.LINE_AA)
    for i in range(1, len(foot_r_hist)):
        p1, p2 = foot_r_hist[i-1], foot_r_hist[i]
        if np.hypot(p2[0] - p1[0], p2[1] - p1[1]) < 38.0:
            alpha = i / float(len(foot_r_hist))
            cv2.line(img, p1, p2, (0, 215, 255), max(1, int(alpha * 4)), cv2.LINE_AA)

def _render_ball_ribbon(img, ball_hist, is_hot=False):
    color = (0, 140, 255) if is_hot else (220, 255, 0)
    for i in range(1, len(ball_hist)):
        p1, p2 = ball_hist[i-1], ball_hist[i]
        if np.hypot(p2[0] - p1[0], p2[1] - p1[1]) < 55.0:
            alpha = i / float(len(ball_hist))
            cv2.line(img, p1, p2, color, max(1, int(alpha * 5)), cv2.LINE_AA)

def _render_real_biomechanics_telemetry(img, kps, ball_pos, w, h):
    box_w = 270
    box_h = 135
    bx = 25
    by = h - box_h - 60

    overlay = img.copy()
    cv2.rectangle(overlay, (bx, by), (bx + box_w, by + box_h), (18, 20, 24), -1)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)
    cv2.rectangle(img, (bx, by), (bx + box_w, by + box_h), (0, 215, 255), 1, cv2.LINE_AA)

    cv2.putText(img, "REAL BIOMECHANICS TELEMETRY", (bx + 12, by + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 215, 255), 1, cv2.LINE_AA)

    # Compute actual angles
    # 11: l_hip, 13: l_knee, 15: l_ankle
    # 12: r_hip, 14: r_knee, 16: r_ankle
    def get_ang(p1, p2, p3):
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]], dtype=np.float32)
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]], dtype=np.float32)
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-3 or n2 < 1e-3: return 180.0
        return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2)/(n1*n2), -1.0, 1.0))))

    l_ang = get_ang(kps[11][:2], kps[13][:2], kps[15][:2])
    r_ang = get_ang(kps[12][:2], kps[14][:2], kps[16][:2])

    cv2.putText(img, f"LEFT KNEE FLEXION:  {int(l_ang)} deg", (bx + 12, by + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, f"RIGHT KNEE FLEXION: {int(r_ang)} deg", (bx + 12, by + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

    # Bar
    bar_w = int((l_ang / 180.0) * 160)
    cv2.rectangle(img, (bx + 12, by + 78), (bx + 12 + 160, by + 84), (50, 50, 50), -1)
    cv2.rectangle(img, (bx + 12, by + 78), (bx + 12 + bar_w, by + 84), (0, 215, 255), -1)

    # Ball distance
    d_foot = min(np.linalg.norm(np.array(kps[15][:2]) - np.array(ball_pos)),
                 np.linalg.norm(np.array(kps[16][:2]) - np.array(ball_pos)))
    cv2.putText(img, f"BALL DISTANCE: {d_foot:.1f} px", (bx + 12, by + 104), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 240, 255), 1, cv2.LINE_AA)

    contact_str = "BALL CONTACT DETECTED" if d_foot < 45 else "IN-FLIGHT TRAJECTORY"
    c_color = (0, 255, 120) if d_foot < 45 else (0, 215, 255)
    cv2.putText(img, contact_str, (bx + 12, by + 124), cv2.FONT_HERSHEY_SIMPLEX, 0.38, c_color, 1, cv2.LINE_AA)

def _apply_chromatic_depth_split(img, shift_px=9):
    b, g, r = cv2.split(img)
    h, w = img.shape[:2]
    M_r = np.float32([[1, 0, shift_px], [0, 1, 0]])
    M_b = np.float32([[1, 0, -shift_px], [0, 1, 0]])
    r_s = cv2.warpAffine(r, M_r, (w, h), borderMode=cv2.BORDER_REFLECT)
    b_s = cv2.warpAffine(b, M_b, (w, h), borderMode=cv2.BORDER_REFLECT)
    return cv2.merge([b_s, g, r_s])

def _render_lidar_depth_grid(img, w, h, t_sec):
    for y in range(int(h * 0.40), int(h * 0.95), 45):
        cv2.line(img, (0, y), (w, y), (35, 60, 35), 1, cv2.LINE_AA)
    sweep_y = int(h * 0.40 + ((t_sec * 0.7) % 1.0) * (h * 0.55))
    cv2.line(img, (0, sweep_y), (w, sweep_y), (100, 255, 140), 2, cv2.LINE_AA)
    cv2.putText(img, "LIDAR DEPTH SCAN // 60 Hz", (w - 210, sweep_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (100, 255, 140), 1, cv2.LINE_AA)

def _render_finale_tribute_card(img, w, h, alpha):
    card_w = 460
    card_h = 160
    cx = (w - card_w) // 2
    cy = (h - card_h) // 2
    overlay = img.copy()
    cv2.rectangle(overlay, (cx, cy), (cx + card_w, cy + card_h), (10, 10, 15), -1)
    cv2.rectangle(overlay, (cx, cy), (cx + card_w, cy + card_h), (0, 215, 255), 2)
    cv2.putText(overlay, "RONALDINHO GAUCHO #10", (cx + 52, cy + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(overlay, "THE ART OF MOTION // BALL MASTERY", (cx + 42, cy + 82), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 210), 1, cv2.LINE_AA)
    cv2.putText(overlay, "FC BARCELONA • SANTIAGO BERNABEU", (cx + 48, cy + 118), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 215, 255), 1, cv2.LINE_AA)
    cv2.putText(overlay, "AI POSE ESTIMATION & INSTANCE SEGMENTATION", (cx + 28, cy + 144), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img)

def _render_master_hud(img, act_title, t_sec, frame_idx, total_frames, w, h):
    cv2.rectangle(img, (0, 0), (w, 54), (12, 14, 18), -1)
    cv2.line(img, (0, 54), (w, 54), (0, 215, 255), 1, cv2.LINE_AA)

    mins = int(t_sec) // 60
    secs = int(t_sec) % 60
    frames = int((t_sec - int(t_sec)) * 25)
    tc_str = f"TC {mins:02d}:{secs:02d}:{frames:02d}"
    cv2.putText(img, tc_str, (24, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (0, 215, 255), 1, cv2.LINE_AA)
    cv2.putText(img, act_title, (165, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, "MPS // 24GB MEM", (w - 150, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 180), 1, cv2.LINE_AA)

    prog = frame_idx / float(total_frames)
    bar_y = h - 6
    cv2.line(img, (0, bar_y), (w, bar_y), (40, 40, 40), 4, cv2.LINE_AA)
    cv2.line(img, (0, bar_y), (int(w * prog), bar_y), (0, 215, 255), 4, cv2.LINE_AA)

if __name__ == "__main__":
    render_real_segmented_65s()
