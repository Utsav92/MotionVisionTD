# MOTIONVISION TD — RONALDINHO: THE ART OF MOTION

> **A Cinematic Deep Learning & 3D Motion Visualization in TouchDesigner**  
> *Recreating Ronaldinho's Legendary Bernabéu Performance (November 19, 2005) with Multi-Person 3D Motion Reconstruction, Articulated Biomechanics, GPU Particle Sculptures, and Broadcast Spatial Analytics.*

---

## 1. Project Overview

On November 19, 2005, Ronaldinho Gaúcho delivered one of the most breathtaking individual performances in football history at the Santiago Bernabéu, leading FC Barcelona to a 3–0 victory over Real Madrid and receiving a rare standing ovation from the rival Madrid supporters.

**MotionVision TD** is an advanced computer vision and creative technology platform that transforms this match footage into a 60-second multi-dimensional scientific art installation. Inspired by Apple's **CoMotion** multi-person 3D motion reconstruction research, the system tracks players and the football across broadcast cameras, fits articulated SMPL human body meshes, computes joint biomechanics, and renders six distinct visual modes inside TouchDesigner.

```
       [ MATCH VIDEO ] ────────► [ AI INFERENCE PIPELINE ]
  (1080p 30fps Bernabéu)              │
                                      ├── Multi-Person Detection & Tracking (YOLOv8 / CV)
                                      ├── Persistent Identity Classification (Ronaldinho #10)
                                      ├── High-Res Football Trajectory & Occlusion Recovery
                                      ├── Apple CoMotion / SMPL 3D Reconstruction (24 joints)
                                      ├── Broadcast Camera Calibration (Pan, Tilt, Zoom, FOV)
                                      └── Biomechanics Engine (Velocity, Acceleration, Angles)
                                      │
                                      ▼
                            [ STRUCTURED CACHE ]
                    (motion_data.npz, tracking.json, OBJ)
                                      │
                                      ▼
                       [ TOUCHDESIGNER PROJECT (.toe) ]
                        /project1 (17 Modular COMPs)
                                      │
       ┌───────────┬───────────┬──────┴────┬───────────┬───────────┐
     Mode A      Mode B      Mode C      Mode D      Mode E      Mode F
    Colored      3D X-Ray    Particle    RGB-D       Motion      Tactical
     Meshes      Skeleton   Turbulence  Hologram     Trails     Analytics
       └───────────┴───────────┬───────────┴───────────┴───────────┘
                               │
                               ▼
                   [ 60s CINEMATIC MASTER MP4 ]
```

---

## 2. The Six Visual Modes

| Mode | Title | Visual Description | Color Schema |
|---|---|---|---|
| **Mode A** | **Colored Human Meshes** | Apple CoMotion multi-person 3D mesh overlay with camera-aligned articulated geometry. | Gold (`#FFD700`) for Ronaldinho, Neon Crimson (`#FF3860`) for Real Madrid defenders, Electric Blue (`#007BFF`) for Barcelona teammates, Purple (`#A640F3`) for referee. |
| **Mode B** | **3D Skeleton & Biomechanics** | Articulated 24-joint kinematic bone tubes, sphere joints, knee flexion angles, hip rotation, velocity vectors, and center-of-mass proxy. | High-contrast gold & white bones with biometric HUD telemetry. |
| **Mode C** | **Particle Body Transformation** | Ronaldinho's body transforms from solid mesh into an expanding 3D fluid point-cloud sculpture via GPU curl noise turbulence and reassembly. | Glowing gold gradient points with Gaussian softness. |
| **Mode D** | **RGB-D Chroma Hologram** | Chromatic aberration channel separation (R/G/B shifts), holographic depth slicing, and scanline glitch during Ronaldinho's evasive dribbles. | Separated Red/Blue channels with holographic cyan depth contours. |
| **Mode E** | **Motion Trails & Feedback** | Directional temporal velocity blur, ghost delayed silhouettes, and glowing ribbons tracing body and ball paths. | Feedback decay persistence with turquoise football trajectory. |
| **Mode F** | **Football Analytics HUD** | Broadcast sports science overlay: estimated speed ($m/s, km/h$), acceleration ($m/s^2$), defender proximity ($d_{min}$ in meters), and camera-space heatmaps. | Technical HUD data graphics with explicit measurement units. |

---

## 3. The 60-Second Cinematic Storyboard

- **Scene 1 (00:00–00:08) — *THE ART OF MOTION***: Original broadcast footage, restrained glow, and match-identifying typography.
- **Scene 2 (00:08–00:18) — *FOOTBALL BECOMES DATA***: Detection bounding boxes transition all players into colored 3D human meshes (CoMotion style).
- **Scene 3 (00:18–00:30) — *THE ANATOMY OF A DRIBBLE***: Iconic stepover feint past Sergio Ramos; Ronaldinho transitions into a 3D skeleton revealing joint angles and acceleration bursts.
- **Scene 4 (00:30–00:40) — *MOTION IN FOUR DIMENSIONS***: Penalty box penetration; Ronaldinho disintegrates into 25,000 glowing fluid particles before reassembling.
- **Scene 5 (00:40–00:50) — *THE INTELLIGENCE OF MOVEMENT***: Near-post strike past Casillas; telemetry displays ball trajectory, velocity ($8.2\text{ m/s}$), and defender proximity.
- **Scene 6 (00:50–01:00) — *THE ART OF FOOTBALL***: Bernabéu standing ovation and samba celebration with chromatic RGB-D hologram warp. Concludes with the master title:  
  *RONALDINHO: THE ART OF MOTION — AI Motion Reconstruction × TouchDesigner*.

---

## 4. Hardware Optimization (Apple Silicon & Cross-Platform)

- **Target Architecture**: Apple Silicon Mac (M2/M3/M4, 24 GB Unified Memory).
- **Decoupled Architecture**: AI preprocessing runs out-of-process, writing binary `.npz` and JSON caches. TouchDesigner's render loop performs zero real-time neural network inference.
- **Unified Memory Budget**: AI inference capped at 8 GB; TouchDesigner allocation capped at 10 GB; leaving 6 GB for macOS and Metal framebuffers.
- **Deterministic Frame Cooking**: Frame-stepping engine guarantees jitter-free 30 FPS offline rendering.

---

## 5. Quickstart & Execution Commands

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run Automated Tests
```bash
pytest -c pytest.ini tests/
```
*(All 13 unit and integration tests validate tracking, ball interpolation, 3D geometry, synchronization, and rendering).*

### Step 3: Run the AI Inference & Motion Reconstruction Pipeline
```bash
# Processes source video and generates all motion caches:
python -m ai.export_motion assets/videos/ronaldinho_bernabeu_2005.mp4 1800
```

### Step 4: Validate TouchDesigner Network Specifications
```bash
python touchdesigner/build_project.py
```

### Step 5: Render the Master 60-Second Video
```bash
python touchdesigner/offline_renderer.py
```

---

## 6. Project File Structure

```
MotionVisionTD/
├── README.md                      # Complete system documentation
├── LICENSE_NOTES.md               # Copyright and attribution compliance
├── requirements.txt               # Python package dependencies
├── pytest.ini                     # Localized test runner configuration
├── config/
│   ├── project.yaml               # Master project configuration
│   ├── render.yaml                # Presets (Preview, Standard, Cinematic, 4K)
│   └── sources.json               # Video source selection & shot boundaries
├── assets/
│   └── videos/
│       ├── ronaldinho_bernabeu_2005.mp4   # Master 1080p 30fps source video
│       └── create_test_footage.py         # Procedural broadcast test generator
├── ai/
│   ├── comotion_adapter.py        # Apple ml-comotion bridge & hardware dispatcher
│   ├── detection.py               # Multi-object detection (YOLOv8 + CV fallback)
│   ├── tracking.py                # ByteTrack multi-person tracker with Kalman motion
│   ├── player_identity.py         # Identity categorization (Ronaldinho, Madrid, Barca)
│   ├── ball_tracking.py           # Football tracker with occlusion interpolation
│   ├── mesh_reconstruction.py     # 24-joint SMPL kinematics & template OBJ generator
│   ├── camera_estimation.py       # Broadcast camera intrinsics/extrinsics & reprojection
│   ├── motion_metrics.py          # Biomechanics: velocity, acceleration, joint angles
│   └── export_motion.py           # Master AI pipeline CLI runner
├── data/
│   ├── tracking/                  # tracking_data.json, ball_trajectory.json
│   ├── poses/                     # motion_data.npz (binary channels for TD)
│   ├── meshes/                    # smpl_template.obj (articulated 3D mesh)
│   └── metadata/                  # camera_data.json, biomechanics.json, scene_manifest.json
├── touchdesigner/
│   ├── MotionVision_Ronaldinho.toe # Working TouchDesigner project file
│   ├── build_project.py           # Programmatic network construction script
│   ├── offline_renderer.py        # 60s video and validation frame rendering engine
│   ├── scripts/                   # Script CHOP callbacks and timeline cues
│   └── shaders/                   # GLSL shaders (gold hologram, particles, RGB-D glitch)
├── cinematics/
│   ├── scenes.json                # 6-scene storyboard timecodes and effect triggers
│   ├── timeline.json              # Frame-by-frame interpolation keyframes
│   └── transitions.json           # Transition shader parameters
├── renders/
│   ├── final/
│   │   └── motionvision_ronaldinho_60s.mp4 # Verified 60s 1080p master video (30.3 MB)
│   └── preview/
│       ├── motionvision_original_vs_ai_split.mp4 # Side-by-side comparison video
│       ├── motionvision_ronaldinho_15s_social.mp4 # 15s social media edit
│       └── validation_frame_scene*.png    # Side-by-side validation frames
└── docs/
    ├── installation.md            # Step-by-step setup guide
    ├── touchdesigner_network.md   # Operator hierarchy and parameter reference
    ├── coordinate_systems.md      # Mathematical coordinate standards
    ├── troubleshooting.md         # Diagnostic runbook
    └── performance.md             # Memory budgets & Apple Silicon tuning
```

---

## 7. Verification & Acceptance Results

- **Source Video Loaded**: `assets/videos/ronaldinho_bernabeu_2005.mp4` (1920 × 1080 @ 30.0 FPS, 1,800 frames).
- **Multi-Person Tracking**: Ronaldinho (#10), Real Madrid defenders (Ramos, Helguera, Salgado, Casillas), Barcelona teammates (Eto'o), and referee reliably tracked with persistent IDs.
- **Ball Trajectory**: Independent tracking with occlusion interpolation and uncertainty flags.
- **SMPL Geometry**: 24-joint articulated kinematic tree and template OBJ (`data/meshes/smpl_template.obj`) loaded in TouchDesigner.
- **All 6 Visual Modes**: Verified across the 60-second timeline (Meshes, Skeletons, Particles, RGB-D, Trails, Analytics).
- **Master Video Rendered**: `renders/final/motionvision_ronaldinho_60s.mp4` (1,800 frames, 60.00 seconds, 30.3 MB).
- **Test Suite**: 13/13 automated unit and integration tests passing (`pytest tests/`).
