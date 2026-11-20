# TouchDesigner Network Architecture — MotionVision TD

```
/project1
  ├── /video_input          # Movie File In TOP, 1080p Crop, Resolution & Timecode
  ├── /motion_data          # File In DATs & Script CHOP parsing motion_data.npz
  ├── /player_identity      # Channel filters for Ronaldinho (Gold), Madrid (Red), Barca (Blue)
  ├── /camera_tracking      # Camera COMPs: Broadcast vs Follow vs Cinematic Orbit
  ├── /smpl_renderer        # File In SOP (smpl_template.obj), Geometry COMPs & GLSL Mats
  ├── /skeleton_renderer    # 24-joint articulated tubes & spheres xray renderer
  ├── /ball_tracking        # Ball Geometry COMP with turquoise trajectory ribbon SOP
  ├── /motion_trails        # Feedback TOP + Level TOP decay + Over TOP persistence
  ├── /particle_system      # Point SOP / POPs / GLSL fluid curl noise turbulence
  ├── /rgbd_visualization   # GLSL depth slicing & chromatic aberration separation
  ├── /heatmap_renderer     # 2D pitch space Gaussian splatting density TOP
  ├── /biomechanics         # HUD Text TOP (velocity, acceleration, knee/hip flexion)
  ├── /cinematic_fx         # Film grain, anamorphic bloom, vignette, LUT grading
  ├── /compositor           # Switch TOP for Modes A-F, Cross TOP transitions & Split Screen
  ├── /timeline             # Timeline CHOP, Cue DAT 60s storyboard sequencer
  ├── /export_manager       # Movie File Out TOP (1080p 30fps H.264 offline stepping)
  └── /ui_controls          # Polished custom parameter interface
```

## Component Reference & Functionality

### 1. `/video_input`
- **`moviefilein1`** (Movie File In TOP): Streams source football video (`assets/videos/ronaldinho_bernabeu_2005.mp4`).
- **`null_video`** (Null TOP): Global video bus referenced by compositors.

### 2. `/motion_data`
- **`filein_tracking`** (File In DAT): JSON table of tracked player bounding boxes and IDs.
- **`filein_ball`** (File In DAT): High-frequency football trajectory with occlusion flags.
- **`script_sync`** (Script CHOP): Loads binary `data/poses/motion_data.npz` directly into fast GPU-accessible motion channels (`p{id}_tx`, `p{id}_ty`, `p{id}_tz`, `p{id}_conf`).

### 3. `/smpl_renderer`
- **`filein_smpl_mesh`** (File In SOP): Loads canonical SMPL template mesh `data/meshes/smpl_template.obj` (1,472–6,890 vertices, 2,208–13,776 faces).
- **`geo_ronaldinho`**: Instances Ronaldinho's mesh with `shaders/gold_hologram.frag`.
- **`geo_defenders`**: Instances Real Madrid players with neon crimson shader.
- **`geo_teammates`**: Instances Barcelona players with electric blue shader.
- **`render_meshes`** (Render TOP): Outputs 1920x1080 anti-aliased geometry pass.

### 4. `/ui_controls`
- Custom Component Parameters tab organized into 6 interactive sections:
  1. **Video**: Load footage, play/pause, scrub, playback rate.
  2. **AI Tracking**: Offline inference pulse, confirm Ronaldinho ID, show bounding boxes and confidence.
  3. **Visualization**: Mode selector (A: Meshes, B: Skeleton, C: Particles, D: RGB-D, E: Trails, F: Analytics), mesh opacity slider, split screen toggle.
  4. **Visual Effects**: Glow intensity, feedback decay, particle density, chroma shift px.
  5. **Camera**: Broadcast, follow player, orbit turntable, isolated 3D body.
  6. **Export**: Resolution presets, movie format, export 60s video trigger.
