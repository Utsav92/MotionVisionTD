# Troubleshooting & Common Issues — MotionVision TD

## 1. Video Loading Issues

### Symptom: `Movie File In TOP shows black or red error flag`
- **Cause**: The video file at `assets/videos/ronaldinho_bernabeu_2005.mp4` is missing or in an unsupported container.
- **Fix**:
  1. Verify the file exists and is readable:
     ```bash
     ls -lh assets/videos/ronaldinho_bernabeu_2005.mp4
     ```
  2. If the authorized match video is not yet placed, regenerate the synthetic test footage:
     ```bash
     python assets/videos/create_test_footage.py assets/videos/ronaldinho_bernabeu_2005.mp4 60
     ```

---

## 2. Python Environment & Sandboxing Permissions

### Symptom: `PermissionError: Access is denied: ... .git` or `pytest.ini`
- **Cause**: Pytest or Ultralytics traversing upward beyond the workspace boundaries.
- **Fix**:
  - Always execute commands from the `MotionVisionTD` project root directory.
  - MotionVision TD includes safe patches in `ai/detection.py` and provides a localized `pytest.ini` that restricts test exploration to the project folder.

---

## 3. TouchDesigner Operator Errors

### Symptom: `Script CHOP shows cook error`
- **Cause**: `data/poses/motion_data.npz` has not been generated yet.
- **Fix**:
  Run the master AI export script before cooking the network:
  ```bash
  python -m ai.export_motion assets/videos/ronaldinho_bernabeu_2005.mp4 1800
  ```

### Symptom: `GLSL shader compile warning on macOS`
- **Cause**: OpenGL / Metal profile discrepancy in TouchDesigner on macOS.
- **Fix**:
  Ensure the GLSL TOP / MAT has its Language version parameter set to `#version 330` or `GLSL 4.50`, which is natively mapped to Metal by TouchDesigner's macOS compiler.

---

## 4. Player Identity Swapping During Occlusion

### Symptom: `Ronaldinho's track ID changes after colliding with Sergio Ramos`
- **Fix**:
  Use the manual confirmation override tool in `ai/player_identity.py` or the TouchDesigner UI:
  ```python
  from ai.player_identity import PlayerIdentityManager, ROLE_RONALDINHO
  mgr = PlayerIdentityManager()
  mgr.confirm_identity(track_id=2, role=ROLE_RONALDINHO, start_frame=540, end_frame=900)
  ```
  This immediately locks Ronaldinho's gold visual identity across the specified frames without requiring full neural network re-inference.
