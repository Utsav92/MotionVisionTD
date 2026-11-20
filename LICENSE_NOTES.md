# MotionVision TD — License & Compliance Notes

## 1. Video Footage & Broadcast Rights
- **Reference Footage**: The YouTube URLs referenced in the project documentation (`https://www.youtube.com/watch?v=RbzfCHJAGHk` and `https://www.youtube.com/watch?v=MZ2GT_35jLQ`) are provided solely as historical visual benchmarks.
- **Copyright Ownership**: Broadcast rights, logos, team kits, and audio recordings of the 2005 La Liga match between Real Madrid CF and FC Barcelona are the property of La Liga, FC Barcelona, Real Madrid CF, and the respective broadcast rights holders.
- **Local Media Policy**: In strict compliance with digital access controls, this repository does NOT bypass technical protection measures, stream-ripping locks, DRM, or authentication. Users must supply their own authorized, legally acquired local copy of the video file (`assets/videos/ronaldinho_bernabeu_2005.mp4`).
- **Synthetic Test Asset**: A procedural synthetic football test generator is provided (`assets/videos/create_test_footage.py`) so that developers and researchers can build, validate, and test the entire multi-person tracking, 3D SMPL reconstruction, and TouchDesigner rendering pipeline without requiring external copyrighted video.

## 2. Research Models & Open Source Code
- **Apple CoMotion**: Licensed under the Apple Sample Code / Research License (see `https://github.com/apple-aiml-research/ml-comotion`). Pretrained checkpoints are intended for non-commercial research and educational experimentation.
- **SMPL Model**: The SMPL Body Model is developed by the Max Planck Institute for Intelligent Systems. Use of SMPL parameters (`.npz` weights) is subject to the MPI SMPL license (for non-commercial research).
- **YOLOv8**: Developed by Ultralytics (AGPL-3.0 / Enterprise License).
- **TouchDesigner**: Commercial software by Derivative Inc. TouchDesigner projects require an active TouchDesigner Non-Commercial, Educational, Commercial, or Pro license.
- **Attribution Policy**: The resulting visualization must be credited as an independent creative exploration:
  *"RONALDINHO: THE ART OF MOTION — AI Motion Reconstruction × TouchDesigner"*.
  It must not be presented as an official commercial partnership with or endorsement by Apple Inc. or FC Barcelona.
