# Installation & Setup Guide — MotionVision TD

## 1. System Requirements

### Recommended Hardware:
- **Platform**: Apple Silicon Mac (Mac mini, Mac Studio, MacBook Pro with M2/M3/M4 chip).
- **Memory**: 24 GB Unified Memory (or greater).
- **Storage**: Minimum 10 GB free SSD storage for video caches and mesh data.
- **Operating System**: macOS Sonoma 14.0+ or Sequoia 15.0+ (Cross-platform compatible with Windows 11 & Linux).

### Software Requirements:
- **TouchDesigner**: Version 2023.11000+ or 2025.30000+ (0.99.2025.33230 validated).
- **Python**: Version 3.10 or 3.11 (64-bit).
- **FFmpeg**: Required for offline video encoding and container packaging.

---

## 2. Environment Setup

### Step A: Clone the Repository
```bash
git clone https://github.com/your-username/MotionVisionTD.git
cd MotionVisionTD
```

### Step B: Create a Dedicated Python Virtual Environment
We recommend creating an isolated virtual environment to avoid conflicts with TouchDesigner's embedded Python:
```bash
python3 -m venv .venv
source .venv/bin/activate    # On macOS/Linux
# or
.venv\Scripts\activate       # On Windows
```

### Step C: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Apple CoMotion Setup (Optional for Full Research Checkpoints)

If using Apple's official `ml-comotion` research checkpoints:
1. Clone Apple's official repository:
   ```bash
   git clone https://github.com/apple-aiml-research/ml-comotion external/ml-comotion
   ```
2. Follow Apple's repository instructions to download pretrained weights into `external/ml-comotion/checkpoints/`.
3. If checkpoints are not downloaded, MotionVision TD will automatically and gracefully execute its verified Articulated SMPL Kinematic Reconstruction fallback.

---

## 4. Running the AI Pipeline

To process video footage (generating tracking, SMPL poses, ball trajectory, and camera parameters):
```bash
# If using user's licensed match video:
python -m ai.export_motion assets/videos/ronaldinho_bernabeu_2005.mp4 1800

# Or to generate and validate with synthetic test footage:
python assets/videos/create_test_footage.py assets/videos/synthetic_ronaldinho_test.mp4 60
python -m ai.export_motion assets/videos/synthetic_ronaldinho_test.mp4 1800
```

---

## 5. Opening the Project in TouchDesigner

1. Launch TouchDesigner.
2. Open `touchdesigner/MotionVision_Ronaldinho.toe`.
3. In the `/project1/ui_controls` component, adjust visual modes, parameters, and cameras.
4. To trigger offline rendering inside TouchDesigner, pulse `Export60sVideo`.
