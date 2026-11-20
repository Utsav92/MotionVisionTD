# Coordinate Systems & Spatial Conventions — MotionVision TD

To ensure spatial alignment between computer vision algorithms, 3D body models, camera projections, and TouchDesigner viewports, MotionVision TD adheres to explicit mathematical coordinate definitions.

---

## 1. Summary Matrix

| Space / System | Handedness | X-Axis | Y-Axis | Z-Axis | Origin | Units |
|---|---|---|---|---|---|---|
| **TouchDesigner World** | Right-Handed | +X: Right | +Y: Up | +Z: Towards Viewer | Center of World (0,0,0) | Meters ($m$) |
| **SMPL Neutral Model** | Right-Handed | +X: Subject Right | +Y: Up | +Z: Forward (facing viewer) | Pelvis (Joint 0) | Meters ($m$) |
| **OpenCV 2D Image** | Left-Handed | +X: Right | +Y: Down | N/A | Top-Left Corner (0,0) | Pixels ($px$) |
| **Broadcast Pitch Space** | Right-Handed | +X: Down Touchline | +Y: Pitch Width | +Z: Up (elevation) | Corner Flag (0,0,0) | Meters ($m$) |

---

## 2. TouchDesigner 3D Space (Right-Handed, Y-Up)

```
        +Y (Up)
         |
         |
         +---- +X (Right)
        /
       /
     +Z (Towards Viewer)
```

- **TouchDesigner Geometry COMPs & Cameras**: Operate in right-handed metric coordinates.
- **Broadcast Camera Position**: Positioned at approximately `(0.0, 4.5, -28.0)` meters, with a downward pitch angle of $-12.5^{\circ}$, framing the pitch action with a $35^{\circ}$ horizontal field of view.

---

## 3. 3D-to-2D Perspective Projection

The pinhole broadcast camera maps world coordinates $(X, Y, Z)$ to image plane coordinates $(u, v)$:

$$\begin{bmatrix} u \\ v \\ 1 \end{bmatrix} \sim \mathbf{K} \cdot \begin{bmatrix} \mathbf{R} & \mathbf{t} \end{bmatrix} \begin{bmatrix} X \\ Y \\ Z \\ 1 \end{bmatrix}$$

Where intrinsic matrix $\mathbf{K}$ is defined as:

$$\mathbf{K} = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$$

- $f_x, f_y$: Focal lengths in pixels $\approx 2800\text{ px}$ (for 1080p telephoto).
- $c_x, c_y$: Principal point at screen center $(960.0, 540.0)\text{ px}$.
- Vertical inversion: Because 2D image coordinates are Y-down and 3D world is Y-up:
  $$v = -\frac{f_y \cdot Y}{Z} + c_y$$

---

## 4. Measurement Units & Confidence Semantics

- **Positions & Distances**: Expressed in **meters ($m$)**.
- **Rotations & Joint Flexions**: Expressed in **degrees ($^{\circ}$)** and Euler radians.
- **Speeds & Velocities**: Expressed in **meters per second ($m/s$)** and kilometers per hour ($km/h$).
- **Scale Factor Notice**: Monocular broadcast video cannot determine absolute millimeter scale without physical lidar sensors. Scale estimates carry a nominal $\pm 8\%$ margin of uncertainty.
