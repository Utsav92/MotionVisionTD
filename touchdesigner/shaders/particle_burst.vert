// MotionVision TD — Particle Transformation Vertex Shader (GLSL)
// Mode C: Animates particles from mesh vertices into fluid turbulence and reassembly

uniform float uTime;
uniform float uExplosionFactor; // 0.0 (solid mesh) to 1.0 (full particle explosion)
uniform float uTurbulenceScale;
uniform mat4 uModelViewProj;

in vec3 P;       // Vertex Position
in vec3 N;       // Normal
in vec4 Cd;      // Point Color (Gold / Blue / Red)
in float id;     // Unique point ID

out vec4 vColor;
out float vLife;

// 3D pseudo-curl / simplex noise
vec3 hash3(float p) {
    vec3 p3 = fract(vec3(p) * vec3(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.xxy + p3.yzz) * p3.zyx);
}

void main()
{
    vec3 randDir = hash3(id) * 2.0 - 1.0;

    // Fluid turbulence field
    vec3 curl = vec3(
        sin(P.y * 3.0 + uTime * 2.0) * cos(P.z * 2.5),
        cos(P.x * 2.5 + uTime * 2.0) * sin(P.z * 3.0),
        sin(P.x * 3.0 + uTime * 1.5) * cos(P.y * 2.5)
    );

    // Particle dispersion trajectory
    vec3 displacedPos = P + (N * 0.4 + randDir * 0.6 + curl * uTurbulenceScale) * uExplosionFactor * 3.5;

    gl_Position = uModelViewProj * vec4(displacedPos, 1.0);
    gl_PointSize = mix(3.0, 7.0, hash3(id * 2.1).x) * (1.0 + uExplosionFactor * 0.5);

    vColor = Cd;
    vLife = 1.0 - uExplosionFactor * 0.5;
}
