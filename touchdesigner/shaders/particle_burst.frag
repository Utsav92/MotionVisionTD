// MotionVision TD — Particle Transformation Fragment Shader (GLSL)
// Mode C: Soft glowing circular point particles with lifetime falloff

in vec4 vColor;
in float vLife;

out vec4 fragColor;

void main()
{
    // Turn square point sprite into soft circular particle
    vec2 coord = gl_PointCoord - vec2(0.5);
    float distSq = dot(coord, coord);
    if (distSq > 0.25) {
        discard;
    }

    // Soft Gaussian falloff
    float alpha = exp(-distSq * 9.0) * vLife * vColor.a;

    // Glowing core
    vec3 glow = vColor.rgb * (1.0 + (0.25 - distSq) * 3.0);

    fragColor = vec4(glow, alpha);
}
