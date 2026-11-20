// MotionVision TD — Ronaldinho Gold Hologram Shader (GLSL)
// Cinematic Fresnel rim lighting and technical scanlines for Ronaldinho's #10 mesh

uniform vec4 uBaseGoldColor;   // [1.0, 0.843, 0.0, 1.0]
uniform vec4 uRimColor;        // [1.0, 0.98, 0.70, 1.0]
uniform float uTime;
uniform float uWireframeMix;
uniform float uFresnelPower;

in vec3 vNormal;
in vec3 vViewDir;
in vec3 vWorldPos;
in vec2 vTexCoord;

out vec4 fragColor;

void main()
{
    vec3 N = normalize(vNormal);
    vec3 V = normalize(vViewDir);

    // Fresnel rim factor
    float NdotV = max(0.0, dot(N, V));
    float fresnel = pow(1.0 - NdotV, uFresnelPower > 0.0 ? uFresnelPower : 2.5);

    // Dynamic holographic scanline pattern
    float scanline = sin((vWorldPos.y + uTime * 2.0) * 45.0) * 0.5 + 0.5;
    scanline = smoothstep(0.3, 0.7, scanline);

    // Wireframe grid lines
    vec2 grid = abs(fract(vTexCoord * 32.0 - 0.5) - 0.5) / fwidth(vTexCoord * 32.0);
    float line = min(grid.x, grid.y);
    float wireframe = 1.0 - min(line, 1.0);

    // Combine shading
    vec3 baseColor = uBaseGoldColor.rgb * (0.65 + 0.35 * NdotV);
    vec3 rim = uRimColor.rgb * fresnel * 1.8;
    vec3 finalRgb = baseColor + rim;

    // Scanline & wireframe pulses
    finalRgb += vec3(0.3, 0.25, 0.05) * scanline * 0.4;
    finalRgb = mix(finalRgb, uRimColor.rgb * 1.5, wireframe * uWireframeMix);

    float alpha = uBaseGoldColor.a * (0.85 + 0.15 * fresnel);
    fragColor = vec4(finalRgb, alpha);
}
