// MotionVision TD — RGB-D Hologram & Chromatic Aberration Shader (GLSL)
// Mode D: RGB channel separation, holographic depth slicing, and glitch scanlines

uniform sampler2D sInputVideo;  // TouchDesigner input match video
uniform sampler2D sDepthMap;    // TouchDesigner rendered mesh depth
uniform float uGlitchIntensity; // [0.0, 1.0]
uniform float uChromaOffsetPx;  // Shift in pixels (e.g. 12.0)
uniform float uTime;
uniform vec2 uResolution;

out vec4 fragColor;

void main()
{
    vec2 uv = gl_FragCoord.xy / uResolution;

    // Temporal scanline glitch jitter
    float scanlineNoise = sin(uv.y * 300.0 + uTime * 25.0);
    float glitchTrigger = step(0.92, sin(uTime * 4.0)) * uGlitchIntensity;
    vec2 jitter = vec2(glitchTrigger * scanlineNoise * 0.015, 0.0);

    // Chromatic separation vectors
    vec2 offset = vec2(uChromaOffsetPx / uResolution.x, 0.0) * (1.0 + glitchTrigger * 2.5);

    // Sample channels independently
    float r = texture(sInputVideo, uv + jitter + offset).r;
    float g = texture(sInputVideo, uv + jitter).g;
    float b = texture(sInputVideo, uv + jitter - offset).b;

    // Depth slicing for player silhouette highlight
    float depth = texture(sDepthMap, uv).r;
    vec3 depthTint = vec3(0.0);
    if (depth > 0.05 && depth < 0.98) {
        // Holographic depth contour
        float depthBands = fract(depth * 18.0 - uTime * 1.5);
        depthTint = mix(vec3(0.0, 0.8, 1.0), vec3(1.0, 0.84, 0.0), depthBands) * 0.45;
    }

    vec3 composite = vec3(r, g, b) + depthTint;

    // Subtle vignette
    float dist = distance(uv, vec2(0.5));
    composite *= (1.0 - dist * 0.35);

    fragColor = vec4(composite, 1.0);
}
