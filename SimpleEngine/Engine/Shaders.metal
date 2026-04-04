//
//  Shaders.metal
//  SimpleEngine
//
//  Shadow mapping shaders — ported from the original Shader.metal
//

#include <metal_stdlib>
using namespace metal;

// Vertex layout: float3 position (offset 0) + 4B pad + float3 normal (offset 16), stride=32
struct VertexIn {
    float3 position [[attribute(0)]];
    float3 normal   [[attribute(1)]];
};

// MARK: - Shadow Pass

struct ShadowOut {
    float4 position [[position]];
};

vertex ShadowOut shadowVertexShader(
    VertexIn in               [[stage_in]],
    constant float4x4 &model  [[buffer(1)]],
    constant float4x4 &light  [[buffer(2)]]
) {
    ShadowOut out;
    out.position = light * model * float4(in.position, 1.0);
    return out;
}

// MARK: - Main Pass

struct VertexOut {
    float4 position      [[position]];
    float3 worldPosition;
    float3 normal;
    float4 lightSpacePosition;
};

vertex VertexOut vertexShader(
    VertexIn in                       [[stage_in]],
    constant float4x4 &model          [[buffer(1)]],
    constant float4x4 &view           [[buffer(2)]],
    constant float4x4 &projection     [[buffer(3)]],
    constant float4x4 &lightSpaceMat  [[buffer(4)]]
) {
    VertexOut out;
    float4 worldPos        = model * float4(in.position, 1.0);
    out.worldPosition      = worldPos.xyz;
    out.position           = projection * view * worldPos;
    out.normal             = normalize((model * float4(in.normal, 0.0)).xyz);
    out.lightSpacePosition = lightSpaceMat * worldPos;
    return out;
}

constexpr sampler shadowSampler(
    coord::normalized,
    filter::linear,
    address::clamp_to_edge,
    compare_func::less_equal
);

fragment float4 fragmentShader(
    VertexOut      in          [[stage_in]],
    depth2d<float> shadowMap   [[texture(0)]]
) {
    float3 lightDir  = float3(0.57735, 0.57735, 0.57735); // normalize(1,1,1)
    float3 diffuse   = max(dot(in.normal, lightDir), 0.0);

    float3 baseColor = float3(0.8, 0.8, 0.8);
    float3 ambient   = float3(0.2);

    float3 ndc        = in.lightSpacePosition.xyz / in.lightSpacePosition.w;
    float2 shadowUV   = ndc.xy * 0.5 + 0.5;
    shadowUV.y        = 1.0 - shadowUV.y;

    float currentDepth = ndc.z;
    float shadow       = 1.0;
    float bias         = max(0.001 * (1.0 - dot(in.normal, lightDir)), 0.001);

    // Only sample inside valid UV range
    if (shadowUV.x >= 0.0 && shadowUV.x <= 1.0 &&
        shadowUV.y >= 0.0 && shadowUV.y <= 1.0) {
        shadow = shadowMap.sample_compare(shadowSampler, shadowUV, currentDepth - bias);
    }

    float4 finalColor = float4(baseColor * (diffuse * shadow + ambient), 1.0);
    return finalColor;
}
