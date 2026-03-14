//
//  Shader.metal
//  LetsUseSwift
//
//  Created by Sora Sugiyama on 2/19/26.
//

#include <metal_stdlib>
using namespace metal;


struct VertexIn
{
    float3 position [[attribute(0)]];
    float3 normal [[attribute(1)]];
};

struct ShadowVertexOut {
    float4 position [[position]];
};

struct VertexOut
{
    float4 position [[position]];
    float3 worldPosition;
    float3 normal;
    float4 lightSpacePosition;
};

vertex ShadowVertexOut shadowVertexShader(VertexIn in [[stage_in]],
                                 constant float4x4& modelMatrix [[buffer(1)]],
                                 constant float4x4& lightSpaceMatrix [[buffer(2)]])
{
    ShadowVertexOut out;
    out.position = lightSpaceMatrix * modelMatrix * float4(in.position, 1.0f);

    return out;
}

vertex VertexOut vertexShader(VertexIn in [[stage_in]],
                              constant float4x4& modelMatrix [[buffer(1)]],
                              constant float4x4& viewMatrix [[buffer(2)]],
                              constant float4x4& projectionMatrix [[buffer(3)]],
                              constant float4x4& lightSpaceMatrix [[buffer(4)]])
{
    VertexOut out;
    
    
    float4 pos4 = modelMatrix * float4(in.position, 1.0f);
    
    out.worldPosition = pos4.xyz;
    out.position = projectionMatrix * viewMatrix * pos4;
    out.normal = normalize((modelMatrix * float4(in.normal, 0.0f)).xyz);
    out.lightSpacePosition = lightSpaceMatrix * pos4;
    
    return out;
}

constexpr sampler shadowSampler(coord::normalized, filter::linear, address::clamp_to_edge, compare_func::less_equal);

fragment float4 fragmentShader(VertexOut in [[stage_in]],
                               depth2d<float> shadowTexture [[texture(0)]])
{
    float3 lightDir{0.57735f, 0.57735f, 0.57735f}; // = normalize(float3{1.0f, 1.0f, 1.0f})
    float3 diffuse = max(dot(in.normal, lightDir), 0.0f);
    
    float3 baseColor = float3{0.8f, 0.8f, 0.8f};
    float3 ambient = float3(0.2f);
    
    float3 ndc = in.lightSpacePosition.xyz / in.lightSpacePosition.w;
    float2 shadowUV = ndc.xy * 0.5f + 0.5f;
    shadowUV.y = 1.0f - shadowUV.y;
    
    float currentDepth = ndc.z;
    float shadow = 1.0f;
    float bias = max(0.001f * (1.0 - dot(in.normal, lightDir)), 0.001f);
    
    if (shadowUV.x >= 0.0f && shadowUV.x <= 1.0f && 0.0f <= shadowUV.y && shadowUV.y <= 1.0f) {
        shadow = shadowTexture.sample_compare(shadowSampler, shadowUV, currentDepth - bias);
    }
    
    float4 finalColor(baseColor * (diffuse * shadow + ambient), 1.0f);
    
    return finalColor;
}
