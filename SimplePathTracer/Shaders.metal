//
//  Shaders.metal
//  SimplePathTracer
//
//  Metal kernels for Path Tracing, Post-Processing, and Standard Rasterization.
//

#include <metal_stdlib>
#import "ShaderTypes.h"

using namespace metal;
using namespace metal::raytracing;

constant uint MAT_OPAQUE = 0;
constant uint MAT_BULB   = 1;
constant uint MAT_GLASS  = 2;

constant float PI = 3.1415926535f;

struct VertexData {
    packed_float3 position;
    float         pad0;
    packed_float3 normal;
    float         pad1;
};

struct MeshGeometry {
    device const VertexData *vertexBuffer [[id(0)]];
    device const uint   *indexBuffer32    [[id(1)]];
    device const ushort *indexBuffer16    [[id(2)]];
    uint isUInt32                         [[id(3)]];
    uint vertexOffset                     [[id(4)]];
    uint indexOffset                      [[id(5)]];
    float4x4 transform                    [[id(6)]];
};

// ── RNG Setup ───────────────────────────────────────────────────────────────

inline uint pcg_hash(uint input) {
    uint state = input * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

inline uint init_rng(uint2 tid, uint2 resolution, uint frameIndex, uint randomSeed) {
    uint pixelIndex = tid.y * resolution.x + tid.x;
    uint seed = pixelIndex ^ (frameIndex * 2654435761u) ^ (randomSeed * 2246822519u);
    return pcg_hash(pcg_hash(seed));
}

inline float rand_float(thread uint &rngState) {
    rngState = rngState * 747796405u + 2891336453u;
    return float(pcg_hash(rngState)) / 4294967296.0f;
}

// ── Cosine Hemisphere Sampling ──────────────────────────────────────────────

inline float3 tangent_to_world(float3 localDir, float3 N) {
    float3 helper = abs(N.x) > 0.99f ? float3(0.0f, 1.0f, 0.0f) : float3(1.0f, 0.0f, 0.0f);
    float3 tangent = normalize(cross(helper, N));
    float3 bitangent = cross(N, tangent);
    return localDir.x * tangent + localDir.y * bitangent + localDir.z * N;
}

inline float3 sample_cosine_hemisphere(float3 N, thread uint &rngState) {
    float r1 = rand_float(rngState);
    float r2 = rand_float(rngState);
    float phi = 2.0f * PI * r1;
    float sinTheta = sqrt(1.0f - r2);
    float x = cos(phi) * sinTheta;
    float y = sin(phi) * sinTheta;
    float z = sqrt(r2);
    return tangent_to_world(float3(x, y, z), N);
}

// ── Normal Calculation ──────────────────────────────────────────────────────

static float3 getFaceNormal(constant MeshGeometry &geom, uint primID) {
    uint i0, i1, i2;
    if (geom.isUInt32) {
        uint base = geom.indexOffset / 4;
        device const uint *idx = geom.indexBuffer32 + base + primID * 3;
        i0 = idx[0]; i1 = idx[1]; i2 = idx[2];
    } else {
        uint base = geom.indexOffset / 2;
        device const ushort *idx = geom.indexBuffer16 + base + primID * 3;
        i0 = idx[0]; i1 = idx[1]; i2 = idx[2];
    }
    uint vBase = geom.vertexOffset / 32;
    float3 p0 = float3(geom.vertexBuffer[vBase + i0].position);
    float3 p1 = float3(geom.vertexBuffer[vBase + i1].position);
    float3 p2 = float3(geom.vertexBuffer[vBase + i2].position);
    float3 geoN = cross(p1 - p0, p2 - p0);
    float lenSq = dot(geoN, geoN);
    return (lenSq > 1e-8) ? (geoN * rsqrt(lenSq)) : float3(0, 1, 0);
}

static float3 getWorldNormal(constant MeshGeometry *geometries, uint instID, uint primID) {
    float3 localN = getFaceNormal(geometries[instID], primID);
    float3 wn = (geometries[instID].transform * float4(localN, 0.0)).xyz;
    return normalize(wn);
}

// ── Environment ─────────────────────────────────────────────────────────────

static float3 skyColor(float3 dir) {
    return float3(0.0f);
}

// ── Fresnel Schlick ─────────────────────────────────────────────────────────

inline float fresnel_schlick(float cosTheta, float ior1, float ior2) {
    float r0 = (ior1 - ior2) / (ior1 + ior2);
    r0 *= r0;
    float c = 1.0f - cosTheta;
    float c2 = c * c;
    return r0 + (1.0f - r0) * c2 * c2 * c;
}

// ── Aces Tonemapping ────────────────────────────────────────────────────────

inline float3 tonemap_aces(float3 x) {
    constexpr float a = 2.51f;
    constexpr float b = 0.03f;
    constexpr float c = 2.43f;
    constexpr float d = 0.59f;
    constexpr float e = 0.14f;
    return saturate((x * (a * x + b)) / (x * (c * x + d) + e));
}

// ── Path Tracer Kernel ──────────────────────────────────────────────────────

kernel void raytracing_kernel(
    uint2 tid [[thread_position_in_grid]],
    texture2d<float, access::read_write> accumulationTarget  [[texture(0)]],
    texture2d<float, access::write>      normalTarget        [[texture(1)]],
    texture2d<float, access::write>      depthTarget         [[texture(2)]],
    texture2d<float, access::write>      albedoTarget        [[texture(3)]],
    constant RayUniforms                &uniforms            [[buffer(0)]],
    acceleration_structure<instancing>   accelStruct         [[buffer(1)]],
    constant MeshGeometry               *geometries          [[buffer(2)]],
    device const InstanceMaterial       *materials           [[buffer(3)]]
) {
    uint width  = accumulationTarget.get_width();
    uint height = accumulationTarget.get_height();
    if (tid.x >= width || tid.y >= height) return;

    // Initialize random seed
    uint rngState = init_rng(tid, uint2(width, height), uniforms.frameIndex, uniforms.randomSeed);

    // Compute ray with anti-aliasing jitter
    float2 jitter = float2(rand_float(rngState), rand_float(rngState)) - 0.5f;
    float2 uv = (float2(tid) + 0.5f + jitter) / float2(width, height);
    uv = uv * 2.0f - 1.0f;
    uv.y = -uv.y;

    float4 clipTarget = uniforms.inverseProjectionMatrix * float4(uv, 1.0, 1.0);
    clipTarget.xyz /= clipTarget.w;
    float3 rayDir = normalize((uniforms.inverseViewMatrix * float4(clipTarget.xyz, 0.0)).xyz);
    float3 rayOrigin = uniforms.cameraPosition;

    float3 throughput = float3(1.0f);
    float3 accumColor = float3(0.0f);

    intersector<instancing, triangle_data> isect;
    isect.force_opacity(forced_opacity::opaque);

    const int MAX_BOUNCES = 10;
    for (int bounce = 0; bounce < MAX_BOUNCES; bounce++) {
        ray r;
        r.origin       = rayOrigin;
        r.direction    = rayDir;
        r.min_distance = 0.001f;
        r.max_distance = 500.0f;

        auto hit = isect.intersect(r, accelStruct);

        if (hit.type == intersection_type::none) {
            accumColor += throughput * skyColor(rayDir);
            if (bounce == 0) {
                normalTarget.write(float4(0.0f), tid);
                depthTarget.write(float4(500.0f), tid);
                albedoTarget.write(float4(0.0f), tid);
            }
            break;
        }

        uint instID = hit.instance_id;
        device const InstanceMaterial &mat = materials[instID];
        float3 hitPos = rayOrigin + rayDir * hit.distance;

        // Compute geometry normal
        float3 N = float3(0.0f);
        if (mat.materialType != MAT_BULB) {
            N = getWorldNormal(geometries, instID, hit.primitive_id);
        } else {
            N = normalize(rayOrigin - hitPos);
        }
        
        bool frontFace = (dot(N, rayDir) < 0.0f);
        if (!frontFace) N = -N;

        if (bounce == 0) {
            normalTarget.write(float4(N, 1.0f), tid);
            depthTarget.write(float4(hit.distance), tid);
            float3 albedoColor = (mat.materialType == MAT_BULB) ? mat.color : (1.0f - mat.color);
            albedoTarget.write(float4(albedoColor, 1.0f), tid);
        }

        // Bulb / Emissive Hit
        if (mat.materialType == MAT_BULB) {
            accumColor += throughput * mat.color * mat.emissionPower;
            break;
        }

        // Glass Material
        if (mat.materialType == MAT_GLASS) {
            float glassIOR = mat.ior;
            float3 glassTint = mat.color;
            
            if (!frontFace) {
                throughput *= exp(-mat.absorption * hit.distance * mat.color);
            }
            
            float eta = frontFace ? (1.0f / glassIOR) : glassIOR;
            float cosI = abs(dot(N, rayDir));
            
            float F = fresnel_schlick(cosI, frontFace ? 1.0f : glassIOR, frontFace ? glassIOR : 1.0f);

            float3 refracted = refract(rayDir, N, eta);
            bool tir = (dot(refracted, refracted) < 0.01f);

            // Refract or reflect based on Fresnel coefficient
            if (tir || rand_float(rngState) < F) {
                // Reflect
                rayDir = reflect(rayDir, N);
                rayOrigin = hitPos + N * 0.001f;
            } else {
                // Refract
                rayDir = refracted;
                rayOrigin = hitPos - N * 0.001f;
            }

            if (max(throughput.x, max(throughput.y, throughput.z)) < 0.01f) break;
            continue;
        }

        // Opaque / Diffuse Material
        rayDir = sample_cosine_hemisphere(N, rngState);
        rayOrigin = hitPos + N * 0.001f;
        throughput *= (1.0f - mat.color);

        // Russian Roulette termination to avoid tracing useless paths
        if (bounce > 2) {
            float q = max(0.05f, 1.0f - max(throughput.x, max(throughput.y, throughput.z)));
            if (rand_float(rngState) < q) {
                break;
            }
            throughput /= (1.0f - q);
        }
    }

    // Read previous accumulated color sum
    float3 prevColor = float3(0.0f);
    if (uniforms.frameIndex > 0) {
        prevColor = accumulationTarget.read(tid).rgb;
    }

    float blendFactor = 1.0f / min(float(uniforms.frameIndex) + 1.0f, 65535.0f);
    float3 finalAccumColor = mix(prevColor, accumColor, blendFactor);
    accumulationTarget.write(float4(finalAccumColor, 1.0f), tid);
}

kernel void postprocess_kernel(
    uint2 tid [[thread_position_in_grid]],
    texture2d<float, access::sample> denoisedSource [[texture(0)]],
    texture2d<float, access::write>  renderTarget   [[texture(1)]]
) {
    uint width  = renderTarget.get_width();
    uint height = renderTarget.get_height();
    if (tid.x >= width || tid.y >= height) return;

    constexpr sampler s(address::clamp_to_edge, filter::linear);
    float2 uv = (float2(tid) + 0.5f) / float2(width, height);

    float3 hdrColor = denoisedSource.sample(s, uv).rgb;
    float3 tonemapped = tonemap_aces(hdrColor);
    float3 gammaCorrected = pow(tonemapped, float3(1.0f / 2.2f));
    renderTarget.write(float4(gammaCorrected, 1.0f), tid);
}

// ── Standard Rasterization Pipeline Shaders ─────────────────────────────────

struct RasterUniforms {
    float4x4 modelMatrix;
    float4x4 viewMatrix;
    float4x4 projectionMatrix;
    float4   color;
    uint     materialType;
};

struct RasterVertexIn {
    float3 position [[attribute(0)]];
    float3 normal   [[attribute(1)]];
};

struct RasterVertexOut {
    float4 position [[position]];
    float3 worldNormal;
    float3 worldPos;
    float4 color;
    uint     materialType;
};

vertex RasterVertexOut raster_vertex(
    RasterVertexIn in [[stage_in]],
    constant RasterUniforms &uniforms [[buffer(1)]]
) {
    RasterVertexOut out;
    float4 worldPos = uniforms.modelMatrix * float4(in.position, 1.0);
    out.position = uniforms.projectionMatrix * uniforms.viewMatrix * worldPos;
    out.worldPos = worldPos.xyz;
    out.worldNormal = normalize((uniforms.modelMatrix * float4(in.normal, 0.0)).xyz);
    out.color = uniforms.color;
    out.materialType = uniforms.materialType;
    return out;
}

fragment float4 raster_fragment(
    RasterVertexOut in [[stage_in]]
) {
    float3 N = normalize(in.worldNormal);
    
    // Directional light from camera/top
    float3 L = normalize(float3(1.0, 2.0, 1.5));
    float diffuse = max(0.2f, dot(N, L));
    
    float3 baseColor = (in.materialType == MAT_BULB) ? in.color.rgb : (1.0f - in.color.rgb);
    float3 finalColor = baseColor * diffuse;
    
    if (in.materialType == MAT_GLASS) {
        // Render glass with transparent blueish tint
        finalColor = mix(finalColor, float3(0.85, 0.95, 1.0), 0.3f);
    } else if (in.materialType == MAT_BULB) {
        // Bulbs are self-illuminated
        finalColor = baseColor * 1.5f;
    }
    
    return float4(finalColor, 1.0f);
}
