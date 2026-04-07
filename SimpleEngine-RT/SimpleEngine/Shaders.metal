//
//  Shaders.metal
//  SimpleEngine (RayTracing Compute Target)
//
//  Optimized: accept_any for shadows, early-out, reduced register pressure,
//  minimized divergence, pre-computed constants.
//

#include <metal_stdlib>
#include "ShaderTypes.h"
using namespace metal;
using namespace metal::raytracing;

// ── Material constants ──────────────────────────────────────────────────────
constant uint MAT_OPAQUE = 0;
constant uint MAT_BULB   = 1;
constant uint MAT_GLASS  = 2;

constant float AMBIENT     = 0.08;
constant int   MAX_BOUNCES = 6;
constant float LIGHT_POWER = 80.0;

// ── Vertex data (packed 32-byte stride) ─────────────────────────────────────
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
    float4x4 transform                   [[id(6)]];
};

// ── Helpers ─────────────────────────────────────────────────────────────────
inline float3 tonemap_aces(float3 x) {
    // Faster ACES approximation (Narkowicz 2015)
    constexpr float a = 2.51f;
    constexpr float b = 0.03f;
    constexpr float c = 2.43f;
    constexpr float d = 0.59f;
    constexpr float e = 0.14f;
    return saturate((x * (a * x + b)) / (x * (c * x + d) + e));
}

inline float fresnel_schlick(float cosTheta, float ior1, float ior2) {
    float r0 = (ior1 - ior2) / (ior1 + ior2);
    r0 *= r0;
    float c = 1.0 - cosTheta;
    float c2 = c * c;
    return r0 + (1.0 - r0) * c2 * c2 * c;  // c^5 = c2*c2*c (3 muls instead of 4)
}

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

static float3 getWorldNormal(constant MeshGeometry *geometries, uint instID, uint primID, uint instanceCount) {
    if (instID < instanceCount) {
        float3 localN = getFaceNormal(geometries[instID], primID);
        float3 wn = (geometries[instID].transform * float4(localN, 0.0)).xyz;
        return normalize(wn);
    }
    return float3(0, 1, 0);
}

static float3 skyColor(float3 dir) {
    float t = saturate(dir.y * 0.5 + 0.5);
    return mix(float3(0.04, 0.04, 0.08), float3(0.08, 0.10, 0.18), t);
}

// ── Shadow test using accept_any_intersection for speed ─────────────────────
// Returns true if the point is in shadow from a given light.
// accept_any_intersection stops at the FIRST hit — much faster than finding closest.
// We then verify the hit isn't glass/bulb (those are transparent to shadows).
static bool testShadow(
    float3 origin, float3 lightDir, float lightDist,
    acceleration_structure<instancing> accelStruct,
    device const InstanceMaterial *materials
) {
    ray shadowRay;
    shadowRay.origin       = origin;
    shadowRay.direction    = lightDir;
    shadowRay.min_distance = 0.001;
    shadowRay.max_distance = lightDist;

    // accept_any = true → GPU stops at first triangle, massive speedup for shadows
    intersector<instancing, triangle_data> shadowIsect;
    shadowIsect.accept_any_intersection(true);
    shadowIsect.force_opacity(forced_opacity::opaque);
    auto hit = shadowIsect.intersect(shadowRay, accelStruct);

    if (hit.type == intersection_type::none) return false;
    // Only opaque objects block light; glass and bulbs are transparent
    return (materials[hit.instance_id].materialType == MAT_OPAQUE);
}

// ── Shade opaque surface with all colored bulbs ─────────────────────────────
static float3 shadeOpaque(
    float3 hitPos, float3 N,
    constant RayUniforms &uniforms,
    acceleration_structure<instancing> accelStruct,
    device const InstanceMaterial *materials
) {
    float3 totalLight = float3(0.0);

    for (uint b = 0; b < uniforms.bulbCount; b++) {
        float3 toLight   = uniforms.bulbWorldPositions[b] - hitPos;
        float  distSq    = dot(toLight, toLight);
        float  lightDist = sqrt(distSq);
        float3 lightDir  = toLight * (1.0 / max(lightDist, 0.001));
        float  NdotL     = max(dot(N, lightDir), 0.0);

        // Early out: skip if surface faces away from this light
        if (NdotL < 0.001) continue;

        float3 biasedOrigin = hitPos + N * 0.01;
        bool shadow = testShadow(biasedOrigin, lightDir, lightDist, accelStruct, materials);

        if (!shadow) {
            float atten = 1.0 / (1.0 + distSq * 0.01);
            totalLight += uniforms.bulbColors[b] * (LIGHT_POWER * NdotL * atten);
        }
    }

    return float3(0.7) * (totalLight + AMBIENT);
}

// ── Main Kernel ─────────────────────────────────────────────────────────────
kernel void raytracing_kernel(
    uint2 tid [[thread_position_in_grid]],
    texture2d<float, access::write>     renderTarget [[texture(0)]],
    constant RayUniforms               &uniforms     [[buffer(0)]],
    acceleration_structure<instancing>  accelStruct   [[buffer(1)]],
    constant MeshGeometry              *geometries    [[buffer(2)]],
    device const InstanceMaterial      *materials     [[buffer(3)]]
) {
    uint width  = renderTarget.get_width();
    uint height = renderTarget.get_height();
    if (tid.x >= width || tid.y >= height) return;

    // ── Primary ray ──────────────────────────────────────────────────────
    float2 uv = (float2(tid) + 0.5) / float2(width, height);
    uv = uv * 2.0 - 1.0;
    uv.y = -uv.y;

    float4 clipTarget = uniforms.inverseProjectionMatrix * float4(uv, 1.0, 1.0);
    clipTarget.xyz /= clipTarget.w;
    float3 rayDir = normalize((uniforms.inverseViewMatrix * float4(clipTarget.xyz, 0.0)).xyz);

    // ── Iterative bouncing ───────────────────────────────────────────────
    float3 rayOrigin = uniforms.cameraPosition;
    float3 throughput = float3(1.0);
    float3 accum = float3(0.0);

    // Reuse one intersector across all bounces
    intersector<instancing, triangle_data> isect;
    isect.force_opacity(forced_opacity::opaque);

    for (int bounce = 0; bounce < MAX_BOUNCES; bounce++) {
        ray currentRay;
        currentRay.origin       = rayOrigin;
        currentRay.direction    = rayDir;
        currentRay.min_distance = 0.001;
        currentRay.max_distance = 500.0;

        auto hit = isect.intersect(currentRay, accelStruct);

        if (hit.type == intersection_type::none) {
            accum += throughput * skyColor(rayDir);
            break;
        }

        uint instID = hit.instance_id;
        device const InstanceMaterial &mat = materials[instID];
        float3 hitPos = rayOrigin + rayDir * hit.distance;

        // ── Emissive bulb ────────────────────────────────────────────────
        if (mat.materialType == MAT_BULB) {
            accum += throughput * mat.color * mat.emissionPower;
            break;
        }

        // Normal
        float3 N = getWorldNormal(geometries, instID, hit.primitive_id, uniforms.instanceCount);
        bool frontFace = (dot(N, rayDir) < 0.0);
        if (!frontFace) N = -N;

        // ── Glass ────────────────────────────────────────────────────────
        if (mat.materialType == MAT_GLASS) {
            float glassIOR = mat.ior;
            float3 glassTint = mat.color;
            float eta = frontFace ? (1.0 / glassIOR) : glassIOR;
            float cosI = abs(dot(N, rayDir));
            float F = fresnel_schlick(cosI, frontFace ? 1.0 : glassIOR, frontFace ? glassIOR : 1.0);

            float3 refracted = refract(rayDir, N, eta);
            bool tir = (dot(refracted, refracted) < 0.001);

            if (tir) {
                rayDir = reflect(rayDir, N);
                rayOrigin = hitPos + rayDir * 0.01;
            } else if (F > 0.5) {
                rayDir = reflect(rayDir, N);
                rayOrigin = hitPos + N * 0.01;
            } else {
                rayDir = refracted;
                rayOrigin = hitPos - N * 0.01;
                if (!frontFace) {
                    throughput *= exp(-mat.absorption * hit.distance * (1.0 - glassTint));
                }
                throughput *= glassTint;
            }

            // Specular highlights on glass from all bulbs
            if (frontFace) {
                for (uint b = 0; b < uniforms.bulbCount; b++) {
                    float3 toL = uniforms.bulbWorldPositions[b] - hitPos;
                    float lDistSq = dot(toL, toL);
                    float3 lDir = toL * rsqrt(max(lDistSq, 0.001));
                    float3 H = normalize(lDir - rayDir);
                    float spec = pow(max(dot(N, H), 0.0), 128.0);
                    float atten = 1.0 / (1.0 + lDistSq * 0.01);
                    accum += throughput * uniforms.bulbColors[b] * spec * 40.0 * atten;
                }
            }

            // Early termination if throughput is negligible
            if (max3(throughput.x, throughput.y, throughput.z) < 0.01) break;
            continue;
        }

        // ── Opaque ───────────────────────────────────────────────────────
        accum += throughput * shadeOpaque(hitPos, N, uniforms, accelStruct, materials);
        break;
    }

    renderTarget.write(float4(tonemap_aces(accum), 1.0), tid);
}
