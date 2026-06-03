//
//  ShaderTypes.h
//  SimpleRayTracer
//
//  Header containing types shared between Metal shaders and Swift source.
//

#ifndef ShaderTypes_h
#define ShaderTypes_h

#ifdef __METAL_VERSION__
#define NS_ENUM(_type, _name) enum _name : _type _name; enum _name : _type
typedef metal::int32_t EnumBackingType;
#else
#import <Foundation/Foundation.h>
typedef NSInteger EnumBackingType;
#endif

#include <simd/simd.h>

// Structures for ray tracing uniforms
typedef struct
{
    matrix_float4x4 inverseViewMatrix;
    matrix_float4x4 inverseProjectionMatrix;
    vector_float3 cameraPosition;
    uint32_t frameIndex;
    uint32_t randomSeed;
    uint32_t _pad[2]; // Align to exactly 160 bytes
} RayUniforms;

// Per-instance material properties (passed as buffer)
// materialType: 0=opaque, 1=bulb, 2=glass
typedef struct
{
    uint32_t materialType;
    float    emissionPower;     // bulb brightness
    float    ior;               // glass index of refraction
    float    absorption;        // glass absorption per unit distance
    vector_float3 color;        // bulb emission color OR glass tint
    float    _pad;              // Align to 16 bytes
} InstanceMaterial;

#endif /* ShaderTypes_h */
