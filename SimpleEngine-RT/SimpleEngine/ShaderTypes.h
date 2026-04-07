//
//  ShaderTypes.h
//  SimpleEngine
//
//  Created by Farkladin on 4/5/26.
//

//
//  Header containing types and enum constants shared between Metal shaders and Swift/ObjC source
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

typedef NS_ENUM(EnumBackingType, BufferIndex)
{
    BufferIndexMeshPositions = 0,
    BufferIndexMeshGenerics  = 1,
    BufferIndexUniforms      = 2
};

typedef NS_ENUM(EnumBackingType, VertexAttribute)
{
    VertexAttributePosition  = 0,
    VertexAttributeTexcoord  = 1,
};

typedef NS_ENUM(EnumBackingType, TextureIndex)
{
    TextureIndexColor    = 0,
};

typedef struct
{
    matrix_float4x4 projectionMatrix;
    matrix_float4x4 modelViewMatrix;
} Uniforms;

#define MAX_BULBS 8

typedef struct
{
    matrix_float4x4 inverseViewMatrix;
    matrix_float4x4 inverseProjectionMatrix;
    vector_float3 cameraPosition;
    uint32_t instanceCount;
    vector_float3 bulbWorldPositions[MAX_BULBS];
    uint32_t bulbInstanceIDs[MAX_BULBS];
    vector_float3 bulbColors[MAX_BULBS];
    uint32_t bulbCount;
    uint32_t _pad[3];
} RayUniforms;

// Per-instance material properties (passed as buffer[3])
// materialType: 0=opaque, 1=bulb, 2=glass
typedef struct
{
    uint32_t materialType;
    float    emissionPower;     // bulb brightness
    float    ior;               // glass index of refraction
    float    absorption;        // glass absorption per unit distance
    vector_float3 color;        // bulb emission color OR glass tint
    float    _pad;
} InstanceMaterial;

#endif /* ShaderTypes_h */
