//
//  Renderer.hpp
//  LetsUseSwift
//
//  Created by Sora Sugiyama on 2/16/26.
//

#ifndef Renderer_hpp
#define Renderer_hpp

#include <Metal/Metal.hpp>
#include <simd/simd.h>
#include <cstdint>
#include <vector>
#include <cstdlib>
#include <iostream>

#include "GameObject.hpp"
#include "Camera.hpp"

namespace MTL
{
class Device;
class CommandQueue;
class RenderPipelineState;
class Buffer;
class DepthStencilState;
class Texture;
}

class Renderer
{
    NS::SharedPtr<MTL::Device> device;
    NS::SharedPtr<MTL::CommandQueue> commandQueue;
    
    NS::SharedPtr<MTL::Buffer> vertexBuffer;
    NS::SharedPtr<MTL::Buffer> indexBuffer;
    
    NS::SharedPtr<MTL::RenderPipelineState> pipelineState;
    NS::SharedPtr<MTL::DepthStencilState> depthStencilState;
    
    NS::SharedPtr<MTL::RenderPipelineState> shadowRenderPipelineState;
    NS::SharedPtr<MTL::DepthStencilState> shadowDepthStencilState;
    NS::SharedPtr<MTL::Texture> shadowTexture;
    
    std::uint32_t indexCount = 0;
    bool isUInt32Index = false;
    
    Camera camera;
    
    simd::float2 viewportSize;
    
    void buildShaders();
    simd::float4x4 makePerspectiveMatrix(float fovRad, float aspect, float nearZ, float farZ) const;
    simd::float4x4 makeOrthogonalMatrix(float left, float right, float bottom, float top, float nearZ, float farZ);
    simd::float4x4 makeLookAt(simd::float3 eye, simd::float3 center, simd::float3 up);
    
    ObjectList objects;
    
public:
    Renderer(void* pDevice);
    ~Renderer();
    
    
    void draw(void* pCurrentDrawable, void* pRenderPassDescriptor);
    void resize(float width, float height);
    void move(bool w, bool a, bool s, bool d, bool space, bool ctrl, float deltaTime);
    void moveToCenter();
    void rotateCamera(float deltaX, float deltaY);
    void addMesh(void* pVertexBuffer, void* pIndexBuffer, std::uint32_t n, bool isUInt32);
    
    static Renderer* create(void* pDevice);
    
    void destroy();
};


#endif /* Renderer_hpp */
