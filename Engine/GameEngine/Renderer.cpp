//
//  Renderer.cpp
//  LetsUseSwift
//
//  Created by Sora Sugiyama on 2/16/26.
//

#include "Renderer.hpp"


Renderer::Renderer(void* pDevice):
device{NS::RetainPtr(static_cast<MTL::Device*>(pDevice))}
{
    commandQueue = NS::TransferPtr(device->newCommandQueue());
    
    buildShaders();
}

Renderer::~Renderer() = default;

void Renderer::buildShaders()
{
    NS::SharedPtr<MTL::Library> defaultLibrary = NS::TransferPtr(device->newDefaultLibrary());
    if (defaultLibrary) {
        // MARK: - Build Vertex Shader
        NS::SharedPtr<MTL::Function> pVertexFn = NS::TransferPtr(defaultLibrary->newFunction(NS::String::string("vertexShader", NS::UTF8StringEncoding)));
        NS::SharedPtr<MTL::Function> pFragmentFn = NS::TransferPtr(defaultLibrary->newFunction(NS::String::string("fragmentShader", NS::UTF8StringEncoding)));
        NS::SharedPtr<MTL::RenderPipelineDescriptor> pPipelineDescriptor = NS::TransferPtr(MTL::RenderPipelineDescriptor::alloc()->init());
        pPipelineDescriptor->setVertexFunction(pVertexFn.get());
        pPipelineDescriptor->setFragmentFunction(pFragmentFn.get());
        pPipelineDescriptor->colorAttachments()->object(0)->setPixelFormat(MTL::PixelFormat::PixelFormatBGRA8Unorm);
        pPipelineDescriptor->setDepthAttachmentPixelFormat(MTL::PixelFormatDepth32Float);
        
        NS::SharedPtr<MTL::VertexDescriptor> vertexDescriptor = NS::TransferPtr(MTL::VertexDescriptor::alloc()->init());
        
        vertexDescriptor->attributes()->object(0)->setFormat(MTL::VertexFormatFloat3);
        vertexDescriptor->attributes()->object(0)->setOffset(0);
        vertexDescriptor->attributes()->object(0)->setBufferIndex(0);
        vertexDescriptor->attributes()->object(1)->setFormat(MTL::VertexFormatFloat3);
        vertexDescriptor->attributes()->object(1)->setOffset(16);
        vertexDescriptor->attributes()->object(1)->setBufferIndex(0);
        vertexDescriptor->layouts()->object(0)->setStride(32);
        
        pPipelineDescriptor->setVertexDescriptor(vertexDescriptor.get());
        
        NS::Error* error = nullptr;
        pipelineState = NS::TransferPtr(device->newRenderPipelineState(pPipelineDescriptor.get(), &error));
        if (!pipelineState) {
            std::cerr<< "Failed to create pipeline state: " << error->localizedDescription()->utf8String() << std::endl;
        }
        
        // MARK: - Build Shadow Shader
        NS::SharedPtr<MTL::Function> shadowVertexFunc = NS::TransferPtr(defaultLibrary->newFunction(NS::String::string("shadowVertexShader", NS::UTF8StringEncoding)));
        NS::SharedPtr<MTL::RenderPipelineDescriptor> pShadowRenderPipelineDescriptor = NS::TransferPtr(MTL::RenderPipelineDescriptor::alloc()->init());
        pShadowRenderPipelineDescriptor->setVertexFunction(shadowVertexFunc.get());
        pShadowRenderPipelineDescriptor->setFragmentFunction(nullptr);
        pShadowRenderPipelineDescriptor->setDepthAttachmentPixelFormat(MTL::PixelFormat::PixelFormatDepth32Float);
        pShadowRenderPipelineDescriptor->setVertexDescriptor(pPipelineDescriptor->vertexDescriptor());
        pShadowRenderPipelineDescriptor->colorAttachments()->object(0)->setPixelFormat(MTL::PixelFormat::PixelFormatInvalid);
        
        NS::Error* shadowError = nullptr;
        shadowRenderPipelineState = NS::TransferPtr(device->newRenderPipelineState(pShadowRenderPipelineDescriptor.get(), &shadowError));
        if (!shadowRenderPipelineState) {
            std::cerr<< "Failed to create pipeline state: " << shadowError->localizedDescription()->utf8String() << std::endl;
        }
        
        NS::SharedPtr<MTL::DepthStencilDescriptor> pDepthStencilDescriptor = NS::TransferPtr(MTL::DepthStencilDescriptor::alloc()->init());
        pDepthStencilDescriptor->setDepthCompareFunction(MTL::CompareFunctionLess);
        pDepthStencilDescriptor->setDepthWriteEnabled(true);
        depthStencilState = NS::TransferPtr(device->newDepthStencilState(pDepthStencilDescriptor.get()));
        
        MTL::TextureDescriptor* pTextureDescriptor = MTL::TextureDescriptor::alloc()->init();
        pTextureDescriptor->setTextureType(MTL::TextureType2D);
        pTextureDescriptor->setPixelFormat(MTL::PixelFormatDepth32Float);
        pTextureDescriptor->setWidth(2048);
        pTextureDescriptor->setHeight(2048);
        pTextureDescriptor->setStorageMode(MTL::StorageModePrivate);
        pTextureDescriptor->setUsage(MTL::TextureUsageRenderTarget | MTL::TextureUsageShaderRead);
        
        shadowTexture = NS::TransferPtr(device->newTexture(pTextureDescriptor));
        
        
        NS::SharedPtr<MTL::DepthStencilDescriptor> pShadowDepthStencilDescriptor = NS::TransferPtr(MTL::DepthStencilDescriptor::alloc()->init());
        pShadowDepthStencilDescriptor->setDepthCompareFunction(MTL::CompareFunctionLessEqual);
        pShadowDepthStencilDescriptor->setDepthWriteEnabled(true);
        
        shadowDepthStencilState = NS::TransferPtr(device->newDepthStencilState(pShadowDepthStencilDescriptor.get()));
    }
}

void Renderer::addMesh(void* pVertexBuffer, void *pIndexBuffer, std::uint32_t n, bool isUInt32)
{
    if (!objects.add_back(pVertexBuffer, pIndexBuffer, n, isUInt32, 0)) {
        std::cerr << "Mesh did not added." << std::endl;
    }
}

simd::float4x4 Renderer::makeOrthogonalMatrix(float left, float right, float bottom, float top, float nearZ, float farZ)
{
    return simd::float4x4{
        simd::float4{2.0f / (right - left), 0.0f, 0.0f, 0.0f},
        simd::float4{0.0f, 2.0f / (top - bottom), 0.0f ,0.0f},
        simd::float4{0.0f, 0.0f, -1.0f / (farZ-nearZ), 0.0f},
        simd::float4{(left + right) / (left - right), (top + bottom) / (top - bottom), nearZ / (nearZ - farZ), 1.0f}
    };
}

simd::float4x4 Renderer::makeLookAt(simd::float3 eye, simd::float3 center, simd::float3 up)
{
    simd::float3 z = simd::normalize(eye - center);
    simd::float3 x = simd::normalize(simd::cross(up, z));
    simd::float3 y = simd::cross(z, x);
    
    return simd::float4x4{
        simd::float4{x.x, y.x, z.x, 0.0f},
        simd::float4{x.y, y.y, z.y, 0.0f},
        simd::float4{x.z, y.z, z.z, 0.0f},
        simd::float4{-simd::dot(x, eye), -simd::dot(y, eye), -simd::dot(z, eye), 1.0f}
    };
}

simd::float4x4 Renderer::makePerspectiveMatrix(float fovRad, float aspect, float nearZ, float farZ) const
{
    float ys = 1.0f / tanf(fovRad * 0.5f);
    float xs = ys / aspect;
    float zs = farZ / (nearZ - farZ);
    
    return simd::float4x4{
        simd::float4{xs, 0.0f, 0.0f, 0.0f},
        simd::float4{0.0f, ys, 0.0f, 0.0f},
        simd::float4{0.0f, 0.0f, zs, -1.0f},
        simd::float4{0.0f, 0.0f, zs * nearZ, 0.0f}
    };
}

void Renderer::resize(float width, float height)
{
    viewportSize.x = width;
    viewportSize.y = height;
}

void Renderer::draw(void* pCurrentDrawable, void* pRenderPassDescriptor)
{
    MTL::Drawable* drawable = (MTL::Drawable*)pCurrentDrawable;
    
    MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
    
    
    const simd::float3 lightPos{5.0f, 5.0f, 5.0f}, lightTarget{0.0f, 0.0f, 0.0f}, lightUp{0.0f, 1.0f, 0.0f};
    simd::float4x4 lightView = makeLookAt(lightPos, lightTarget, lightUp);
    simd::float4x4 lightProjection = makeOrthogonalMatrix(-10.0f, 10.0f, -10.0f, 10.0f, 0.01f, 50.0f);
    simd::float4x4 lightSpaceMatrix = lightProjection * lightView;
    
    simd::float4x4 viewMatrix = camera.getViewMatrix();
    simd::float4x4 modelMatrix = matrix_identity_float4x4;
    
    float aspect = viewportSize.x / viewportSize.y;
    simd::float4x4 projectionMatrix = makePerspectiveMatrix(45.0f * Consts::PIf / 180.0f, aspect, 0.1f, 100.0f);
    
    // MARK: - [Shadow Pass]
    NS::SharedPtr<MTL::RenderPassDescriptor> shadowRenderPassDescriptor = NS::TransferPtr(MTL::RenderPassDescriptor::alloc()->init());
    shadowRenderPassDescriptor->depthAttachment()->setTexture(shadowTexture.get());
    shadowRenderPassDescriptor->depthAttachment()->setClearDepth(1.0);
    shadowRenderPassDescriptor->depthAttachment()->setLoadAction(MTL::LoadActionClear);
    shadowRenderPassDescriptor->depthAttachment()->setStoreAction(MTL::StoreActionStore);
    
    MTL::RenderCommandEncoder* shadowRenderCommandEncoder = commandBuffer->renderCommandEncoder(shadowRenderPassDescriptor.get());
    shadowRenderCommandEncoder->setRenderPipelineState(shadowRenderPipelineState.get());
    shadowRenderCommandEncoder->setDepthStencilState(shadowDepthStencilState.get());
    //shadowRenderCommandEncoder->setCullMode(MTL::CullModeFront);
    
    for(std::size_t i=0; i < objects.size; ++i) {
        MTL::IndexType indexType = objects[i].isUInt32Index ?  MTL::IndexTypeUInt32 : MTL::IndexTypeUInt16;
        const GameObject& obj = objects[i];
        
        shadowRenderCommandEncoder->setVertexBuffer(obj.vertexBuffer.get(), 0, 0);
        shadowRenderCommandEncoder->setVertexBytes(&modelMatrix, sizeof(modelMatrix), 1);
        shadowRenderCommandEncoder->setVertexBytes(&lightSpaceMatrix, sizeof(lightSpaceMatrix), 2);
        
        shadowRenderCommandEncoder->drawIndexedPrimitives(MTL::PrimitiveTypeTriangle, obj.indexCount, indexType, obj.indexBuffer.get(), 0);
    }
    shadowRenderCommandEncoder->endEncoding();
    
    
    //MARK: - [Main Pass]
    
    NS::SharedPtr<MTL::RenderPassDescriptor> renderPassDescriptor = NS::RetainPtr(static_cast<MTL::RenderPassDescriptor*>(pRenderPassDescriptor));
    MTL::RenderCommandEncoder* renderCommandEncoder = commandBuffer->renderCommandEncoder(renderPassDescriptor.get());
    
    for(std::size_t i=0; i < objects.size; ++i) {
        MTL::IndexType indexType = objects[i].isUInt32Index ?  MTL::IndexTypeUInt32 : MTL::IndexTypeUInt16;
        const GameObject& obj = objects[i];
        
        renderCommandEncoder->setRenderPipelineState(pipelineState.get());
        renderCommandEncoder->setDepthStencilState(depthStencilState.get());
        
        renderCommandEncoder->setVertexBuffer(obj.vertexBuffer.get(), 0, 0);
        renderCommandEncoder->setVertexBytes(&modelMatrix, sizeof(modelMatrix), 1);
        renderCommandEncoder->setVertexBytes(&viewMatrix, sizeof(viewMatrix), 2);
        renderCommandEncoder->setVertexBytes(&projectionMatrix, sizeof(projectionMatrix), 3);
        renderCommandEncoder->setVertexBytes(&lightSpaceMatrix, sizeof(lightSpaceMatrix), 4);
        renderCommandEncoder->setFragmentTexture(shadowTexture.get(), 0);
        
        renderCommandEncoder->drawIndexedPrimitives(MTL::PrimitiveTypeTriangle, obj.indexCount, indexType, obj.indexBuffer.get(), 0);
    }
    
    renderCommandEncoder->endEncoding();
    commandBuffer->presentDrawable(drawable);
    commandBuffer->commit();
}

void Renderer::move(bool w, bool a, bool s, bool d, bool space, bool ctrl, float deltaTime)
{
    float v = 5.0f * deltaTime;
    if(w) camera.moveFront(v);
    if(s) camera.moveFront(-v);
    if(a) camera.moveSide(-v);
    if(d) camera.moveSide(v);
    if(space) camera.moveUp(v);
    if(ctrl) camera.moveUp(-v);
}

void Renderer::moveToCenter()
{
    camera.moveToCenter();
}

void Renderer::rotateCamera(float deltaX, float deltaY)
{
    camera.rotate(deltaX, deltaY);
}

Renderer* Renderer::create(void* pDevice)
{
    return new Renderer(pDevice);
}

void Renderer::destroy()
{
    delete this;
}
