//
//  Renderer.swift
//  SimplePathTracer
//
//  Manages both the rasterization (Setup Mode) and progressive path-tracing (Tracing Mode) pipelines.
//

import Metal
import MetalKit
import simd
import ModelIO
import Foundation

enum RenderMode {
    case setup
    case tracing
}

final class Renderer {

    let device: MTLDevice
    private let commandQueue: MTLCommandQueue

    // Pipeline States
    private var computePipelineState: MTLComputePipelineState!
    private var postProcessPipelineState: MTLComputePipelineState!
    private var rasterPipelineState: MTLRenderPipelineState!
    private var rasterDepthState: MTLDepthStencilState!
    
    private var geometryArgumentEncoder: MTLArgumentEncoder!
    private var geometryArgumentBuffer: MTLBuffer?
    private var materialBuffer: MTLBuffer?
    private var allGeometryBuffers: [MTLBuffer] = []
    
    private var instanceAccelerationStructure: MTLAccelerationStructure?
    private var primitiveAccelerationStructures: [MTLAccelerationStructure] = []

    let camera = Camera()
    private var viewportSize: SIMD2<Float> = .zero
    let objects = ObjectList()

    private var accumulationTexture: MTLTexture?
    private var normalTexture: MTLTexture?
    private var depthTexture: MTLTexture?
    private var albedoTexture: MTLTexture?
    private var denoisedAccumulationTexture: MTLTexture?
    private var outputTexture: MTLTexture? // Stores final tonemapped output
    
    private let denoiser = MLXDenoiser()
    private var denoiseExchangeBuffer: MTLBuffer?
    
    // UI State variables
    var renderMode: RenderMode = .setup {
        didSet {
            if renderMode == .setup {
                frameIndex = 0
            }
        }
    }
    var isPaused: Bool = false
    var isDenoisingActive: Bool = false
    
    var currentFrameIndex: UInt32 { frameIndex }
    private var frameIndex: UInt32 = 0

    // Keep ModelIO mesh buffers allocated
    private var retainedMeshes: [MTKMesh] = []
    private var retainedAssets: [MDLAsset] = []
    private var retainedAllocators: [MTKMeshBufferAllocator] = []

    init?(device: MTLDevice) {
        guard let queue = device.makeCommandQueue() else { return nil }
        self.device = device
        self.commandQueue = queue
        buildPipelines()
    }

    private func buildPipelines() {
        guard let library = device.makeDefaultLibrary() else {
            fatalError("Failed to load Metal library.")
        }
        
        guard let rayKernel = library.makeFunction(name: "raytracing_kernel"),
              let postKernel = library.makeFunction(name: "postprocess_kernel"),
              let vertexFn = library.makeFunction(name: "raster_vertex"),
              let fragmentFn = library.makeFunction(name: "raster_fragment") else {
            fatalError("Failed to load shader functions. Make sure Shaders.metal is built.")
        }
        
        do {
            // Compute Pipelines
            computePipelineState = try device.makeComputePipelineState(function: rayKernel)
            postProcessPipelineState = try device.makeComputePipelineState(function: postKernel)
            geometryArgumentEncoder = rayKernel.makeArgumentEncoder(bufferIndex: 2)
            
            // Raster Pipeline
            let pipelineDesc = MTLRenderPipelineDescriptor()
            pipelineDesc.vertexFunction = vertexFn
            pipelineDesc.fragmentFunction = fragmentFn
            pipelineDesc.colorAttachments[0].pixelFormat = .bgra8Unorm
            pipelineDesc.depthAttachmentPixelFormat = .depth32Float
            
            let mtlVertexDesc = MTLVertexDescriptor()
            // Attribute 0: position
            mtlVertexDesc.attributes[0].format = .float3
            mtlVertexDesc.attributes[0].offset = 0
            mtlVertexDesc.attributes[0].bufferIndex = 0
            // Attribute 1: normal
            mtlVertexDesc.attributes[1].format = .float3
            mtlVertexDesc.attributes[1].offset = 16
            mtlVertexDesc.attributes[1].bufferIndex = 0
            // Layout 0: 32-byte stride
            mtlVertexDesc.layouts[0].stride = 32
            pipelineDesc.vertexDescriptor = mtlVertexDesc
            
            rasterPipelineState = try device.makeRenderPipelineState(descriptor: pipelineDesc)
            
            // Depth Stencil State
            let depthDesc = MTLDepthStencilDescriptor()
            depthDesc.depthCompareFunction = .less
            depthDesc.isDepthWriteEnabled = true
            rasterDepthState = device.makeDepthStencilState(descriptor: depthDesc)
            
            print("Pipelines initialized successfully.")
        } catch {
            fatalError("Failed to create pipeline states: \(error)")
        }
    }

    func buildAccelerationStructures() {
        guard objects.count > 0 else {
            print("No objects to build acceleration structures for")
            return
        }

        var primitiveStructures: [MTLAccelerationStructure] = []
        var instanceDescriptors: [MTLAccelerationStructureInstanceDescriptor] = []
        var instanceMaterials: [InstanceMaterial] = []
        allGeometryBuffers.removeAll()

        var validCount = 0
        for i in 0..<objects.count {
            let obj = objects[i]
            if obj.vertexBuffer != nil, obj.indexBuffer != nil, Int(obj.indexCount) % 3 == 0, obj.indexCount > 0 {
                validCount += 1
            }
        }

        let encodedLen = geometryArgumentEncoder.encodedLength
        let argBufLength = encodedLen * max(1, validCount)
        guard let argBuffer = device.makeBuffer(length: argBufLength, options: .storageModeShared) else {
            print("Failed to make geometry argument buffer")
            return
        }

        // Phase 1: Build Primitive (Bottom-Level) Acceleration Structures
        for i in 0..<objects.count {
            let obj = objects[i]
            guard let vb = obj.vertexBuffer, let ib = obj.indexBuffer else {
                continue
            }
            guard Int(obj.indexCount) % 3 == 0, obj.indexCount > 0 else {
                continue
            }

            let geometryDesc = MTLAccelerationStructureTriangleGeometryDescriptor()
            geometryDesc.vertexBuffer = vb
            geometryDesc.vertexBufferOffset = obj.vertexOffset
            geometryDesc.vertexStride = 32
            geometryDesc.indexBuffer = ib
            geometryDesc.indexBufferOffset = obj.indexOffset
            geometryDesc.indexType = obj.isUInt32Index ? .uint32 : .uint16
            geometryDesc.triangleCount = Int(obj.indexCount) / 3
            geometryDesc.opaque = true

            let primitiveDesc = MTLPrimitiveAccelerationStructureDescriptor()
            primitiveDesc.geometryDescriptors = [geometryDesc]

            let sizes = device.accelerationStructureSizes(descriptor: primitiveDesc)
            guard sizes.accelerationStructureSize > 0,
                  let primAS = device.makeAccelerationStructure(size: sizes.accelerationStructureSize),
                  let scratch = device.makeBuffer(length: max(sizes.buildScratchBufferSize, 4),
                                                  options: .storageModePrivate) else {
                print("Object \(i): could not allocate primitive structure")
                continue
            }

            guard let cmdBuf = commandQueue.makeCommandBuffer(),
                  let encoder = cmdBuf.makeAccelerationStructureCommandEncoder() else { continue }
            encoder.build(accelerationStructure: primAS,
                          descriptor: primitiveDesc,
                          scratchBuffer: scratch,
                          scratchBufferOffset: 0)
            encoder.endEncoding()
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()

            let instanceIndex = primitiveStructures.count

            // Compute transformation matrix
            let modelMatrix = makeTransformMatrix(position: obj.position, rotation: obj.rotation, scale: obj.scale)

            // Setup Instance Descriptor
            var desc = MTLAccelerationStructureInstanceDescriptor()
            desc.accelerationStructureIndex = UInt32(instanceIndex)
            desc.intersectionFunctionTableOffset = 0
            desc.mask = 0xFF
            desc.options = .opaque
            desc.transformationMatrix.columns.0.x = modelMatrix.columns.0.x
            desc.transformationMatrix.columns.0.y = modelMatrix.columns.0.y
            desc.transformationMatrix.columns.0.z = modelMatrix.columns.0.z
            
            desc.transformationMatrix.columns.1.x = modelMatrix.columns.1.x
            desc.transformationMatrix.columns.1.y = modelMatrix.columns.1.y
            desc.transformationMatrix.columns.1.z = modelMatrix.columns.1.z
            
            desc.transformationMatrix.columns.2.x = modelMatrix.columns.2.x
            desc.transformationMatrix.columns.2.y = modelMatrix.columns.2.y
            desc.transformationMatrix.columns.2.z = modelMatrix.columns.2.z
            
            desc.transformationMatrix.columns.3.x = modelMatrix.columns.3.x
            desc.transformationMatrix.columns.3.y = modelMatrix.columns.3.y
            desc.transformationMatrix.columns.3.z = modelMatrix.columns.3.z

            // Fill Geometry Argument Buffer for this instance
            geometryArgumentEncoder.setArgumentBuffer(argBuffer, startOffset: 0, arrayElement: instanceIndex)
            geometryArgumentEncoder.setBuffer(vb, offset: 0, index: 0)
            geometryArgumentEncoder.setBuffer(ib, offset: 0, index: 1)
            geometryArgumentEncoder.setBuffer(ib, offset: 0, index: 2)
            allGeometryBuffers.append(vb)
            allGeometryBuffers.append(ib)

            let ptrIsUInt32 = geometryArgumentEncoder.constantData(at: 3).assumingMemoryBound(to: UInt32.self)
            ptrIsUInt32.pointee = obj.isUInt32Index ? 1 : 0
            let ptrVOffset = geometryArgumentEncoder.constantData(at: 4).assumingMemoryBound(to: UInt32.self)
            ptrVOffset.pointee = UInt32(obj.vertexOffset)
            let ptrIOffset = geometryArgumentEncoder.constantData(at: 5).assumingMemoryBound(to: UInt32.self)
            ptrIOffset.pointee = UInt32(obj.indexOffset)
            let ptrTransform = geometryArgumentEncoder.constantData(at: 6).assumingMemoryBound(to: simd_float4x4.self)
            ptrTransform.pointee = modelMatrix

            // Build InstanceMaterial for GPU (maps to ShaderTypes.h)
            var gpuMat = InstanceMaterial()
            gpuMat.materialType = obj.material.materialType
            gpuMat.emissionPower = obj.material.emissionPower
            gpuMat.ior = obj.material.ior
            gpuMat.absorption = obj.material.absorption
            gpuMat.color = obj.material.color
            instanceMaterials.append(gpuMat)

            primitiveStructures.append(primAS)
            instanceDescriptors.append(desc)
        }

        guard !primitiveStructures.isEmpty else {
            print("No valid primitive structures built")
            return
        }

        self.geometryArgumentBuffer = argBuffer
        
        // Build material buffer
        self.materialBuffer = device.makeBuffer(
            bytes: instanceMaterials,
            length: MemoryLayout<InstanceMaterial>.stride * instanceMaterials.count,
            options: .storageModeShared)

        // Phase 2: Build Instance (Top-Level) Acceleration Structure
        guard let instanceBuffer = device.makeBuffer(
            bytes: instanceDescriptors,
            length: MemoryLayout<MTLAccelerationStructureInstanceDescriptor>.stride * instanceDescriptors.count,
            options: .storageModeShared) else {
            print("Could not allocate instance buffer")
            return
        }

        let instanceDesc = MTLInstanceAccelerationStructureDescriptor()
        instanceDesc.instancedAccelerationStructures = primitiveStructures
        instanceDesc.instanceCount = primitiveStructures.count
        instanceDesc.instanceDescriptorBuffer = instanceBuffer

        let instanceSizes = device.accelerationStructureSizes(descriptor: instanceDesc)
        guard instanceSizes.accelerationStructureSize > 0,
              let finalAS = device.makeAccelerationStructure(size: instanceSizes.accelerationStructureSize),
              let instScratch = device.makeBuffer(length: max(instanceSizes.buildScratchBufferSize, 4),
                                                  options: .storageModePrivate) else {
            print("Could not allocate instance acceleration structure")
            return
        }

        guard let instCmdBuf = commandQueue.makeCommandBuffer(),
              let instEncoder = instCmdBuf.makeAccelerationStructureCommandEncoder() else { return }
        instEncoder.build(accelerationStructure: finalAS,
                          descriptor: instanceDesc,
                          scratchBuffer: instScratch,
                          scratchBufferOffset: 0)
        instEncoder.endEncoding()
        instCmdBuf.commit()
        instCmdBuf.waitUntilCompleted()

        self.primitiveAccelerationStructures = primitiveStructures
        self.instanceAccelerationStructure = finalAS
        print("Top-Level Acceleration Structure built successfully.")
    }

    func addMesh(vertexBuffer: MTLBuffer, vertexOffset: Int = 0,
                 indexBuffer: MTLBuffer,  indexOffset: Int  = 0,
                 indexCount: UInt32, isUInt32: Bool,
                 material: InstanceMaterial,
                 position: SIMD3<Float> = .zero,
                 rotation: SIMD3<Float> = .zero,
                 scale: SIMD3<Float> = SIMD3(1, 1, 1)) {
        objects.addBack(vertexBuffer: vertexBuffer,
                        vertexOffset: vertexOffset,
                        indexBuffer:  indexBuffer,
                        indexOffset:  indexOffset,
                        indexCount:   indexCount,
                        isUInt32:     isUInt32,
                        position:     position,
                        rotation:     rotation,
                        scale:        scale,
                        material:     material)
    }

    func resize(width: Float, height: Float) {
        let w = max(1.0, width)
        let h = max(1.0, height)
        if viewportSize.x != w || viewportSize.y != h {
            viewportSize = SIMD2(w, h)
            createAccumulationTexture(width: Int(w), height: Int(h))
            frameIndex = 0
        }
    }

    private func createAccumulationTexture(width: Int, height: Int) {
        // Full resolution rendering
        let renderWidth = max(1, width)
        let renderHeight = max(1, height)

        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba32Float,
            width: renderWidth,
            height: renderHeight,
            mipmapped: false
        )
        desc.usage = [.shaderRead, .shaderWrite]
        desc.storageMode = .private
        accumulationTexture = device.makeTexture(descriptor: desc)
        denoisedAccumulationTexture = device.makeTexture(descriptor: desc)

        let normalDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba32Float,
            width: renderWidth,
            height: renderHeight,
            mipmapped: false
        )
        normalDesc.usage = [.shaderRead, .shaderWrite]
        normalDesc.storageMode = .private
        normalTexture = device.makeTexture(descriptor: normalDesc)

        let depthDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba32Float,
            width: renderWidth,
            height: renderHeight,
            mipmapped: false
        )
        depthDesc.usage = [.shaderRead, .shaderWrite]
        depthDesc.storageMode = .private
        depthTexture = device.makeTexture(descriptor: depthDesc)

        let albedoDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba32Float,
            width: renderWidth,
            height: renderHeight,
            mipmapped: false
        )
        albedoDesc.usage = [.shaderRead, .shaderWrite]
        albedoDesc.storageMode = .private
        albedoTexture = device.makeTexture(descriptor: albedoDesc)

        // Persistent output texture (.bgra8Unorm) matching MTKView format
        let outDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm,
            width: renderWidth,
            height: renderHeight,
            mipmapped: false
        )
        outDesc.usage = [.shaderRead, .shaderWrite]
        outDesc.storageMode = .private
        outputTexture = device.makeTexture(descriptor: outDesc)

        // Allocate CPU/GPU Shared Buffer for exchange: 16 * 5 = 80 bytes per pixel
        let pixelCount = renderWidth * renderHeight
        let exchangeBufSize = pixelCount * 80
        denoiseExchangeBuffer = device.makeBuffer(length: exchangeBufSize, options: .storageModeShared)

        print("Accumulation, G-Buffer, Denoise, and Output buffers resized (1:1): \(renderWidth)x\(renderHeight)")
    }

    func move(w: Bool, a: Bool, s: Bool, d: Bool, space: Bool, lCtrl: Bool, deltaTime: Float) {
        // Only allow camera movements in Setup Mode
        guard renderMode == .setup else { return }
        
        let v = 5.0 * deltaTime
        camera.moveFront(w ? v : (s ? -v : 0))
        camera.moveSide(d ? v : (a ? -v : 0))
        camera.moveUp(space ? v : (lCtrl ? -v : 0))
    }

    func rotateCamera(deltaX: Float, deltaY: Float) {
        // Only allow camera rotations in Setup Mode
        guard renderMode == .setup else { return }
        camera.rotate(mouseDeltaX: deltaX, mouseDeltaY: deltaY)
    }

    // MARK: - Draw Loop

    func draw(in view: MTKView) {
        guard let drawable = view.currentDrawable,
              let renderPassDescriptor = view.currentRenderPassDescriptor else {
            return
        }
        
        let width = Int(viewportSize.x)
        let height = Int(viewportSize.y)
        guard width > 0 && height > 0 else { return }
        
        if renderMode == .setup {
            // ─── Setup Mode: Standard Rasterization ───
            guard let cmdBuffer = commandQueue.makeCommandBuffer(),
                  let encoder = cmdBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) else {
                return
            }
            
            encoder.setRenderPipelineState(rasterPipelineState)
            encoder.setDepthStencilState(rasterDepthState)
            
            let aspect = viewportSize.x / viewportSize.y
            let projMatrix = makePerspectiveMatrix(fovRad: 60 * .pi / 180, aspect: aspect, nearZ: 0.1, farZ: 200)
            let viewMatrix = camera.viewMatrix
            
            // Render all geometry sequentially
            for i in 0..<objects.count {
                let obj = objects[i]
                guard let vb = obj.vertexBuffer, let ib = obj.indexBuffer else { continue }
                
                var uniforms = RasterUniforms()
                let model = makeTransformMatrix(position: obj.position, rotation: obj.rotation, scale: obj.scale)
                uniforms.modelMatrix = model
                uniforms.viewMatrix = viewMatrix
                uniforms.projectionMatrix = projMatrix
                uniforms.color = SIMD4<Float>(obj.material.color, 1.0)
                uniforms.materialType = obj.material.materialType
                
                encoder.setVertexBytes(&uniforms, length: MemoryLayout<RasterUniforms>.stride, index: 1)
                encoder.setVertexBuffer(vb, offset: obj.vertexOffset, index: 0)
                
                encoder.drawIndexedPrimitives(
                    type: .triangle,
                    indexCount: Int(obj.indexCount),
                    indexType: obj.isUInt32Index ? .uint32 : .uint16,
                    indexBuffer: ib,
                    indexBufferOffset: obj.indexOffset
                )
            }
            
            encoder.endEncoding()
            cmdBuffer.present(drawable)
            cmdBuffer.commit()
            
        } else {
            // ─── Tracing Mode: Progressive Path Tracing ───
            guard let cmdBuffer = commandQueue.makeCommandBuffer(),
                  let accelStruct = instanceAccelerationStructure,
                  let accumTexture = accumulationTexture,
                  let normalTex = normalTexture,
                  let depthTex = depthTexture,
                  let albedoTex = albedoTexture,
                  let denoisedAccumTex = denoisedAccumulationTexture,
                  let exchangeBuf = denoiseExchangeBuffer else { return }

            let renderWidth = accumTexture.width
            let renderHeight = accumTexture.height

            // Calculate camera matrix only when we are path tracing and not paused
            if !isPaused {
                if camera.isDirty {
                    frameIndex = 0
                    camera.resetDirty()
                }

                let viewMatrix = camera.viewMatrix
                let aspect = viewportSize.x / viewportSize.y
                let projMatrix = makePerspectiveMatrix(fovRad: 60 * .pi / 180, aspect: aspect, nearZ: 0.1, farZ: 200)

                var rng = SystemRandomNumberGenerator()
                let cryptoSeed = rng.next() as UInt32

                var uniforms = RayUniforms()
                uniforms.inverseViewMatrix = viewMatrix.inverse
                uniforms.inverseProjectionMatrix = projMatrix.inverse
                uniforms.cameraPosition = camera.position
                uniforms.frameIndex = frameIndex
                uniforms.randomSeed = cryptoSeed

                // 1. Ray Tracing Compute Pass
                guard let encoder = cmdBuffer.makeComputeCommandEncoder() else {
                    cmdBuffer.commit()
                    return
                }
                encoder.setComputePipelineState(computePipelineState)
                encoder.setTexture(accumTexture, index: 0)
                encoder.setTexture(normalTex, index: 1)
                encoder.setTexture(depthTex, index: 2)
                encoder.setTexture(albedoTex, index: 3)
                encoder.setBytes(&uniforms, length: MemoryLayout<RayUniforms>.stride, index: 0)
                encoder.setAccelerationStructure(accelStruct, bufferIndex: 1)
                
                if let geomBuf = geometryArgumentBuffer {
                    encoder.setBuffer(geomBuf, offset: 0, index: 2)
                }
                if let matBuf = materialBuffer {
                    encoder.setBuffer(matBuf, offset: 0, index: 3)
                }
                
                if !allGeometryBuffers.isEmpty {
                    encoder.useResources(allGeometryBuffers, usage: .read)
                }
                encoder.useResources(primitiveAccelerationStructures, usage: .read)

                let threadsPerGroup = MTLSize(width: 8, height: 8, depth: 1)
                let totalThreads = MTLSize(width: renderWidth, height: renderHeight, depth: 1)
                encoder.dispatchThreads(totalThreads, threadsPerThreadgroup: threadsPerGroup)
                encoder.endEncoding()
                
                self.frameIndex += 1
            }

            // 2. Presenting (With or Without MLX Denoising)
            if isDenoisingActive {
                // Export color & G-Buffers to exchange buffer via Blit encoder
                guard let blitEncoder = cmdBuffer.makeBlitCommandEncoder() else {
                    cmdBuffer.commit()
                    return
                }
                
                let pixelCount = renderWidth * renderHeight
                let size16 = pixelCount * 16
                
                // Copy Color
                blitEncoder.copy(from: accumTexture, sourceSlice: 0, sourceLevel: 0,
                                 sourceOrigin: MTLOriginMake(0, 0, 0), sourceSize: MTLSizeMake(renderWidth, renderHeight, 1),
                                 to: exchangeBuf, destinationOffset: 0, destinationBytesPerRow: renderWidth * 16, destinationBytesPerImage: size16)
                
                // Copy Normal
                blitEncoder.copy(from: normalTex, sourceSlice: 0, sourceLevel: 0,
                                 sourceOrigin: MTLOriginMake(0, 0, 0), sourceSize: MTLSizeMake(renderWidth, renderHeight, 1),
                                 to: exchangeBuf, destinationOffset: size16, destinationBytesPerRow: renderWidth * 16, destinationBytesPerImage: size16)
                
                // Copy Depth
                blitEncoder.copy(from: depthTex, sourceSlice: 0, sourceLevel: 0,
                                 sourceOrigin: MTLOriginMake(0, 0, 0), sourceSize: MTLSizeMake(renderWidth, renderHeight, 1),
                                 to: exchangeBuf, destinationOffset: size16 * 2, destinationBytesPerRow: renderWidth * 16, destinationBytesPerImage: size16)
                
                // Copy Albedo
                blitEncoder.copy(from: albedoTex, sourceSlice: 0, sourceLevel: 0,
                                 sourceOrigin: MTLOriginMake(0, 0, 0), sourceSize: MTLSizeMake(renderWidth, renderHeight, 1),
                                 to: exchangeBuf, destinationOffset: size16 * 3, destinationBytesPerRow: renderWidth * 16, destinationBytesPerImage: size16)
                
                blitEncoder.endEncoding()
                
                // Commit commands and wait for G-buffer export to complete
                cmdBuffer.commit()
                cmdBuffer.waitUntilCompleted()

                // Run MLX A-trous wavelet denoising
                let contents = exchangeBuf.contents()
                let colorPtr = contents
                let normalPtr = contents + size16
                let depthPtr = contents + size16 * 2
                let albedoPtr = contents + size16 * 3
                let outPtr = contents + size16 * 4
                
                denoiser.denoise(width: renderWidth, height: renderHeight,
                                 colorPtr: colorPtr,
                                 normalPtr: normalPtr,
                                 depthPtr: depthPtr,
                                 albedoPtr: albedoPtr,
                                 outPtr: outPtr)
                
                // Upload denoised array back to GPU texture
                guard let finalCmdBuffer = commandQueue.makeCommandBuffer(),
                      let uploadBlit = finalCmdBuffer.makeBlitCommandEncoder() else { return }
                
                uploadBlit.copy(from: exchangeBuf, sourceOffset: size16 * 4, sourceBytesPerRow: renderWidth * 16, sourceBytesPerImage: size16,
                                sourceSize: MTLSizeMake(renderWidth, renderHeight, 1),
                                to: denoisedAccumTex, destinationSlice: 0, destinationLevel: 0, destinationOrigin: MTLOriginMake(0, 0, 0))
                uploadBlit.endEncoding()
                
                // Tonemap & upscale denoised accumulation texture to outputTexture
                guard let finalEncoder = finalCmdBuffer.makeComputeCommandEncoder() else {
                    finalCmdBuffer.commit()
                    return
                }
                finalEncoder.setComputePipelineState(postProcessPipelineState)
                finalEncoder.setTexture(denoisedAccumTex, index: 0)
                finalEncoder.setTexture(outputTexture, index: 1)
                
                let threadsPerGroup = MTLSize(width: 8, height: 8, depth: 1)
                let postTotalThreads = MTLSize(width: width, height: height, depth: 1)
                finalEncoder.dispatchThreads(postTotalThreads, threadsPerThreadgroup: threadsPerGroup)
                finalEncoder.endEncoding()
                
                // Copy outputTexture -> drawable.texture via Blit
                guard let presentBlit = finalCmdBuffer.makeBlitCommandEncoder() else {
                    finalCmdBuffer.commit()
                    return
                }
                presentBlit.copy(from: outputTexture!, to: drawable.texture)
                presentBlit.endEncoding()
                
                finalCmdBuffer.present(drawable)
                finalCmdBuffer.commit()
            } else {
                // Tonemap raw progressive accumulation texture directly to outputTexture
                guard let bypassEncoder = cmdBuffer.makeComputeCommandEncoder() else {
                    cmdBuffer.commit()
                    return
                }
                bypassEncoder.setComputePipelineState(postProcessPipelineState)
                bypassEncoder.setTexture(accumTexture, index: 0)
                bypassEncoder.setTexture(outputTexture, index: 1)
                
                let threadsPerGroup = MTLSize(width: 8, height: 8, depth: 1)
                let postTotalThreads = MTLSize(width: width, height: height, depth: 1)
                bypassEncoder.dispatchThreads(postTotalThreads, threadsPerThreadgroup: threadsPerGroup)
                bypassEncoder.endEncoding()
                
                // Copy outputTexture -> drawable.texture via Blit
                guard let presentBlit = cmdBuffer.makeBlitCommandEncoder() else {
                    cmdBuffer.commit()
                    return
                }
                presentBlit.copy(from: outputTexture!, to: drawable.texture)
                presentBlit.endEncoding()
                
                cmdBuffer.present(drawable)
                cmdBuffer.commit()
            }
        }
    }

    private func makePerspectiveMatrix(fovRad: Float, aspect: Float, nearZ: Float, farZ: Float) -> simd_float4x4 {
        let ys = 1 / tan(fovRad * 0.5)
        let xs = ys / aspect
        let zs = farZ / (nearZ - farZ)
        return simd_float4x4(columns: (
            SIMD4(xs, 0, 0, 0),
            SIMD4(0, ys, 0, 0),
            SIMD4(0, 0, zs, -1),
            SIMD4(0, 0, zs * nearZ, 0)
        ))
    }

    private func makeTransformMatrix(position: SIMD3<Float>, rotation: SIMD3<Float>, scale: SIMD3<Float>) -> simd_float4x4 {
        let translation = simd_float4x4(columns: (
            SIMD4(1, 0, 0, 0),
            SIMD4(0, 1, 0, 0),
            SIMD4(0, 0, 1, 0),
            SIMD4(position.x, position.y, position.z, 1)
        ))
        
        let scaleMat = simd_float4x4(columns: (
            SIMD4(scale.x, 0, 0, 0),
            SIMD4(0, scale.y, 0, 0),
            SIMD4(0, 0, scale.z, 0),
            SIMD4(0, 0, 0, 1)
        ))
        
        let radX = rotation.x * .pi / 180.0
        let radY = rotation.y * .pi / 180.0
        let radZ = rotation.z * .pi / 180.0
        
        let rotX = simd_float4x4(columns: (
            SIMD4(1, 0, 0, 0),
            SIMD4(0, cos(radX), sin(radX), 0),
            SIMD4(0, -sin(radX), cos(radX), 0),
            SIMD4(0, 0, 0, 1)
        ))
        
        let rotY = simd_float4x4(columns: (
            SIMD4(cos(radY), 0, -sin(radY), 0),
            SIMD4(0, 1, 0, 0),
            SIMD4(sin(radY), 0, cos(radY), 0),
            SIMD4(0, 0, 0, 1)
        ))
        
        let rotZ = simd_float4x4(columns: (
            SIMD4(cos(radZ), sin(radZ), 0, 0),
            SIMD4(-sin(radZ), cos(radZ), 0, 0),
            SIMD4(0, 0, 1, 0),
            SIMD4(0, 0, 0, 1)
        ))
        
        let rotationMat = rotZ * rotY * rotX
        
        return translation * rotationMat * scaleMat
    }

    // MARK: - Image Export

    func exportImage(to url: URL) {
        guard let outTex = outputTexture else {
            print("Export failed: Output texture is not allocated yet")
            return
        }
        
        let width = outTex.width
        let height = outTex.height
        
        let bytesPerPixel = 4 // BGRA8
        let bytesPerRow = width * bytesPerPixel
        let imageSize = bytesPerRow * height
        
        guard let cpuBuffer = device.makeBuffer(length: imageSize, options: .storageModeShared) else {
            print("Export failed: Could not allocate CPU-readable buffer")
            return
        }
        
        // Copy Texture to Shared Buffer via Blit Command Encoder
        guard let cmdBuffer = commandQueue.makeCommandBuffer(),
              let blitEncoder = cmdBuffer.makeBlitCommandEncoder() else {
            return
        }
        
        blitEncoder.copy(from: outTex, sourceSlice: 0, sourceLevel: 0,
                         sourceOrigin: MTLOriginMake(0, 0, 0), sourceSize: MTLSizeMake(width, height, 1),
                         to: cpuBuffer, destinationOffset: 0, destinationBytesPerRow: bytesPerRow, destinationBytesPerImage: imageSize)
        blitEncoder.endEncoding()
        
        cmdBuffer.commit()
        cmdBuffer.waitUntilCompleted()
        
        // Create CGImage from BGRA bytes
        let dataProvider = CGDataProvider(data: NSData(bytes: cpuBuffer.contents(), length: imageSize))!
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        // BGRA format corresponds to Little Endian + Premultiplied First in CoreGraphics
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue)
        
        guard let cgImage = CGImage(
            width: width,
            height: height,
            bitsPerComponent: 8,
            bitsPerPixel: 32,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: bitmapInfo,
            provider: dataProvider,
            decode: nil,
            shouldInterpolate: false,
            intent: .defaultIntent
        ) else {
            print("Export failed: Could not instantiate CGImage")
            return
        }
        
        // Save as PNG
        let imageRep = NSBitmapImageRep(cgImage: cgImage)
        guard let pngData = imageRep.representation(using: .png, properties: [:]) else {
            print("Export failed: Could not format data as PNG")
            return
        }
        
        do {
            try pngData.write(to: url)
            print("Success: Image successfully exported to \(url.path)")
        } catch {
            print("Export failed: Could not write file to disk. \(error)")
        }
    }

    // MARK: - OBJ Asset Loading

    enum ModelType: String, Decodable {
        case `default`   = "Default"
        case bulb        = "Bulb"
        case liquidGlass = "LiquidGlass"
    }

    struct ModelEntry: Decodable {
        let name: String
        let type: ModelType
        let color: [Float]?
        let emissionPower: Float?
        let ior: Float?
        let absorption: Float?
        let position: [Float]?
        let rotation: [Float]?
        let scale: [Float]?
    }

    func loadScene() {
        guard let url = Bundle.main.url(forResource: "Models", withExtension: "plist"),
              let data = try? Data(contentsOf: url),
              let entries = try? PropertyListDecoder().decode([ModelEntry].self, from: data) else {
            print("Models.plist not found in bundle or invalid")
            return
        }
        for entry in entries {
            loadModel(entry: entry)
        }
    }

    private func loadModel(entry: ModelEntry) {
        let name = entry.name
        guard let url = Bundle.main.url(forResource: name, withExtension: "obj") else {
            print("\(name).obj not found in main bundle")
            return
        }

        let allocator = MTKMeshBufferAllocator(device: device)
        retainedAllocators.append(allocator)

        let vertexDescriptor = MDLVertexDescriptor()
        vertexDescriptor.attributes[0] = MDLVertexAttribute(name: MDLVertexAttributePosition,
                                                            format: .float3,
                                                            offset: 0,
                                                            bufferIndex: 0)
        vertexDescriptor.attributes[1] = MDLVertexAttribute(name: MDLVertexAttributeNormal,
                                                            format: .float3,
                                                            offset: 16,
                                                            bufferIndex: 0)
        vertexDescriptor.layouts[0] = MDLVertexBufferLayout(stride: 32)

        let asset = MDLAsset(url: url, vertexDescriptor: vertexDescriptor, bufferAllocator: allocator)
        retainedAssets.append(asset)
        
        do {
            let (_, mtkMeshes) = try MTKMesh.newMeshes(asset: asset, device: device)
            for mesh in mtkMeshes {
                retainedMeshes.append(mesh)
                let vertexBuffer = mesh.vertexBuffers[0].buffer
                let vertexOffset = mesh.vertexBuffers[0].offset

                let defaultColor = { () -> SIMD3<Float> in
                    switch entry.type {
                    case .bulb:        return SIMD3(1.0, 0.92, 0.65)
                    case .liquidGlass: return SIMD3(0.1, 0.05, 0.0) // 1.0 - [0.9, 0.95, 1.0]
                    default:           return SIMD3(0.3, 0.3, 0.3) // 1.0 - [0.7, 0.7, 0.7]
                    }
                }()

                let color = { () -> SIMD3<Float> in
                    if let c = entry.color, c.count >= 3 {
                        return SIMD3(c[0], c[1], c[2])
                    }
                    return defaultColor
                }()

                let matProps = InstanceMaterial(
                    materialType: { () -> UInt32 in
                        switch entry.type {
                        case .bulb:        return MaterialType.bulb
                        case .liquidGlass: return MaterialType.glass
                        default:           return MaterialType.opaque
                        }
                    }(),
                    emissionPower: entry.emissionPower ?? 0.0,
                    ior: entry.ior ?? 1.5,
                    absorption: entry.absorption ?? 0.05,
                    color: color,
                    _pad: 0
                )

                let pos = { () -> SIMD3<Float> in
                    if let p = entry.position, p.count >= 3 {
                        return SIMD3(p[0], p[1], p[2])
                    }
                    if entry.type == .bulb {
                        return SIMD3(0, -19.0, 0)
                    }
                    return .zero
                }()
                
                let rot = { () -> SIMD3<Float> in
                    if let r = entry.rotation, r.count >= 3 {
                        return SIMD3(r[0], r[1], r[2])
                    }
                    return .zero
                }()
                
                let scl = { () -> SIMD3<Float> in
                    if let s = entry.scale, s.count >= 3 {
                        return SIMD3(s[0], s[1], s[2])
                    }
                    return SIMD3(1, 1, 1)
                }()

                for submesh in mesh.submeshes {
                    addMesh(
                        vertexBuffer: vertexBuffer,
                        vertexOffset: vertexOffset,
                        indexBuffer:  submesh.indexBuffer.buffer,
                        indexOffset:  submesh.indexBuffer.offset,
                        indexCount:   UInt32(submesh.indexCount),
                        isUInt32:     submesh.indexType == .uint32,
                        material:     matProps,
                        position:     pos,
                        rotation:     rot,
                        scale:        scl
                    )
                }
            }
            print("Successfully loaded model: \(name).obj")
        } catch {
            print("Error loading \(name).obj: \(error)")
        }
    }
}

// Swift representation matching Shaders.metal structure
struct RasterUniforms {
    var modelMatrix: matrix_float4x4 = matrix_identity_float4x4
    var viewMatrix: matrix_float4x4 = matrix_identity_float4x4
    var projectionMatrix: matrix_float4x4 = matrix_identity_float4x4
    var color: SIMD4<Float> = .zero
    var materialType: UInt32 = 0
    var padding0: Float = 0
    var padding1: Float = 0
    var padding2: Float = 0
}
