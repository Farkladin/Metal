//
//  Renderer.swift
//  SimpleEngine
//
//  Compute-based Ray-Tracing Pipeline with per-instance material support.
//

import Metal
import MetalKit
import simd

final class Renderer {

    let device: MTLDevice
    private let commandQueue: MTLCommandQueue

    private var computePipelineState: MTLComputePipelineState!
    private var geometryArgumentEncoder: MTLArgumentEncoder!
    private var geometryArgumentBuffer: MTLBuffer?
    private var materialBuffer: MTLBuffer?          // per-instance InstanceMaterial
    private var allGeometryBuffers: [MTLBuffer] = []
    
    private var instanceAccelerationStructure: MTLAccelerationStructure?
    private var primitiveAccelerationStructures: [MTLAccelerationStructure] = []
    private var instanceDescriptorBuffer: MTLBuffer?

    private var bulbInstanceIDs: [UInt32] = []
    private var bulbWorldPositions: [SIMD3<Float>] = []
    private var bulbColors: [SIMD3<Float>] = []

    private let camera = Camera()
    private var viewportSize: SIMD2<Float> = .zero
    private let objects = ObjectList()

    init?(device: MTLDevice) {
        guard let queue = device.makeCommandQueue() else { return nil }
        self.device = device
        self.commandQueue = queue
        buildComputePipeline()
    }

    private func buildComputePipeline() {
        guard let library = device.makeDefaultLibrary(),
              let rayKernel = library.makeFunction(name: "raytracing_kernel") else {
            print("Failed to load raytracing_kernel")
            return
        }
        do {
            computePipelineState = try device.makeComputePipelineState(function: rayKernel)
            geometryArgumentEncoder = rayKernel.makeArgumentEncoder(bufferIndex: 2)
            print("Compute Pipeline Built successfully")
        } catch {
            print("Failed to create compute pipeline state: \(error)")
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
        bulbInstanceIDs.removeAll()
        bulbWorldPositions.removeAll()
        bulbColors.removeAll()

        let encodedLen = geometryArgumentEncoder.encodedLength
        let argBufLength = encodedLen * objects.count
        guard let argBuffer = device.makeBuffer(length: argBufLength, options: .storageModeShared) else {
            print("Failed to make arg buffer")
            return
        }

        // ── Phase 1: Build PRIMITIVE acceleration structures ─────────────────
        for i in 0..<objects.count {
            let obj = objects[i]
            guard let vb = obj.vertexBuffer, let ib = obj.indexBuffer else {
                print("Object \(i) missing buffers – skipping")
                continue
            }
            guard Int(obj.indexCount) % 3 == 0, obj.indexCount > 0 else {
                print("Object \(i) invalid indexCount \(obj.indexCount) – skipping")
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
                print("Object \(i): could not allocate – skipping")
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
            let mat = obj.material

            if mat.materialType == MaterialType.bulb {
                let centroid = computeMeshCentroid(vb: vb, vertexOffset: obj.vertexOffset,
                                                   ib: ib, indexOffset: obj.indexOffset,
                                                   indexCount: obj.indexCount,
                                                   isUInt32: obj.isUInt32Index,
                                                   yOffset: obj.y0)
                bulbInstanceIDs.append(UInt32(instanceIndex))
                bulbWorldPositions.append(centroid)
                bulbColors.append(mat.color)
                print("Bulb found: instance_id = \(instanceIndex), worldPos = \(centroid), color = \(mat.color)")
            }

            // Instance Descriptor
            var desc = MTLAccelerationStructureInstanceDescriptor()
            desc.accelerationStructureIndex = UInt32(instanceIndex)
            desc.intersectionFunctionTableOffset = 0
            desc.mask = 0xFF
            desc.options = .opaque
            desc.transformationMatrix.columns.0.x = 1.0
            desc.transformationMatrix.columns.1.y = 1.0
            desc.transformationMatrix.columns.2.z = 1.0
            desc.transformationMatrix.columns.3.y = obj.y0

            // Argument Buffer
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
            var transform = matrix_identity_float4x4
            transform.columns.3.y = obj.y0
            ptrTransform.pointee = transform

            // Build InstanceMaterial for GPU
            var gpuMat = InstanceMaterial()
            gpuMat.materialType = mat.materialType
            gpuMat.emissionPower = mat.emissionPower
            gpuMat.ior = mat.ior
            gpuMat.absorption = mat.absorption
            gpuMat.color = mat.color
            instanceMaterials.append(gpuMat)

            primitiveStructures.append(primAS)
            instanceDescriptors.append(desc)
        }

        guard !primitiveStructures.isEmpty else {
            print("No valid primitive structures built – aborting")
            return
        }

        self.geometryArgumentBuffer = argBuffer
        
        // Build material buffer
        self.materialBuffer = device.makeBuffer(
            bytes: instanceMaterials,
            length: MemoryLayout<InstanceMaterial>.stride * instanceMaterials.count,
            options: .storageModeShared)

        print("Built \(primitiveStructures.count) primitive structures. Bulb IDs = \(bulbInstanceIDs)")

        // ── Phase 2: Instance acceleration structure ─────────────────────────
        guard let instanceBuffer = device.makeBuffer(
            bytes: instanceDescriptors,
            length: MemoryLayout<MTLAccelerationStructureInstanceDescriptor>.stride * instanceDescriptors.count,
            options: .storageModeShared) else {
            print("Could not allocate instance buffer")
            return
        }
        self.instanceDescriptorBuffer = instanceBuffer

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
        print("Instance Acceleration Structure built! total=\(primitiveStructures.count)")
    }

    func addMesh(vertexBuffer: MTLBuffer, vertexOffset: Int = 0,
                 indexBuffer: MTLBuffer,  indexOffset: Int  = 0,
                 indexCount: UInt32, isUInt32: Bool,
                 material: MeshMaterialProps, y0: Float = 0) {
        objects.addBack(vertexBuffer: vertexBuffer,
                        vertexOffset: vertexOffset,
                        indexBuffer:  indexBuffer,
                        indexOffset:  indexOffset,
                        indexCount:   indexCount,
                        isUInt32:     isUInt32,
                        y0:           y0,
                        material:     material)
    }

    func resize(width: Float, height: Float) {
        viewportSize = SIMD2(width, height)
    }

    func move(w: Bool, a: Bool, s: Bool, d: Bool, space: Bool, lCtrl: Bool, deltaTime: Float) {
        let v = 10.0 * deltaTime
        if w     { camera.moveFront( v) }
        if s     { camera.moveFront(-v) }
        if a     { camera.moveSide (-v) }
        if d     { camera.moveSide ( v) }
        if space { camera.moveUp( v) }
        if lCtrl { camera.moveUp(-v) }
    }

    func moveToCenter() { camera.moveToCenter() }

    func rotateCamera(deltaX: Float, deltaY: Float) {
        camera.rotate(mouseDeltaX: deltaX, mouseDeltaY: deltaY)
    }

    // MARK: - Draw
    func draw(drawable: CAMetalDrawable, renderPassDescriptor: MTLRenderPassDescriptor) {
        guard viewportSize.x > 0 && viewportSize.y > 0,
              let cmdBuffer = commandQueue.makeCommandBuffer(),
              let accelStruct = instanceAccelerationStructure else { return }

        let viewMatrix = camera.viewMatrix
        let aspect = viewportSize.x / viewportSize.y
        let projMatrix = makePerspectiveMatrix(fovRad: 60 * .pi / 180, aspect: aspect, nearZ: 0.1, farZ: 200)

        let invView = viewMatrix.inverse
        var uniforms = RayUniforms()
        uniforms.inverseViewMatrix = invView
        uniforms.inverseProjectionMatrix = projMatrix.inverse
        uniforms.cameraPosition = SIMD3(invView.columns.3.x, invView.columns.3.y, invView.columns.3.z)
        uniforms.instanceCount = UInt32(primitiveAccelerationStructures.count)
        let bulbCount = min(bulbWorldPositions.count, Int(MAX_BULBS))
        uniforms.bulbCount = UInt32(bulbCount)
        
        withUnsafeMutablePointer(to: &uniforms.bulbWorldPositions) { tuple in
            tuple.withMemoryRebound(to: SIMD3<Float>.self, capacity: Int(MAX_BULBS)) { ptr in
                for i in 0..<bulbCount { ptr[i] = bulbWorldPositions[i] }
            }
        }
        withUnsafeMutablePointer(to: &uniforms.bulbInstanceIDs) { tuple in
            tuple.withMemoryRebound(to: UInt32.self, capacity: Int(MAX_BULBS)) { ptr in
                for i in 0..<bulbCount { ptr[i] = bulbInstanceIDs[i] }
            }
        }
        withUnsafeMutablePointer(to: &uniforms.bulbColors) { tuple in
            tuple.withMemoryRebound(to: SIMD3<Float>.self, capacity: Int(MAX_BULBS)) { ptr in
                for i in 0..<bulbCount { ptr[i] = bulbColors[i] }
            }
        }

        guard let encoder = cmdBuffer.makeComputeCommandEncoder() else {
            cmdBuffer.commit(); return
        }
        encoder.setComputePipelineState(computePipelineState)
        encoder.setTexture(drawable.texture, index: 0)
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

        let threadsPerGroup = MTLSize(
            width: computePipelineState.threadExecutionWidth,
            height: computePipelineState.maxTotalThreadsPerThreadgroup / computePipelineState.threadExecutionWidth,
            depth: 1
        )
        let totalThreads = MTLSize(width: drawable.texture.width,
                                   height: drawable.texture.height,
                                   depth: 1)
        encoder.dispatchThreads(totalThreads, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()

        cmdBuffer.addCompletedHandler { cb in
            if let error = cb.error {
                print("GPU Error:", error.localizedDescription)
            }
        }

        cmdBuffer.present(drawable)
        cmdBuffer.commit()
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

    private func computeMeshCentroid(vb: MTLBuffer, vertexOffset: Int,
                                     ib: MTLBuffer, indexOffset: Int,
                                     indexCount: UInt32, isUInt32: Bool,
                                     yOffset: Float) -> SIMD3<Float> {
        let stride = 32
        let vertexBase = vb.contents().advanced(by: vertexOffset)
        var sum = SIMD3<Float>(0, 0, 0)
        var count: Float = 0
        let sampleCount = min(Int(indexCount), 300)
        
        if isUInt32 {
            let indices = ib.contents().advanced(by: indexOffset).assumingMemoryBound(to: UInt32.self)
            for j in 0..<sampleCount {
                let idx = Int(indices[j])
                let px = vertexBase.advanced(by: idx * stride).assumingMemoryBound(to: Float.self)
                sum += SIMD3(px[0], px[1], px[2])
                count += 1
            }
        } else {
            let indices = ib.contents().advanced(by: indexOffset).assumingMemoryBound(to: UInt16.self)
            for j in 0..<sampleCount {
                let idx = Int(indices[j])
                let px = vertexBase.advanced(by: idx * stride).assumingMemoryBound(to: Float.self)
                sum += SIMD3(px[0], px[1], px[2])
                count += 1
            }
        }
        
        if count > 0 { sum /= count }
        sum.y += yOffset
        return sum
    }
}
