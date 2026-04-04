//
//  Renderer.swift
//  SimpleEngine
//
//  Swift port of the C++ Renderer class (shadow mapping included).
//  GameViewController drives the frame loop; this class only renders.
//

import Metal
import MetalKit
import simd

final class Renderer {

    // MARK: - Metal Objects

    let device: MTLDevice
    private let commandQueue: MTLCommandQueue

    private var pipelineState: MTLRenderPipelineState!
    private var depthStencilState: MTLDepthStencilState!

    private var shadowPipelineState: MTLRenderPipelineState!
    private var shadowDepthStencilState: MTLDepthStencilState!
    private var shadowTexture: MTLTexture!

    // MARK: - State

    private let camera = Camera()
    private var viewportSize: SIMD2<Float> = .zero
    private let objects = ObjectList()

    // MARK: - Init

    init?(device: MTLDevice) {
        guard let queue = device.makeCommandQueue() else { return nil }
        self.device = device
        self.commandQueue = queue
        buildShaders()
    }

    // MARK: - Pipeline / Shader Setup

    private func buildShaders() {
        guard let library = device.makeDefaultLibrary() else {
            print("Failed to load default shader library")
            return
        }

        let vertexDesc = makeVertexDescriptor()

        // Main pass pipeline
        let mainDesc = MTLRenderPipelineDescriptor()
        mainDesc.vertexFunction   = library.makeFunction(name: "vertexShader")
        mainDesc.fragmentFunction = library.makeFunction(name: "fragmentShader")
        mainDesc.vertexDescriptor = vertexDesc
        mainDesc.colorAttachments[0].pixelFormat = .bgra8Unorm
        mainDesc.depthAttachmentPixelFormat      = .depth32Float

        // Shadow pass pipeline (depth-only, no fragment function)
        let shadowDesc = MTLRenderPipelineDescriptor()
        shadowDesc.vertexFunction   = library.makeFunction(name: "shadowVertexShader")
        shadowDesc.fragmentFunction = nil
        shadowDesc.vertexDescriptor = vertexDesc
        shadowDesc.colorAttachments[0].pixelFormat = .invalid
        shadowDesc.depthAttachmentPixelFormat      = .depth32Float

        do {
            pipelineState       = try device.makeRenderPipelineState(descriptor: mainDesc)
            shadowPipelineState = try device.makeRenderPipelineState(descriptor: shadowDesc)
        } catch {
            print("Failed to create pipeline state: \(error)")
            return
        }

        // Main pass depth stencil
        let depthDesc = MTLDepthStencilDescriptor()
        depthDesc.depthCompareFunction = .less
        depthDesc.isDepthWriteEnabled  = true
        depthStencilState = device.makeDepthStencilState(descriptor: depthDesc)

        // Shadow pass depth stencil
        let shadowDepthDesc = MTLDepthStencilDescriptor()
        shadowDepthDesc.depthCompareFunction = .lessEqual
        shadowDepthDesc.isDepthWriteEnabled  = true
        shadowDepthStencilState = device.makeDepthStencilState(descriptor: shadowDepthDesc)

        // 2048×2048 shadow map texture
        let texDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .depth32Float,
            width: 2048, height: 2048,
            mipmapped: false
        )
        texDesc.storageMode = .private
        texDesc.usage       = [.renderTarget, .shaderRead]
        shadowTexture = device.makeTexture(descriptor: texDesc)
    }

    private func makeVertexDescriptor() -> MTLVertexDescriptor {
        // Vertex layout matching the C++ engine:
        //   attribute 0: float3 position, offset=0
        //   attribute 1: float3 normal,   offset=16 (4-byte padding after position)
        //   stride: 32
        let desc = MTLVertexDescriptor()
        desc.attributes[0].format      = .float3
        desc.attributes[0].offset      = 0
        desc.attributes[0].bufferIndex = 0
        desc.attributes[1].format      = .float3
        desc.attributes[1].offset      = 16
        desc.attributes[1].bufferIndex = 0
        desc.layouts[0].stride = 32
        return desc
    }

    // MARK: - Public API

    func addMesh(vertexBuffer: MTLBuffer, indexBuffer: MTLBuffer,
                 indexCount: UInt32, isUInt32: Bool) {
        if !objects.addBack(vertexBuffer: vertexBuffer, indexBuffer: indexBuffer,
                            indexCount: indexCount, isUInt32: isUInt32) {
            print("Mesh was not added.")
        }
    }

    func resize(width: Float, height: Float) {
        viewportSize = SIMD2(width, height)
    }

    func move(w: Bool, a: Bool, s: Bool, d: Bool,
              space: Bool, lCtrl: Bool, deltaTime: Float) {
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

    func draw(drawable: MTLDrawable, renderPassDescriptor: MTLRenderPassDescriptor) {
        guard viewportSize.x > 0 && viewportSize.y > 0,
              let cmdBuffer = commandQueue.makeCommandBuffer() else { return }

        // Compute matrices once per frame
        let lightPos: SIMD3<Float> = SIMD3(5, 5, 5)
        let lightView = makeLookAt(eye: lightPos, center: .zero, up: SIMD3(0, 1, 0))
        let lightProj = makeOrthogonalMatrix(left: -10, right: 10,
                                             bottom: -10, top: 10,
                                             nearZ: 0.01, farZ: 50)
        var lightSpaceMatrix = lightProj * lightView
        var viewMatrix       = camera.viewMatrix
        var modelMatrix      = matrix_identity_float4x4
        let aspect           = viewportSize.x / viewportSize.y
        var projMatrix       = makePerspectiveMatrix(fovRad: 45 * .pi / 180,
                                                    aspect: aspect,
                                                    nearZ: 0.1, farZ: 100)

        // MARK: Shadow Pass
        let shadowPassDesc = MTLRenderPassDescriptor()
        shadowPassDesc.depthAttachment.texture     = shadowTexture
        shadowPassDesc.depthAttachment.clearDepth  = 1.0
        shadowPassDesc.depthAttachment.loadAction  = .clear
        shadowPassDesc.depthAttachment.storeAction = .store

        if let shadowEncoder = cmdBuffer.makeRenderCommandEncoder(descriptor: shadowPassDesc) {
            shadowEncoder.setRenderPipelineState(shadowPipelineState)
            shadowEncoder.setDepthStencilState(shadowDepthStencilState)

            // Optimization: bind shared uniforms once outside the draw loop
            shadowEncoder.setVertexBytes(&modelMatrix,
                                         length: MemoryLayout<simd_float4x4>.size, index: 1)
            shadowEncoder.setVertexBytes(&lightSpaceMatrix,
                                         length: MemoryLayout<simd_float4x4>.size, index: 2)

            for i in 0..<objects.count {
                let obj = objects[i]
                guard let vb = obj.vertexBuffer, let ib = obj.indexBuffer else { continue }
                shadowEncoder.setVertexBuffer(vb, offset: 0, index: 0)
                shadowEncoder.drawIndexedPrimitives(
                    type: .triangle,
                    indexCount: Int(obj.indexCount),
                    indexType: obj.isUInt32Index ? .uint32 : .uint16,
                    indexBuffer: ib,
                    indexBufferOffset: 0
                )
            }
            shadowEncoder.endEncoding()
        }

        // MARK: Main Pass
        if let encoder = cmdBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) {
            encoder.setRenderPipelineState(pipelineState)
            encoder.setDepthStencilState(depthStencilState)

            // Optimization: bind shared uniforms once outside the draw loop
            encoder.setVertexBytes(&modelMatrix,
                                   length: MemoryLayout<simd_float4x4>.size, index: 1)
            encoder.setVertexBytes(&viewMatrix,
                                   length: MemoryLayout<simd_float4x4>.size, index: 2)
            encoder.setVertexBytes(&projMatrix,
                                   length: MemoryLayout<simd_float4x4>.size, index: 3)
            encoder.setVertexBytes(&lightSpaceMatrix,
                                   length: MemoryLayout<simd_float4x4>.size, index: 4)
            encoder.setFragmentTexture(shadowTexture, index: 0)

            for i in 0..<objects.count {
                let obj = objects[i]
                guard let vb = obj.vertexBuffer, let ib = obj.indexBuffer else { continue }
                encoder.setVertexBuffer(vb, offset: 0, index: 0)
                encoder.drawIndexedPrimitives(
                    type: .triangle,
                    indexCount: Int(obj.indexCount),
                    indexType: obj.isUInt32Index ? .uint32 : .uint16,
                    indexBuffer: ib,
                    indexBufferOffset: 0
                )
            }
            encoder.endEncoding()
        }

        cmdBuffer.present(drawable)
        cmdBuffer.commit()
    }

    // MARK: - Matrix Helpers (matching C++ implementation)

    private func makeOrthogonalMatrix(left: Float, right: Float,
                                      bottom: Float, top: Float,
                                      nearZ: Float, farZ: Float) -> simd_float4x4 {
        simd_float4x4(columns: (
            SIMD4(2 / (right - left), 0, 0, 0),
            SIMD4(0, 2 / (top - bottom), 0, 0),
            SIMD4(0, 0, -1 / (farZ - nearZ), 0),
            SIMD4((left + right) / (left - right),
                  (top + bottom) / (top - bottom),
                  nearZ / (nearZ - farZ), 1)
        ))
    }

    private func makeLookAt(eye: SIMD3<Float>,
                            center: SIMD3<Float>,
                            up: SIMD3<Float>) -> simd_float4x4 {
        let z = normalize(eye - center)
        let x = normalize(cross(up, z))
        let y = cross(z, x)
        return simd_float4x4(columns: (
            SIMD4(x.x, y.x, z.x, 0),
            SIMD4(x.y, y.y, z.y, 0),
            SIMD4(x.z, y.z, z.z, 0),
            SIMD4(-dot(x, eye), -dot(y, eye), -dot(z, eye), 1)
        ))
    }

    private func makePerspectiveMatrix(fovRad: Float, aspect: Float,
                                       nearZ: Float, farZ: Float) -> simd_float4x4 {
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
}
