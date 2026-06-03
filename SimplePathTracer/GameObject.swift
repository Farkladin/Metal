//
//  GameObject.swift
//  SimpleRayTracer
//
//  Swift representations of game objects and materials.
//

import Metal
import simd

// Material types - must match the values in ShaderTypes.h and metal shader (MAT_OPAQUE, MAT_BULB, MAT_GLASS)
struct MaterialType {
    static let opaque: UInt32 = 0
    static let bulb:   UInt32 = 1
    static let glass:  UInt32 = 2
}

struct GameObject {
    var vertexBuffer: MTLBuffer?
    var vertexOffset: Int    = 0
    var indexBuffer:  MTLBuffer?
    var indexOffset:  Int    = 0
    var indexCount:   UInt32 = 0
    var isUInt32Index: Bool  = false
    
    // Transform parameters
    var position: SIMD3<Float> = .zero
    var rotation: SIMD3<Float> = .zero
    var scale: SIMD3<Float> = SIMD3(1, 1, 1)
    
    var material: InstanceMaterial = InstanceMaterial(
        materialType: MaterialType.opaque,
        emissionPower: 0,
        ior: 1.5,
        absorption: 0.02,
        color: SIMD3(0.7, 0.7, 0.7),
        _pad: 0
    )
}

final class ObjectList {
    private static let maxSize = 128

    private var arr: ContiguousArray<GameObject>
    private(set) var count: Int = 0

    init() {
        arr = ContiguousArray(repeating: GameObject(), count: ObjectList.maxSize)
    }

    @discardableResult
    func addBack(vertexBuffer: MTLBuffer, vertexOffset: Int = 0,
                 indexBuffer: MTLBuffer,  indexOffset: Int = 0,
                 indexCount: UInt32, isUInt32: Bool,
                 position: SIMD3<Float> = .zero,
                 rotation: SIMD3<Float> = .zero,
                 scale: SIMD3<Float> = SIMD3(1, 1, 1),
                 material: InstanceMaterial) -> Bool {
        guard count < ObjectList.maxSize else {
            print("ObjectList: Out of Bounds")
            return false
        }
        arr[count] = GameObject(
            vertexBuffer:  vertexBuffer,
            vertexOffset:  vertexOffset,
            indexBuffer:   indexBuffer,
            indexOffset:   indexOffset,
            indexCount:    indexCount,
            isUInt32Index: isUInt32,
            position:      position,
            rotation:      rotation,
            scale:         scale,
            material:      material
        )
        count += 1
        return true
    }


    subscript(index: Int) -> GameObject { arr[index] }
}

