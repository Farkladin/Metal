//
//  GameObject.swift
//  SimpleEngine
//
//  Swift port of the C++ GameObject / ObjectList
//

import Metal

// Material type constants (must match shader)
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
    var y0: Float = 0
    var material: MeshMaterialProps = MeshMaterialProps(
        materialType: MaterialType.opaque,
        color: SIMD3(0.7, 0.7, 0.7),
        emissionPower: 0,
        ior: 1.5,
        absorption: 0.02
    )
}

final class ObjectList {
    private static let maxSize = 1024

    private var arr: ContiguousArray<GameObject>
    private(set) var count: Int = 0

    init() {
        arr = ContiguousArray(repeating: GameObject(), count: ObjectList.maxSize)
    }

    @discardableResult
    func addBack(vertexBuffer: MTLBuffer, vertexOffset: Int = 0,
                 indexBuffer: MTLBuffer,  indexOffset: Int = 0,
                 indexCount: UInt32, isUInt32: Bool,
                 y0: Float = 0, material: MeshMaterialProps) -> Bool {
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
            y0:            y0,
            material:      material
        )
        count += 1
        return true
    }

    func pop(at index: Int) {
        guard index >= 0 && index < count else { return }
        count -= 1
        if index != count {
            arr[index] = arr[count]
        }
        arr[count] = GameObject()
    }

    subscript(index: Int) -> GameObject { arr[index] }
}
