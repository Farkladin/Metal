//
//  GameObject.swift
//  SimpleEngine
//
//  Swift port of the C++ GameObject / ObjectList
//

import Metal

struct GameObject {
    var vertexBuffer: MTLBuffer?
    var indexBuffer:  MTLBuffer?
    var indexCount:   UInt32 = 0
    var isUInt32Index: Bool  = false
    var y0: Float = 0
}

final class ObjectList {
    private static let maxSize = 1024

    private var arr: ContiguousArray<GameObject>
    private(set) var count: Int = 0

    init() {
        // ContiguousArray gives contiguous memory layout for better cache performance
        arr = ContiguousArray(repeating: GameObject(), count: ObjectList.maxSize)
    }

    @discardableResult
    func addBack(vertexBuffer: MTLBuffer, indexBuffer: MTLBuffer,
                 indexCount: UInt32, isUInt32: Bool, y0: Float = 0) -> Bool {
        guard count < ObjectList.maxSize else {
            print("ObjectList: Out of Bounds")
            return false
        }
        arr[count] = GameObject(
            vertexBuffer: vertexBuffer,
            indexBuffer:  indexBuffer,
            indexCount:   indexCount,
            isUInt32Index: isUInt32,
            y0: y0
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
