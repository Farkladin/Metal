//
//  MeshHelper.swift
//  SimpleEngine
//
//  Procedural mesh builders.
//  Vertex format: float3 position (+ 1 float pad) + float3 normal (+ 1 float pad) = 32 bytes
//

import Metal

enum MeshHelper {

    // Returns (vertexBuffer, indexBuffer, indexCount) or nil on failure.
    // Indices are UInt16 — pass isUInt32: false to addMesh().

    static func makePlane(device: MTLDevice, halfSize: Float = 5) -> (MTLBuffer, MTLBuffer, UInt32)? {
        let s = halfSize
        // [x, y, z, pad,  nx, ny, nz, pad]
        var verts: [Float] = [
            -s, 0, -s, 0,   0, 1, 0, 0,
             s, 0, -s, 0,   0, 1, 0, 0,
             s, 0,  s, 0,   0, 1, 0, 0,
            -s, 0,  s, 0,   0, 1, 0, 0,
        ]
        var idx: [UInt16] = [0, 1, 2,  0, 2, 3]
        return makeBuffers(device: device, verts: &verts, idx: &idx)
    }

    static func makeCube(device: MTLDevice,
                         center: (x: Float, y: Float, z: Float) = (0, 0.5, 0),
                         size: Float = 1) -> (MTLBuffer, MTLBuffer, UInt32)? {
        let (cx, cy, cz) = center
        let h = size / 2

        // 6 faces × 4 vertices, each [x,y,z,0, nx,ny,nz,0]
        var verts: [Float] = [
            // +Y (top)
            cx-h, cy+h, cz-h, 0,  0, 1, 0, 0,
            cx+h, cy+h, cz-h, 0,  0, 1, 0, 0,
            cx+h, cy+h, cz+h, 0,  0, 1, 0, 0,
            cx-h, cy+h, cz+h, 0,  0, 1, 0, 0,
            // -Y (bottom)
            cx-h, cy-h, cz+h, 0,  0,-1, 0, 0,
            cx+h, cy-h, cz+h, 0,  0,-1, 0, 0,
            cx+h, cy-h, cz-h, 0,  0,-1, 0, 0,
            cx-h, cy-h, cz-h, 0,  0,-1, 0, 0,
            // +X (right)
            cx+h, cy-h, cz-h, 0,  1, 0, 0, 0,
            cx+h, cy-h, cz+h, 0,  1, 0, 0, 0,
            cx+h, cy+h, cz+h, 0,  1, 0, 0, 0,
            cx+h, cy+h, cz-h, 0,  1, 0, 0, 0,
            // -X (left)
            cx-h, cy-h, cz+h, 0, -1, 0, 0, 0,
            cx-h, cy-h, cz-h, 0, -1, 0, 0, 0,
            cx-h, cy+h, cz-h, 0, -1, 0, 0, 0,
            cx-h, cy+h, cz+h, 0, -1, 0, 0, 0,
            // +Z (front)
            cx-h, cy-h, cz+h, 0,  0, 0, 1, 0,
            cx+h, cy-h, cz+h, 0,  0, 0, 1, 0,
            cx+h, cy+h, cz+h, 0,  0, 0, 1, 0,
            cx-h, cy+h, cz+h, 0,  0, 0, 1, 0,
            // -Z (back)
            cx+h, cy-h, cz-h, 0,  0, 0,-1, 0,
            cx-h, cy-h, cz-h, 0,  0, 0,-1, 0,
            cx-h, cy+h, cz-h, 0,  0, 0,-1, 0,
            cx+h, cy+h, cz-h, 0,  0, 0,-1, 0,
        ]

        var idx: [UInt16] = []
        for face in 0..<6 {
            let b = UInt16(face * 4)
            idx += [b, b+1, b+2,  b, b+2, b+3]
        }

        return makeBuffers(device: device, verts: &verts, idx: &idx)
    }

    // MARK: - Private

    private static func makeBuffers(device: MTLDevice,
                                    verts: inout [Float],
                                    idx: inout [UInt16]) -> (MTLBuffer, MTLBuffer, UInt32)? {
        guard let vb = device.makeBuffer(bytes: &verts,
                                         length: verts.count * MemoryLayout<Float>.size,
                                         options: .storageModeShared),
              let ib = device.makeBuffer(bytes: &idx,
                                         length: idx.count * MemoryLayout<UInt16>.size,
                                         options: .storageModeShared)
        else { return nil }
        return (vb, ib, UInt32(idx.count))
    }
}
