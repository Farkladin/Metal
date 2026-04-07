//
//  Camera.swift
//  SimpleEngine
//
//  Swift port of the C++ Camera class
//

import simd

final class Camera {
    private var position: SIMD3<Float>
    private var unitDeltaFront: SIMD3<Float>
    private var unitDeltaSide: SIMD3<Float>
    private let unitDeltaUp: SIMD3<Float> = SIMD3(0, 1, 0)
    private var yaw: Float
    private var pitch: Float

    // Cache deg-to-rad conversion to avoid repeated division
    private static let deg2Rad: Float = .pi / 180

    init() {
        position       = SIMD3(0, 1, 5)
        unitDeltaFront = SIMD3(1, 0, 0)
        unitDeltaSide  = SIMD3(0, 0, 1)
        yaw   = -90
        pitch = 0
        update()
    }

    func moveToCenter() {
        position = SIMD3(0, 1, 5)
        yaw   = -90
        pitch = 0
        update()
    }

    func moveFront(_ delta: Float) { position += unitDeltaFront * delta }
    func moveSide (_ delta: Float) { position += unitDeltaSide  * delta }
    func moveUp   (_ delta: Float) { position += unitDeltaUp    * delta }

    func rotate(mouseDeltaX: Float, mouseDeltaY: Float) {
        yaw  += mouseDeltaX
        pitch = max(-89, min(89, pitch + mouseDeltaY))
        update()
    }

    private func update() {
        let cosPitch = cos(pitch * Self.deg2Rad)
        let newFront = SIMD3<Float>(
            cos(yaw * Self.deg2Rad) * cosPitch,
            sin(pitch * Self.deg2Rad),
            sin(yaw * Self.deg2Rad) * cosPitch
        )
        unitDeltaFront = normalize(newFront)
        unitDeltaSide  = normalize(cross(unitDeltaFront, unitDeltaUp))
    }

    var viewMatrix: simd_float4x4 {
        let z = -unitDeltaFront
        let x = normalize(cross(unitDeltaUp, z))
        let y = cross(z, x)
        return simd_float4x4(columns: (
            SIMD4(x.x, y.x, z.x, 0),
            SIMD4(x.y, y.y, z.y, 0),
            SIMD4(x.z, y.z, z.z, 0),
            SIMD4(-dot(x, position), -dot(y, position), -dot(z, position), 1)
        ))
    }
}
