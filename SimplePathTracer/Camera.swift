//
//  Camera.swift
//  SimpleRayTracer
//
//  Swift class representing the camera with dirty state tracking.
//

import simd

final class Camera {
        private(set) var position: SIMD3<Float>
    private var unitDeltaFront: SIMD3<Float> = SIMD3(0, 0, -1)
    private var unitDeltaSide: SIMD3<Float> = SIMD3(1, 0, 0)
    private let unitDeltaUp: SIMD3<Float> = SIMD3(0, 1, 0)
    private var yaw: Float
    private var pitch: Float
    
    // Tracks if camera properties have changed since last reset
    var isDirty: Bool = true

    // Cache deg-to-rad conversion
    private static let deg2Rad: Float = .pi / 180

    init() {
        position = SIMD3(0, 1.5, 6)
        yaw   = -90
        pitch = 0
        update()
    }

    func moveToCenter() {
        position = SIMD3(0, 1.5, 6)
        yaw   = -90
        pitch = 0
        update()
        isDirty = true
    }

    func setPreset(position: SIMD3<Float>, yaw: Float, pitch: Float) {
        self.position = position
        self.yaw = yaw
        self.pitch = pitch
        update()
        isDirty = true
    }

    func moveFront(_ delta: Float) {
        if delta != 0 {
            position += unitDeltaFront * delta
            isDirty = true
        }
    }
    
    func moveSide (_ delta: Float) {
        if delta != 0 {
            position += unitDeltaSide * delta
            isDirty = true
        }
    }
    
    func moveUp   (_ delta: Float) {
        if delta != 0 {
            position += unitDeltaUp * delta
            isDirty = true
        }
    }

    func rotate(mouseDeltaX: Float, mouseDeltaY: Float) {
        if mouseDeltaX != 0 || mouseDeltaY != 0 {
            yaw  += mouseDeltaX
            pitch = max(-89, min(89, pitch + mouseDeltaY))
            update()
            isDirty = true
        }
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

    func resetDirty() {
        isDirty = false
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
