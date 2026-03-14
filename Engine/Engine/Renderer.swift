//
//  Renderer.swift
//  LetsUseSwift
//
//  Created by Sora Sugiyama on 2/23/26.
//

import Foundation
import Metal
import GameEngine

public class GameRenderer
{
    private var unsafeRenderer: UnsafeMutablePointer<Renderer>!
    
    public init?(device: MTLDevice) {
        let pDevice = Unmanaged.passUnretained(device).toOpaque()
        
        guard let pUnsafeRenderer = Renderer.create(pDevice) else {
            return nil;
        }

        
        self.unsafeRenderer = pUnsafeRenderer;
    }
    
    deinit {
        self.unsafeRenderer.pointee.destroy()
    }
    
    public func resize(width: Float, height: Float) {
        unsafeRenderer.pointee.resize(width, height)
    }
    
    public func draw(drawable: MTLDrawable, renderPassDescriptor: MTLRenderPassDescriptor) {
        let pDrawable = Unmanaged.passUnretained(drawable).toOpaque()
        let pRenderPassDescriptor = Unmanaged.passUnretained(renderPassDescriptor).toOpaque()
        
        unsafeRenderer.pointee.draw(pDrawable, pRenderPassDescriptor)
    }
    
    public func move(w: Bool, a: Bool, s: Bool, d: Bool, space: Bool, lCtrl: Bool, deltaTime: Float) {
        unsafeRenderer.pointee.move(w, a, s, d, space, lCtrl, deltaTime)
    }
    
    public func rotateCamera(deltaX: Float, deltaY: Float) {
        unsafeRenderer.pointee.rotateCamera(deltaX, deltaY)
    }
    
    public func moveToCenter() {
        unsafeRenderer.pointee.moveToCenter()
    }
    
    public func addMesh(vertexBuffer: MTLBuffer, indexBuffer: MTLBuffer, n: UInt32, isUInt32: Bool) {
        let pV = Unmanaged.passUnretained(vertexBuffer).toOpaque()
        let pI = Unmanaged.passUnretained(indexBuffer).toOpaque()
        
        unsafeRenderer.pointee.addMesh(pV, pI, n, isUInt32)
    }
}
