//
//  GameViewController.swift
//  LetsUseSwift
//
//  Created by Sora Sugiyama on 2/13/26.
//

import Cocoa
import MetalKit
import ModelIO
import GameEngine

enum KeyCode: CGKeyCode
{
    case a = 0
    case s = 1
    case d = 2
    case w = 13
    case space = 49
    case lCtrl = 59
    case esc = 53
    case c = 8
}

class GameViewController: NSViewController
{
    var mtkView: MTKView!
    var renderer: GameRenderer!
    
    var lastFrameTime: CFTimeInterval = 0
    var isMouseAssociated: Bool = true
    private var trackingArea: NSTrackingArea?
    
    func loadModel(device: MTLDevice, name: String) {
        guard let url = Bundle.main.url(forResource: name, withExtension: "obj") else {
            fatalError(name + ".obj is not found")
        }
        
        let allocator = MTKMeshBufferAllocator(device: device)
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
        
        do {
            let (_, mtkMeshes) = try MTKMesh.newMeshes(asset: asset, device: device)
            
            for mesh in mtkMeshes {
                let vertexBuffer = mesh.vertexBuffers[0].buffer
                
                for submesh in mesh.submeshes {
                    let indexBuffer = submesh.indexBuffer.buffer
                    let numIndex = UInt32(submesh.indexCount)
                    let isUInt32 = submesh.indexType == .uint32
                    
                    self.renderer.addMesh(vertexBuffer: vertexBuffer,
                                          indexBuffer: indexBuffer,
                                          n: numIndex,
                                          isUInt32: isUInt32)
                }
            }
            print("Model loaded: \(name)")
        } catch {
            print("Failed to load model: \(name)")
        }
    }
    
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        guard let view = self.view as? MTKView else {
            print("View is not MTKView")
            return
        }
        self.mtkView = view
        
        guard let defaultDevice = MTLCreateSystemDefaultDevice() else {
            print("Metal is not supported")
            return
        }
        
        self.mtkView.device = defaultDevice
        self.mtkView.clearColor = MTLClearColor(red: 0.1, green: 0.1, blue: 0.1, alpha: 1.0)
        self.mtkView.depthStencilPixelFormat = .depth32Float
        self.mtkView.clearDepth = 1.0
        
        self.renderer = GameRenderer(device: defaultDevice)
        self.mtkView.delegate = self
        
        loadModel(device: defaultDevice, name: "pillar")
        loadModel(device: defaultDevice, name: "floatingBox")
        loadModel(device: defaultDevice, name: "simpleFloor")
        
        let size = self.mtkView.drawableSize
        self.renderer.resize(width: Float(size.width), height: Float(size.height))
        
        CGAssociateMouseAndMouseCursorPosition(boolean_t(0))
        NSCursor.hide()
        
        self.lastFrameTime = CACurrentMediaTime()
    }
    
    override func viewDidAppear() {
        super.viewDidAppear()
        self.view.window?.makeFirstResponder(self)
    }
    
    private func setupTrackingArea() {
        if let existingArea = self.trackingArea {
            self.mtkView.removeTrackingArea(existingArea)
        }
        
        let option: NSTrackingArea.Options = [.mouseEnteredAndExited, .mouseMoved, .activeInKeyWindow, .inVisibleRect]
        trackingArea = NSTrackingArea(rect: self.mtkView.bounds, options: option, owner: self, userInfo: nil)
        
        self.view.addTrackingArea(trackingArea!)
    }
    
    override func viewDidLayout() {
        super.viewDidLayout()
        
        setupTrackingArea();
    }
    
    // MARK: - Inputs
    
    override var acceptsFirstResponder: Bool {
        return true
    }
    
    override func keyDown(with event: NSEvent) {
        if event.modifierFlags.contains(.option) && event.keyCode == KeyCode.c.rawValue {
            renderer.moveToCenter()
            return
        } else if event.keyCode == KeyCode.esc.rawValue {
            if isMouseAssociated {
                CGAssociateMouseAndMouseCursorPosition(boolean_t(1))
                NSCursor.unhide();
                self.mtkView.removeTrackingArea(trackingArea!)
                isMouseAssociated = false;
            } else {
                CGAssociateMouseAndMouseCursorPosition(boolean_t(0))
                NSCursor.hide()
                setupTrackingArea()
                isMouseAssociated = true;
            }
        }
    }
    
    override func mouseMoved(with event: NSEvent) {
        let sensitivity: Float = 0.1
        let xOffset: Float = Float(event.deltaX) * sensitivity
        let yOffset: Float = Float(event.deltaY) * sensitivity
        
        renderer.rotateCamera(deltaX: xOffset, deltaY: yOffset)
    }
}


extension GameViewController: MTKViewDelegate
{
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        renderer.resize(width: Float(size.width), height: Float(size.height))
    }
    
    func draw(in view: MTKView) {
        let currentTime = CACurrentMediaTime()
        let deltaTime = Float(currentTime - lastFrameTime)
        lastFrameTime = currentTime
        
        let w: Bool = CGEventSource.keyState(.combinedSessionState, key: KeyCode.w.rawValue)
        let a: Bool = CGEventSource.keyState(.combinedSessionState, key: KeyCode.a.rawValue)
        let s: Bool = CGEventSource.keyState(.combinedSessionState, key: KeyCode.s.rawValue)
        let d: Bool = CGEventSource.keyState(.combinedSessionState, key: KeyCode.d.rawValue)
        let space: Bool = CGEventSource.keyState(.combinedSessionState, key: KeyCode.space.rawValue)
        let lCtrl: Bool = CGEventSource.keyState(.combinedSessionState, key: KeyCode.lCtrl.rawValue)
        
        renderer.move(w: w, a: a, s: s, d: d, space: space, lCtrl: lCtrl, deltaTime: deltaTime)
        
        guard let drawable = view.currentDrawable,
              let renderPassDescriptor = view.currentRenderPassDescriptor else { return }
        
        renderer.draw(drawable: drawable, renderPassDescriptor: renderPassDescriptor)
    }
}
