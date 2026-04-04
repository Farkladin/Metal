//
//  GameViewController.swift
//  SimpleEngine
//
//  Drives the render loop, loads OBJ models, and handles all input.
//

import Cocoa
import MetalKit
import ModelIO
import CoreGraphics

// Material type for each mesh — used later for ray-tracing material properties
enum ModelType: String, Decodable {
    case `default`  = "Default"
    case bulb       = "Bulb"
    case liquidGlass = "LiquidGlass"
}

struct ModelEntry: Decodable {
    let name: String
    let type: ModelType
}

// Raw macOS key codes
enum KeyCode: CGKeyCode {
    case a     = 0
    case s     = 1
    case d     = 2
    case w     = 13
    case c     = 8
    case space = 49
    case lCtrl = 59
    case esc   = 53
}

class GameViewController: NSViewController {

    var mtkView: MTKView!
    var renderer: Renderer!

    var lastFrameTime: CFTimeInterval = 0
    var isMouseLocked: Bool = true       // true = game mode (cursor hidden, mouse disassociated)
    private var trackingArea: NSTrackingArea?

    // MARK: - OBJ Loading

    private func loadModels(device: MTLDevice) {
        guard let url     = Bundle.main.url(forResource: "Models", withExtension: "plist"),
              let data    = try? Data(contentsOf: url),
              let entries = try? PropertyListDecoder().decode([ModelEntry].self, from: data) else {
            print("Models.plist not found or invalid")
            return
        }
        for entry in entries {
            loadModel(device: device, entry: entry)
        }
    }

    func loadModel(device: MTLDevice, entry: ModelEntry) {
        let name = entry.name
        guard let url = Bundle.main.url(forResource: name, withExtension: "obj") else {
            print("\(name).obj not found in bundle")
            return
        }

        let allocator = MTKMeshBufferAllocator(device: device)

        // Vertex descriptor must match the render pipeline (float3 pos + pad + float3 normal, stride 32)
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

        let asset = MDLAsset(url: url,
                             vertexDescriptor: vertexDescriptor,
                             bufferAllocator: allocator)
        do {
            let (_, mtkMeshes) = try MTKMesh.newMeshes(asset: asset, device: device)
            for mesh in mtkMeshes {
                let vertexBuffer = mesh.vertexBuffers[0].buffer
                for submesh in mesh.submeshes {
                    renderer.addMesh(vertexBuffer: vertexBuffer,
                                     indexBuffer:  submesh.indexBuffer.buffer,
                                     indexCount:   UInt32(submesh.indexCount),
                                     isUInt32:     submesh.indexType == .uint32)
                }
            }
            print("Loaded: \(name).obj")
        } catch {
            print("Failed to load \(name).obj: \(error)")
        }
    }

    // MARK: - View Lifecycle

    override func viewDidLoad() {
        super.viewDidLoad()

        guard let view = self.view as? MTKView else {
            print("View is not MTKView")
            return
        }
        self.mtkView = view

        guard let device = MTLCreateSystemDefaultDevice() else {
            print("Metal is not supported")
            return
        }

        self.mtkView.device              = device
        self.mtkView.clearColor          = MTLClearColor(red: 0.1, green: 0.1, blue: 0.1, alpha: 1.0)
        self.mtkView.depthStencilPixelFormat = .depth32Float
        self.mtkView.clearDepth          = 1.0

        guard let newRenderer = Renderer(device: device) else {
            print("Renderer cannot be initialized")
            return
        }
        self.renderer = newRenderer
        self.mtkView.delegate = self

        // Load scene models listed in Models.plist
        loadModels(device: device)

        let size = self.mtkView.drawableSize
        self.renderer.resize(width: Float(size.width), height: Float(size.height))

        // Lock cursor and enter game mode
        CGAssociateMouseAndMouseCursorPosition(0)
        NSCursor.hide()

        self.lastFrameTime = CACurrentMediaTime()
    }

    override func viewDidAppear() {
        super.viewDidAppear()
        self.view.window?.makeFirstResponder(self)
    }

    // MARK: - Tracking Area (required for mouseMoved events)

    private func setupTrackingArea() {
        if let existing = trackingArea {
            self.mtkView.removeTrackingArea(existing)
        }
        let options: NSTrackingArea.Options = [
            .mouseEnteredAndExited, .mouseMoved,
            .activeInKeyWindow, .inVisibleRect
        ]
        trackingArea = NSTrackingArea(rect: self.mtkView.bounds,
                                      options: options,
                                      owner: self,
                                      userInfo: nil)
        self.mtkView.addTrackingArea(trackingArea!)
    }

    override func viewDidLayout() {
        super.viewDidLayout()
        setupTrackingArea()
    }

    // MARK: - Keyboard Input

    override var acceptsFirstResponder: Bool { true }

    override func keyDown(with event: NSEvent) {
        // Option + C: reset camera
        if event.modifierFlags.contains(.option) && event.keyCode == KeyCode.c.rawValue {
            renderer.moveToCenter()
            return
        }

        // ESC: toggle mouse lock
        if event.keyCode == KeyCode.esc.rawValue {
            if isMouseLocked {
                CGAssociateMouseAndMouseCursorPosition(1)
                NSCursor.unhide()
                if let area = trackingArea { self.mtkView.removeTrackingArea(area) }
                isMouseLocked = false
            } else {
                CGAssociateMouseAndMouseCursorPosition(0)
                NSCursor.hide()
                setupTrackingArea()
                isMouseLocked = true
            }
        }
    }

    // MARK: - Mouse Look

    override func mouseMoved(with event: NSEvent) {
        let sensitivity: Float = 0.1
        renderer.rotateCamera(deltaX: Float(event.deltaX) * sensitivity,
                              deltaY: Float(event.deltaY) * sensitivity)
    }
}

// MARK: - MTKViewDelegate

extension GameViewController: MTKViewDelegate {

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        renderer.resize(width: Float(size.width), height: Float(size.height))
    }

    func draw(in view: MTKView) {
        let now = CACurrentMediaTime()
        let deltaTime = Float(now - lastFrameTime)
        lastFrameTime = now

        // CGEventSource.keyState polls raw hardware state — lowest overhead, no event queue
        let w     = CGEventSource.keyState(.combinedSessionState, key: KeyCode.w.rawValue)
        let a     = CGEventSource.keyState(.combinedSessionState, key: KeyCode.a.rawValue)
        let s     = CGEventSource.keyState(.combinedSessionState, key: KeyCode.s.rawValue)
        let d     = CGEventSource.keyState(.combinedSessionState, key: KeyCode.d.rawValue)
        let space = CGEventSource.keyState(.combinedSessionState, key: KeyCode.space.rawValue)
        let lCtrl = CGEventSource.keyState(.combinedSessionState, key: KeyCode.lCtrl.rawValue)

        renderer.move(w: w, a: a, s: s, d: d, space: space, lCtrl: lCtrl, deltaTime: deltaTime)

        guard let drawable           = view.currentDrawable,
              let renderPassDescriptor = view.currentRenderPassDescriptor else { return }

        renderer.draw(drawable: drawable, renderPassDescriptor: renderPassDescriptor)
    }
}
