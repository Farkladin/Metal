//
//  ContentView.swift
//  SimplePathTracer
//

import SwiftUI
import MetalKit
import Combine
import UniformTypeIdentifiers

struct ContentView: View {
    @State private var renderer: Renderer = {
        guard let device = MTLCreateSystemDefaultDevice(),
              let r = Renderer(device: device) else {
            fatalError("Metal is not supported on this device.")
        }
        return r
    }()
    
    @State private var currentSPP: UInt32 = 0
    @State private var status: String = "Setup Mode"
    @State private var selectedPreset = "Front"
    @State private var renderModeState: RenderMode = .setup
    @State private var isPausedState = false
    @State private var isDenoisingActiveState = false
    
    let timer = Timer.publish(every: 0.1, on: .main, in: .common).autoconnect()
    
    var body: some View {
        HStack(spacing: 0) {
            // ─── Metal rendering viewport ───
            MetalView(renderer: renderer)
                .frame(minWidth: 600, minHeight: 500)
            
            // ─── Beautiful Glassmorphic Sidebar ───
            VStack(alignment: .leading, spacing: 20) {
                // Header
                Text("Simple Path Tracer")
                    .font(.system(size: 20, weight: .bold, design: .rounded))
                    .foregroundColor(.white)
                    .padding(.bottom, 5)
                
                // Status Section
                VStack(alignment: .leading, spacing: 5) {
                    Text("STATUS")
                        .font(.caption)
                        .fontWeight(.semibold)
                        .foregroundColor(.gray)
                    
                    HStack {
                        Circle()
                            .fill(statusColor)
                            .frame(width: 8, height: 8)
                        Text(status)
                            .font(.system(.body, design: .monospaced))
                            .foregroundColor(.white)
                    }
                    .padding(.vertical, 8)
                    .padding(.horizontal, 12)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(Color.white.opacity(0.05))
                    .cornerRadius(8)
                }
                
                // Stats Section
                VStack(alignment: .leading, spacing: 5) {
                    Text("SAMPLES PER PIXEL (SPP)")
                        .font(.caption)
                        .fontWeight(.semibold)
                        .foregroundColor(.gray)
                    
                    Text("\(currentSPP)")
                        .font(.system(size: 28, weight: .bold, design: .monospaced))
                        .foregroundColor(.green)
                        .padding(.vertical, 8)
                        .padding(.horizontal, 12)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(Color.white.opacity(0.05))
                        .cornerRadius(8)
                }
                
                Divider()
                    .background(Color.white.opacity(0.1))
                
                // Ray Tracing Controls
                VStack(alignment: .leading, spacing: 10) {
                    Text("CONTROLS")
                        .font(.caption)
                        .fontWeight(.semibold)
                        .foregroundColor(.gray)
                    
                    if renderModeState == .setup {
                        // "Start Path Tracing" Button
                        Button(action: startPathTracing) {
                            HStack {
                                Image(systemName: "play.fill")
                                Text("Start Path Tracing")
                                    .fontWeight(.bold)
                            }
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 10)
                            .foregroundColor(.white)
                            .background(Color.blue)
                            .cornerRadius(8)
                        }
                        .buttonStyle(.plain)
                    } else {
                        // "Pause/Resume" & "Stop" Row
                        HStack(spacing: 10) {
                            Button(action: togglePause) {
                                HStack {
                                    Image(systemName: isPausedState ? "play.fill" : "pause.fill")
                                    Text(isPausedState ? "Resume" : "Pause")
                                        .fontWeight(.bold)
                                }
                                .frame(maxWidth: .infinity)
                                .padding(.vertical, 10)
                                .foregroundColor(isDenoisingActiveState ? .gray : .white)
                                .background(isDenoisingActiveState ? Color.white.opacity(0.05) : (isPausedState ? Color.green : Color.orange))
                                .cornerRadius(8)
                            }
                            .buttonStyle(.plain)
                            .disabled(isDenoisingActiveState)
                            
                            Button(action: stopPathTracing) {
                                HStack {
                                    Image(systemName: "gobackward")
                                    Text("Reset")
                                        .fontWeight(.bold)
                                }
                                .frame(maxWidth: .infinity)
                                .padding(.vertical, 10)
                                .foregroundColor(.white)
                                .background(Color.red)
                                .cornerRadius(8)
                            }
                            .buttonStyle(.plain)
                        }
                    }
                    
                    Button(action: {
                        isDenoisingActiveState.toggle()
                        renderer.isDenoisingActive = isDenoisingActiveState
                        if isDenoisingActiveState && !renderer.isPaused {
                            renderer.isPaused = true
                            isPausedState = true
                        }
                    }) {
                        HStack {
                            Image(systemName: "sparkles")
                            Text(isDenoisingActiveState ? "Denoising ON" : "Denoise Image")
                                .fontWeight(.bold)
                        }
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 10)
                        .foregroundColor(renderModeState == .setup ? .gray : .white)
                        .background(renderModeState == .setup ? Color.white.opacity(0.02) : (isDenoisingActiveState ? Color.green : Color.white.opacity(0.1)))
                        .cornerRadius(8)
                    }
                    .buttonStyle(.plain)
                    .disabled(renderModeState == .setup)
                    .padding(.top, 5)
                    
                    // Export Image Button
                    Button(action: exportImage) {
                        HStack {
                            Image(systemName: "square.and.arrow.down")
                            Text("Export Image")
                                .fontWeight(.bold)
                        }
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 10)
                        .foregroundColor(.white)
                        .background(Color.purple)
                        .cornerRadius(8)
                    }
                    .buttonStyle(.plain)
                    .padding(.top, 5)
                }
                
                Divider()
                    .background(Color.white.opacity(0.1))
                
                // Camera Presets
                VStack(alignment: .leading, spacing: 10) {
                    Text("CAMERA PRESETS")
                        .font(.caption)
                        .fontWeight(.semibold)
                        .foregroundColor(.gray)
                    
                    Text("Camera position is locked during tracing. Change presets in Setup mode.")
                        .font(.caption2)
                        .foregroundColor(.gray.opacity(0.8))
                        .padding(.bottom, 2)
                    
                    let presets = ["Front", "Top", "Left", "Right", "Close-up"]
                    
                    Grid(horizontalSpacing: 8, verticalSpacing: 8) {
                        GridRow {
                            presetButton(name: "Front")
                            presetButton(name: "Top")
                        }
                        GridRow {
                            presetButton(name: "Left")
                            presetButton(name: "Right")
                        }
                        GridRow {
                            presetButton(name: "Close-up")
                        }
                    }
                }
                .disabled(renderModeState == .tracing)
                .opacity(renderModeState == .tracing ? 0.5 : 1.0)
                
                Divider()
                    .background(Color.white.opacity(0.1))
                
                // Help / Keyboard shortcuts
                VStack(alignment: .leading, spacing: 8) {
                    Text("NAVIGATION HELP")
                        .font(.caption)
                        .fontWeight(.semibold)
                        .foregroundColor(.gray)
                    
                    Group {
                        Text("• **Click window** to capture mouse cursor")
                        Text("• **WASD** to move Camera front/side")
                        Text("• **Space / Ctrl** to move Up / Down")
                        Text("• **Mouse movement** to look around")
                        Text("• **ESC** to release mouse focus")
                        Text("• **Option + C** to reset camera")
                    }
                    .font(.system(size: 11))
                    .foregroundColor(.gray)
                }
                
                Spacer()
            }
            .padding()
            .frame(width: 280)
            .background(Color(red: 0.1, green: 0.1, blue: 0.12))
        }
        .frame(minWidth: 880, minHeight: 500)
        .preferredColorScheme(.dark)
        .onReceive(timer) { _ in
            self.currentSPP = renderer.currentFrameIndex
            self.renderModeState = renderer.renderMode
            self.isPausedState = renderer.isPaused
            self.isDenoisingActiveState = renderer.isDenoisingActive
            
            if renderer.renderMode == .setup {
                self.status = "Setup Mode (Free Camera)"
            } else {
                self.status = renderer.isPaused ? "Paused (Lock)" : "Tracing (Lock)"
            }
        }
    }
    
    private var statusColor: Color {
        if renderModeState == .setup {
            return .blue
        }
        return isPausedState ? .orange : .green
    }
    
    @ViewBuilder
    private func presetButton(name: String) -> some View {
        Button(action: {
            selectedPreset = name
            applyPreset(name: name)
        }) {
            Text(name)
                .font(.system(size: 12, weight: .medium))
                .frame(maxWidth: .infinity)
                .padding(.vertical, 8)
                .foregroundColor(selectedPreset == name ? .white : .gray)
                .background(selectedPreset == name ? Color.blue.opacity(0.4) : Color.white.opacity(0.05))
                .cornerRadius(6)
                .overlay(
                    RoundedRectangle(cornerRadius: 6)
                        .stroke(selectedPreset == name ? Color.blue : Color.clear, lineWidth: 1)
                )
        }
        .buttonStyle(.plain)
    }
    
    private func startPathTracing() {
        renderer.isPaused = false
        renderer.renderMode = .tracing
    }
    
    private func togglePause() {
        renderer.isPaused.toggle()
    }
    
    private func stopPathTracing() {
        renderer.renderMode = .setup
        renderer.isPaused = false
        renderer.isDenoisingActive = false
        isPausedState = false
        isDenoisingActiveState = false
        selectedPreset = "Front"
        applyPreset(name: "Front")
    }
    
    private func exportImage() {
        let savePanel = NSSavePanel()
        savePanel.allowedContentTypes = [.png]
        savePanel.canCreateDirectories = true
        savePanel.nameFieldStringValue = "render.png"
        
        savePanel.begin { response in
            if response == .OK, let url = savePanel.url {
                renderer.exportImage(to: url)
            }
        }
    }
    
    private func applyPreset(name: String) {
        switch name {
        case "Front":
            renderer.camera.setPreset(position: SIMD3(0, 1.5, 6), yaw: -90, pitch: 0)
        case "Top":
            renderer.camera.setPreset(position: SIMD3(0, 10, 0.1), yaw: -90, pitch: -89)
        case "Left":
            renderer.camera.setPreset(position: SIMD3(-8, 1.5, 0), yaw: 0, pitch: 0)
        case "Right":
            renderer.camera.setPreset(position: SIMD3(8, 1.5, 0), yaw: 180, pitch: 0)
        case "Close-up":
            renderer.camera.setPreset(position: SIMD3(0, 0.5, 3), yaw: -90, pitch: 10)
        default:
            renderer.camera.moveToCenter()
        }
    }
}

// ─── MetalView NSViewRepresentable Wrapper ───
struct MetalView: NSViewRepresentable {
    var renderer: Renderer
    
    func makeNSView(context: Context) -> InteractiveMTKView {
        let mtkView = InteractiveMTKView()
        mtkView.device = renderer.device
        mtkView.clearColor = MTLClearColor(red: 0.0, green: 0.0, blue: 0.0, alpha: 1.0)
        mtkView.framebufferOnly = false
        mtkView.depthStencilPixelFormat = .depth32Float
        mtkView.clearDepth = 1.0
        mtkView.delegate = context.coordinator
        mtkView.preferredFramesPerSecond = 60
        mtkView.renderer = renderer
        
        // Initial setup
        renderer.loadScene()
        renderer.buildAccelerationStructures()
        
        return mtkView
    }
    
    func updateNSView(_ nsView: InteractiveMTKView, context: Context) {
        // Handled via state triggers
    }
    
    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }
    
    class Coordinator: NSObject, MTKViewDelegate {
        var parent: MetalView
        var lastFrameTime: CFTimeInterval = 0
        
        init(_ parent: MetalView) {
            self.parent = parent
            self.lastFrameTime = CACurrentMediaTime()
        }
        
        func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
            parent.renderer.resize(width: Float(size.width), height: Float(size.height))
        }
        
        func draw(in view: MTKView) {
            let now = CACurrentMediaTime()
            let deltaTime = Float(max(0.0001, now - lastFrameTime))
            lastFrameTime = now
            
            // Poll keyboard state only when in Setup Mode
            if parent.renderer.renderMode == .setup {
                let w     = CGEventSource.keyState(.combinedSessionState, key: 13) // KeyCode.w
                let a     = CGEventSource.keyState(.combinedSessionState, key: 0)  // KeyCode.a
                let s     = CGEventSource.keyState(.combinedSessionState, key: 1)  // KeyCode.s
                let d     = CGEventSource.keyState(.combinedSessionState, key: 2)  // KeyCode.d
                let space = CGEventSource.keyState(.combinedSessionState, key: 49) // KeyCode.space
                let lCtrl = CGEventSource.keyState(.combinedSessionState, key: 59) // KeyCode.lCtrl
                
                parent.renderer.move(w: w, a: a, s: s, d: d, space: space, lCtrl: lCtrl, deltaTime: deltaTime)
            }
            
            parent.renderer.draw(in: view)
        }
    }
}

// ─── Custom Interactive MTKView Subclass ───
class InteractiveMTKView: MTKView {
    weak var renderer: Renderer?
    var isMouseLocked: Bool = false
    private var trackingArea: NSTrackingArea?
    
    override var acceptsFirstResponder: Bool { true }
    
    override func viewDidMoveToWindow() {
        super.viewDidMoveToWindow()
        setupTrackingArea()
    }
    
    override func setFrameSize(_ newSize: NSSize) {
        super.setFrameSize(newSize)
        setupTrackingArea()
    }
    
    private func setupTrackingArea() {
        if let existing = trackingArea {
            removeTrackingArea(existing)
        }
        let options: NSTrackingArea.Options = [
            .mouseEnteredAndExited, .mouseMoved,
            .activeInKeyWindow, .inVisibleRect
        ]
        trackingArea = NSTrackingArea(rect: bounds, options: options, owner: self, userInfo: nil)
        addTrackingArea(trackingArea!)
    }
    
    func lockMouse() {
        guard !isMouseLocked else { return }
        CGAssociateMouseAndMouseCursorPosition(0)
        NSCursor.hide()
        isMouseLocked = true
        window?.makeFirstResponder(self)
    }
    
    func unlockMouse() {
        guard isMouseLocked else { return }
        CGAssociateMouseAndMouseCursorPosition(1)
        NSCursor.unhide()
        isMouseLocked = false
    }
    
    override func mouseDown(with event: NSEvent) {
        // Clicking view locks mouse only if in Setup Mode
        if renderer?.renderMode == .setup {
            lockMouse()
        }
    }
    
    override func keyDown(with event: NSEvent) {
        // Option + C resets camera
        if event.modifierFlags.contains(.option) && event.keyCode == 8 { // KeyCode.c = 8
            renderer?.camera.moveToCenter()
            return
        }
        
        // ESC releases mouse
        if event.keyCode == 53 { // KeyCode.esc = 53
            unlockMouse()
        }
    }
    
    override func mouseMoved(with event: NSEvent) {
        guard isMouseLocked, renderer?.renderMode == .setup else { return }
        let sensitivity: Float = 0.1
        renderer?.rotateCamera(deltaX: Float(event.deltaX) * sensitivity,
                               deltaY: Float(-event.deltaY) * sensitivity)
    }
}
