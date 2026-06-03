//
//  MLXDenoiser.swift
//  SimplePathTracer
//
//  MLX-Swift based edge-preserving reconstruction filter (A-Trous Wavelet Denoising)
//

import Foundation
import MLX
import Metal

public final class MLXDenoiser {
    
    public init() {}
    
    /// Performs G-buffer guided A-trous wavelet denoising on the CPU/GPU shared memory buffer.
    ///
    /// - Parameters:
    ///   - width: Width of the render target.
    ///   - height: Height of the render target.
    ///   - colorPtr: CPU-accessible raw pointer to path-traced color (RGBA, 16 bytes/pixel).
    ///   - normalPtr: CPU-accessible raw pointer to world-space normals (RGBA, 16 bytes/pixel).
    ///   - depthPtr: CPU-accessible raw pointer to depth map (RGBA, 16 bytes/pixel).
    ///   - albedoPtr: CPU-accessible raw pointer to base material color (RGBA, 16 bytes/pixel).
    ///   - outPtr: CPU-accessible raw pointer to write the denoised result (RGBA, 16 bytes/pixel).
    public func denoise(width: Int, height: Int,
                        colorPtr: UnsafeRawPointer,
                        normalPtr: UnsafeRawPointer,
                        depthPtr: UnsafeRawPointer,
                        albedoPtr: UnsafeRawPointer,
                        outPtr: UnsafeMutableRawPointer) {
        
        let pixelCount = width * height
        
        // Wrap pointers in UnsafeBufferPointer
        let colorBuf = UnsafeBufferPointer<Float>(start: colorPtr.assumingMemoryBound(to: Float.self), count: pixelCount * 4)
        let normalBuf = UnsafeBufferPointer<Float>(start: normalPtr.assumingMemoryBound(to: Float.self), count: pixelCount * 4)
        let depthBuf = UnsafeBufferPointer<Float>(start: depthPtr.assumingMemoryBound(to: Float.self), count: pixelCount * 4)
        let albedoBuf = UnsafeBufferPointer<Float>(start: albedoPtr.assumingMemoryBound(to: Float.self), count: pixelCount * 4)
        
        // Wrap in MLXArrays of shape [H, W, C]
        let colorArray = MLXArray(colorBuf, [height, width, 4])
        let normalArray = MLXArray(normalBuf, [height, width, 4])
        let depthArray = MLXArray(depthBuf, [height, width, 4])
        let albedoArray = MLXArray(albedoBuf, [height, width, 4])
        
        // Split channels to isolate color, albedo, normal, and depth components
        let splitColor = split(colorArray, parts: 4, axis: 2)
        let splitAlbedo = split(albedoArray, parts: 4, axis: 2)
        
        // Demodulate albedo: light = color / (albedo + epsilon)
        let eps: Float = 0.008
        let r = splitColor[0] / (splitAlbedo[0] + eps)
        let g = splitColor[1] / (splitAlbedo[1] + eps)
        let b = splitColor[2] / (splitAlbedo[2] + eps)
        let light = concatenated([r, g, b], axis: 2)
        
        let splitNormal = split(normalArray, parts: 4, axis: 2)
        let splitDepth = split(depthArray, parts: 4, axis: 2)
        let depthPArray = splitDepth[0] // Depth is stored in red channel (and others)
        
        // 5x5 B-Spline kernel weights
        let kernelWeights: [[Float]] = [
            [1.0/256.0, 4.0/256.0,  6.0/256.0,  4.0/256.0,  1.0/256.0],
            [4.0/256.0, 16.0/256.0, 24.0/256.0, 16.0/256.0, 4.0/256.0],
            [6.0/256.0, 24.0/256.0, 36.0/256.0, 24.0/256.0, 6.0/256.0],
            [4.0/256.0, 16.0/256.0, 24.0/256.0, 16.0/256.0, 4.0/256.0],
            [1.0/256.0, 4.0/256.0,  6.0/256.0,  4.0/256.0,  1.0/256.0]
        ]
        
        let sigma_n: Float = 128.0
        let sigma_d: Float = 1.0
        
        var currentLight = light
        
        // Multi-pass Edge-Avoiding A-Trous Wavelet filter (steps: 1, 2, 4, 8, 16)
        let strides = [1, 2, 4, 8, 16]
        for step in strides {
            var sumWeightedLight = MLXArray.zeros([height, width, 3], type: Float.self)
            var sumWeights = MLXArray.zeros([height, width, 1], type: Float.self)
            
            for dy in -2...2 {
                for dx in -2...2 {
                    let w_s = kernelWeights[dy + 2][dx + 2]
                    
                    // Shift using roll along Y and X axes
                    let light_q = roll(roll(currentLight, shift: dy * step, axis: 0), shift: dx * step, axis: 1)
                    
                    let n_q0 = roll(roll(splitNormal[0], shift: dy * step, axis: 0), shift: dx * step, axis: 1)
                    let n_q1 = roll(roll(splitNormal[1], shift: dy * step, axis: 0), shift: dx * step, axis: 1)
                    let n_q2 = roll(roll(splitNormal[2], shift: dy * step, axis: 0), shift: dx * step, axis: 1)
                    
                    let depthQArray = roll(roll(depthPArray, shift: dy * step, axis: 0), shift: dx * step, axis: 1)
                    
                    // Normal weight: w_n = exp(-max(0.0, 1.0 - dot(n_p, n_q)) * sigma_n)
                    let term0 = splitNormal[0] * n_q0
                    let term1 = splitNormal[1] * n_q1
                    let term2 = splitNormal[2] * n_q2
                    let dot_val = term0 + term1 + term2
                    let one_minus_dot = 1.0 - dot_val
                    let max_val = MLX.maximum(0.0, one_minus_dot)
                    let sigma_n_arr = MLXArray(sigma_n)
                    let w_n_exponent = -max_val * sigma_n_arr
                    let w_n = exp(w_n_exponent)
                    
                    // Depth weight: w_d = exp(-abs(d_p - d_q) / (d_p * sigma_d + epsilon))
                    let d_diff = MLX.abs(depthPArray - depthQArray)
                    let depthDenom = depthPArray * MLXArray(sigma_d) + 1e-4
                    let d_diff_neg = -d_diff
                    let w_d_exponent = d_diff_neg / depthDenom
                    let w_d = exp(w_d_exponent)
                    
                    // Combined weight
                    let w_temp = w_s * w_n
                    let w = w_temp * w_d
                    
                    sumWeightedLight = sumWeightedLight + w * light_q
                    sumWeights = sumWeights + w
                }
            }
            
            currentLight = sumWeightedLight / (sumWeights + 1e-4)
        }
        
        // Remodulate: finalColor = filteredLight * albedo
        let albedo3Channel = concatenated([splitAlbedo[0], splitAlbedo[1], splitAlbedo[2]], axis: 2)
        let filteredRGB = currentLight * albedo3Channel
        
        // Preserve color alpha (alpha channel of color)
        let finalColor = concatenated([filteredRGB, splitColor[3]], axis: 2)
        
        // Force evaluation and copy the result back to outPtr
        finalColor.eval()
        let arrayData = finalColor.asData(access: MLXArray.AccessMethod.noCopyIfContiguous)
        arrayData.data.withUnsafeBytes { buffer in
            outPtr.copyMemory(from: buffer.baseAddress!, byteCount: buffer.count)
        }
    }
}
