//
//  Camera.hpp
//  LetsUseSwift
//
//  Created by Sora Sugiyama on 2/15/26.
//

#ifndef Camera_hpp
#define Camera_hpp

#include <cmath>
#include <simd/simd.h>
#include "Consts.hpp"

class Camera
{
    simd::float3 position, unitDeltaFront, unitDeltaSide, unitDeltaUp;
    constexpr static simd::float3 worldUp{0.57735f, 0.57735f, 0.57735f};
    
    float yaw, pitch;
    
    void update();
    
public:
    Camera();
    
    void moveFront(float delta);
    void moveSide(float delta);
    void moveUp(float delta);
    void rotate(float mouseDeltaX, float mouseDeltaY);
    void moveToCenter();
    
    simd::float4x4 getViewMatrix() const;
};

#endif /* Camera_hpp */
