//
//  Camera.cpp
//  LetsUseSwift
//
//  Created by Sora Sugiyama on 2/15/26.
//

#include "Camera.hpp"
#include <iostream>
Camera::Camera() :
position{0.0f, 1.0f, 5.0f},
unitDeltaFront{1.0f, 0.0f, 0.0f},
unitDeltaSide{0.0f, 0.0f, 1.0f},
unitDeltaUp{0.0f, 1.0f, 0.0f},
yaw{-90.f},
pitch{0.0f}
{
    update();
}

void Camera::moveToCenter()
{
    position = simd::float3{0.0f, 1.0f, 5.0f};
    yaw = -90.0f;
    pitch = 0.0f;
    
    update();
}

void Camera::moveFront(float delta)
{
    position += unitDeltaFront * delta;
}

void Camera::moveSide(float delta)
{
    position += unitDeltaSide * delta;
}

void Camera::moveUp(float delta)
{
    position += unitDeltaUp * delta;
}

void Camera::rotate(float mouseDeltaX, float mouseDeltaY)
{
    yaw += mouseDeltaX;
    pitch += mouseDeltaY;
    
    if (pitch > 89.0f)
    {
        pitch = 89.0f;
    }
    else if (pitch < -89.0f)
    {
        pitch = -89.0f;
    }
    
    update();
}

void Camera::update()
{
    const float cosPitch = cosf(pitch * Consts::PIf / 180.0f);
    simd::float3 newFront;
    newFront.x = cosf(yaw * Consts::PIf / 180.0f) * cosPitch;
    newFront.y = sinf(pitch * Consts::PIf / 180.0f);
    newFront.z = sinf(yaw * Consts::PIf / 180.0f) * cosPitch;
    
    unitDeltaFront = simd::normalize(newFront);
    unitDeltaSide = simd::normalize(simd::cross(unitDeltaFront, unitDeltaUp));
}

simd::float4x4 Camera::getViewMatrix() const
{    
    simd::float3 zAxis = -unitDeltaFront;
    simd::float3 xAxis = simd::normalize(simd::cross(unitDeltaUp, zAxis));
    simd::float3 yAxis = simd::cross(zAxis, xAxis);
    
    return simd::float4x4{
        simd::float4{xAxis.x, yAxis.x, zAxis.x, 0.0f},
        simd::float4{xAxis.y, yAxis.y, zAxis.y, 0.0f},
        simd::float4{xAxis.z, yAxis.z, zAxis.z, 0.0f},
        simd::float4{-simd::dot(xAxis, position), -simd::dot(yAxis, position), -simd::dot(zAxis, position), 1.0f}
    };
}
