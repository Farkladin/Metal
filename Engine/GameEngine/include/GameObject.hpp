//
//  GameObject.hpp
//  LetsUseSwift
//
//  Created by Sora Sugiyama on 3/5/26.
//

#ifndef GameObject_hpp
#define GameObject_hpp

#include <cstddef>
#include <cstdint>
#include <simd/simd.h>
#include <array>
#include <iostream>
#include <Foundation/Foundation.hpp>

namespace MTL
{
class Buffer;
}

struct GameObject
{
    NS::SharedPtr<MTL::Buffer> vertexBuffer = nullptr;
    NS::SharedPtr<MTL::Buffer> indexBuffer = nullptr;
    std::uint32_t indexCount = 0u;
    bool isUInt32Index = false;
    
    float y0 = 0.0f;
};

class ObjectList
{
    static constexpr size_t MAX_SIZE = 1024;
    
    std::array<GameObject, MAX_SIZE> arr;
    
public:
    std::size_t size;
    
    ObjectList();
    ~ObjectList();
    
    void pop(const std::size_t index);
    std::uint8_t add_back(void* pVertexBuffer, void* pIndexBuffer, std::uint32_t n, bool isUInt32, float y0);
    
    const GameObject& operator[](const std::size_t index) const;
};

#endif /* GameObject_hpp */
