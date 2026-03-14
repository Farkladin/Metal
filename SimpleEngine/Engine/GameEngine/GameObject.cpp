//
//  GameObject.cpp
//  LetsUseSwift
//
//  Created by Sora Sugiyama on 3/5/26.
//

#include "GameObject.hpp"
#include <Metal/Metal.hpp>

ObjectList::ObjectList():
size{0U}
{}

ObjectList::~ObjectList() = default;

uint8_t ObjectList::add_back(void* pVertexBuffer, void *pIndexBuffer, std::uint32_t n, bool isUInt32, float y0)
{
    if (size >= MAX_SIZE) {
        std::cerr << "Out of Bounds" << std::endl;
        return 0U;
    }
    
    GameObject &back = arr[size++];
    
    back.vertexBuffer = NS::RetainPtr(static_cast<MTL::Buffer*>(pVertexBuffer));
    back.indexBuffer = NS::RetainPtr(static_cast<MTL::Buffer*>(pIndexBuffer));
    
    back.indexCount = n;
    back.isUInt32Index = isUInt32;
    back.y0 = y0;
    return 1U;
}

void ObjectList::pop(const std::size_t index)
{
    if (index >= size || index <0) {
        return;
    }
    
    --size;
    GameObject &cur = arr[index], &back = arr[size];
    
    if(size != index) {
        cur = std::move(back);
    }
    
    back = GameObject();
}

const GameObject& ObjectList::operator[](const std::size_t index) const
{
    return arr[index];
}
