#include "core/allocator.h"
#include "core/common.h"
#include <cstddef>
#include <utility>

namespace infini
{
    Allocator::Allocator(Runtime runtime) : runtime(runtime)
    {
        used = 0;
        peak = 0;
        ptr = nullptr;

        // 'alignment' defaults to sizeof(uint64_t), because it is the length of
        // the longest data type currently supported by the DataType field of
        // the tensor
        alignment = sizeof(uint64_t);
    }

    Allocator::~Allocator()
    {
        if (this->ptr != nullptr)
        {
            runtime->dealloc(this->ptr);
        }
    }

    size_t Allocator::alloc(size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        // pad the size to the multiple of alignment
        size = this->getAlignedSize(size);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来分配内存，返回起始地址偏移量
        // =================================== 作业 ===================================
        // 更新used大小
        used += size;
        // 有freeblock
        for (auto& block : freeBlocks) {
            if (block.second >= size) {
                auto offset = block.first;
                auto space = block.second - size;
                freeBlocks.erase(offset);

                if (space > 0) {
                    freeBlocks[offset + size] = space;
                }

                return offset;
            }
        }

        // 无freeblock，直接分配
        peak += size;
        return peak - size;
    }

    void Allocator::free(size_t addr, size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        size = getAlignedSize(size);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来回收内存
        // =================================== 作业 ===================================
        used-=size;

        // 释放最后一个内存块
        if(addr == peak-size){
            peak-=size;
            return;
        }

        // 否则合并相邻的空闲块
        for (auto& block : freeBlocks) {
            if (block.first+block.second==addr) {
                block.second+=size;
                return;
            }
            if (addr+size==block.first) {
                freeBlocks[addr]+=block.second;
                freeBlocks.erase(block.first);
                return;
            }
        }

        // 无相邻的块，则直接添加
        freeBlocks[addr]=size;
        return;
    }

    void *Allocator::getPtr()
    {
        if (this->ptr == nullptr)
        {
            this->ptr = runtime->alloc(this->peak);
            printf("Allocator really alloc: %p %lu bytes\n", this->ptr, peak);
        }
        return this->ptr;
    }

    size_t Allocator::getAlignedSize(size_t size)
    {
        return ((size - 1) / this->alignment + 1) * this->alignment;
    }

    void Allocator::info()
    {
        std::cout << "Used memory: " << this->used
                  << ", peak memory: " << this->peak << std::endl;
    }
}
