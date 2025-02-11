#include "core/allocator.h"
#include "core/common.h"
#include <cstddef>
#include <iostream>
#include <ostream>
#include <utility>

namespace infini {
Allocator::Allocator(Runtime runtime)
    : runtime(runtime)
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
    if (this->ptr != nullptr) {
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
            // 有空块可以容纳新的size
            auto offset = block.first;
            auto space = block.second - size;

            // 更新空块链表的内容
            freeBlocks.erase(offset);

            if (space > 0) {
                // 原空块内还有剩余的空间，则在空块链表中添加这个剩余的空间
                freeBlocks[offset + size] = space;
            }

            return offset;
        } else if (block.first == (peak - block.second)) {
            // 内存池的末尾有空块, 但是不足以容纳size，则进行扩容
            auto offset = block.first;

            // 记录新的peak
            peak += size - block.second;
            freeBlocks.erase(offset);
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
    used -= size;

    // 合并相邻的空闲块
    for (auto& block : freeBlocks) {
        if (block.first + block.second == addr) {
            // 释放的空间上方有空块，则和上方合并
            block.second += size;
            return;
        }
        if (addr + size == block.first) {
            // 释放的空间下方有空块
            // 需注意合并下方需要在freeBlocks链表中添加释放的空间地址，并删除被合并的下方的空块
            freeBlocks[addr] += block.second;
            freeBlocks.erase(block.first);
            return;
        }
    }

    // 无相邻的块，则直接添加
    freeBlocks[addr] = size;
    return;
}

void* Allocator::getPtr()
{
    if (this->ptr == nullptr) {
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
