#include "utils/operator_utils.h"
#include "core/runtime.h"
#include <algorithm>
#include <cstddef>
#include <fstream>

namespace infini {

Shape infer_broadcast(const Shape& A, const Shape& B)
{

    // =================================== 作业 ===================================
    // TODO：对 A 和 B 进行双向广播，返回广播后的形状。
    // REF: https://github.com/onnx/onnx/blob/main/docs/Broadcasting.md
    // =================================== 作业 ===================================
    Shape result = {};
    // 1. shape完全相同，直接返回
    if (A == B) {
        return A;
    } else if (A != B && A.size() == B.size()) { // 2. 形状不同但是维度相同，按位取更大的
        for (int i = 0; i < static_cast<int>(A.size()); ++i) {
            if (A[i] < B[i]) {
                result.push_back(B[i]);
            } else {
                result.push_back(A[i]);
            }
        }
    } else { //3. 维度不相同
        int pad = A.size() - B.size();
        // A的维度更小，那么在A的shape前填充1，再按位取大的
        if (pad < 0) {
            for (int i = 0; i < -pad; ++i) {
                result.push_back(B[i]);
            }
            for (int i = -pad, j = 0; i < static_cast<int>(B.size()); ++i, ++j) {
                if (A[j] < B[i]) {
                    result.push_back(B[i]);
                } else {
                    result.push_back(A[j]);
                }
            }
        } else { // B的维度更小，同理
            for (int i = 0; i < pad; ++i) {
                result.push_back(A[i]);
            }
            for (int i = pad, j = 0; i < static_cast<int>(A.size()); ++i, ++j) {
                if (A[i] < B[j]) {
                    result.push_back(B[j]);
                } else {
                    result.push_back(A[i]);
                }
            }
        }
    }

    return result;
}

int get_real_axis(const int& axis, const int& rank)
{
    IT_ASSERT(rank >= 1);
    IT_ASSERT(axis >= -rank && axis <= (rank - 1));
    int newAxis;
    if (axis < 0) {
        newAxis = rank + axis;
    } else {
        newAxis = axis;
    }
    return newAxis;
}

Shape locate_index(size_t inputN, const Shape& shape)
{
    Shape ans(shape.size());
    auto i = ans.rbegin();
    auto j = shape.rbegin(), ej = shape.rend();
    while (j != ej) {
        auto div = std::div(inputN, *j++);
        *i++ = div.rem;
        inputN = div.quot;
    }
    return ans;
}

size_t delocate_index(const Shape& shapeIndex, const Shape& shape,
    const Shape& stride)
{
    size_t ans = 0;
    Shape index(shapeIndex.size());
    IT_ASSERT(shapeIndex.size() == shape.size());
    IT_ASSERT(shape.size() == stride.size());
    for (size_t i = 0; i < shape.size(); ++i) {
        index[i] = shapeIndex[i] % shape[i];
        ans += index[i] * stride[i];
    }
    return ans;
}

std::string device_to_str(Device device)
{
    std::string deviceStr;
    switch (device) {
    case Device::CPU:
        return "CPU";
    default:
        IT_TODO_HALT();
    }
}

std::string get_kernel_attrs_str(const KernelAttrs& kernelAttrs)
{
    std::string deviceStr = device_to_str(std::get<0>(kernelAttrs));
    std::string opStr = OpType(std::get<1>(kernelAttrs)).toString();
    return deviceStr + ", " + opStr;
}

} // namespace infini
