#include "utils/operator_utils.h"
#include "core/common.h"
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
    if (A.empty() && B.empty()) {
        return {};
    }
    auto A_ = A;
    auto B_ = B;
    int rankA = A.size();
    int rankB = B.size();

    // 取更大的维度作为结果维度
    int rank = std::max(rankA, rankB);

    //如果是A更小，那么在A前面补全1
    if (rankA < rank) {
        int offset = rank - rankA;
        for (int i = 0; i < offset; ++i) {
            A_.insert(A_.begin(), 1);
        }
    }
    //同理，B更小
    if (rankB < rank) {
        int offset = rank - rankB;
        for (int i = 0; i < offset; ++i) {
            B_.insert(B_.begin(), 1);
        }
    }

    Shape ret;
    for (int i = 0; i < rank; ++i) {
        IT_ASSERT(A_[i] == B_[i] || A_[i] == 1 || B_[i] == 1);
        ret.push_back(std::max(A_[i], B_[i]));
    }
    return ret;
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
