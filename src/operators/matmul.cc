#include "operators/matmul.h"
#include "core/common.h"
#include "utils/operator_utils.h"
#include <array>
#include <iostream>
#include <ostream>
#include <tuple>
#include <vector>

namespace infini {

MatmulObj::MatmulObj(GraphObj* graph, Tensor A, Tensor B, Tensor C, bool transA,
    bool transB)
    : OperatorObj(OpType::MatMul, TensorVec { A, B }, { C })
    , transA(transA)
    , transB(transB)
{
    IT_ASSERT(checkValid(graph));
}

string MatmulObj::toString() const
{
    std::ostringstream os;
    os << "Matmul([" << (transA ? "A^T" : "A") << "," << (transB ? "B^T" : "B]")
       << ",A=" << inputs[0]->getGuid()
       << ",B=" << inputs[1]->getGuid() << ",C=" << outputs[0]->getGuid()
       << ",mnk=[" << m << "," << n << "," << k << "])";
    return os.str();
}

optional<vector<Shape>> MatmulObj::inferShape(const TensorVec& inputs)
{
    // =================================== 作业 ===================================
    // TODO：返回经过 matmul 操作后的 shape
    // REF: https://github.com/onnx/onnx/blob/main/docs/Operators.md#gemm
    // =================================== 作业 ===================================
    // m*k X k*n
    auto A = inputs[0], B = inputs[1];
    auto shapeA = A->getDims();
    auto shapeB = B->getDims();
    int rankA = A->getRank(); // Rank is the Shape of TensorDims
    int rankB = B->getRank();

    // 获取除了最后两个元素的shape，并对他们进行双向广播的推导
    Shape shapeA1(shapeA.begin(), shapeA.begin() + (rankA - 2));
    Shape shapeB1(shapeB.begin(), shapeB.begin() + (rankB - 2));
    Shape ret = infer_broadcast(shapeA1, shapeB1);

    // 根据是否转置获取两个要进行矩阵乘法的向量的k，同时这两个k要相等才可进行计算
    auto kA = *(transA ? shapeA.rbegin() + 1 : shapeA.rbegin());
    auto kB = *(transB ? shapeB.rbegin() : shapeB.rbegin() + 1);
    IT_ASSERT(kA == kB);

    // 根据是否转置获取m，n
    m = *(transA ? shapeA.rbegin() : shapeA.rbegin() + 1);
    n = *(transB ? shapeB.rbegin() + 1 : shapeB.rbegin());
    k = kA;
    // 把最终的矩阵乘法运算结果的形状放入结果中
    ret.emplace_back(m);
    ret.emplace_back(n);
    return { { ret } };
}
} // namespace infini