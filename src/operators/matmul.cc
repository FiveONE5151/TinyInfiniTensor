#include "operators/matmul.h"
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
    auto shapeA = inputs[0]->getDims();
    auto shapeB = inputs[1]->getDims();
    std::cout << "shapeA: " << vecToString(shapeA) << std::endl;
    std::cout << "shapeB: " << vecToString(shapeB) << std::endl;

    auto rankA = inputs[0]->getRank();
    auto rankB = inputs[1]->getRank();
    std::cout << "rankA: " << rankA << std::endl;
    std::cout << "rankB: " << rankB << std::endl;
    // 1. 2d-2d
    if (rankA == 2 && rankA == rankB) {
        if (shapeA[1] != shapeB[0]) {
            return std::nullopt;
        }
        return vector<Shape> { Shape { shapeA[0], shapeB[1] } };
    } else if (rankA == 1 && rankB == 1) {
        return {};
    }
    // 2. 1D-2D
    else if (rankA == 1 && rankB == 2) {
        if (shapeA[0] != shapeB[rankB - 2]) {
            return std::nullopt;
        }
        return vector<Shape> { { 1, shapeB[1] } };
    }
    // 3. 2D-1D
    else if (rankA == 2 && rankB == 1) {
        if (shapeA[1] != shapeB[0]) {
            return std::nullopt;
        }
        return vector<Shape> { { shapeA[0], 1 } };
    }
    // 4. 1D-ND
    else if (rankA == 1 && rankB > 2) {
        std::cout << "Enter rankA==2 and rankB>2\n";
        if (shapeA[0] != shapeB[rankB - 2]) {
            std::cerr << "last two indices cannot be multiplied\n";
            return std::nullopt;
        }
        Shape result = shapeB;
        result.erase(result.end() - 2);
        std::cout << "result shape: " << vecToString(result) << std::endl;
        return vector<Shape> { result };
    }
    // 5. ND-1D
    else if (rankA > 2 && rankB == 1) {
        if (shapeB[0] != shapeA[rankA - 1]) {
            return std::nullopt;
        }
        Shape result = shapeA;
        result.pop_back();
        return vector<Shape> { result };
    }
    // 6. ND-ND
    else if (rankA > 2 && rankB > 2) {
        std::cout << "Enter matmul ND-ND branch:\n";
        if (transA) {
            std::swap(shapeA[rankA - 2], shapeA[rankA - 1]);
        }
        if (transB) {
            std::swap(shapeB[rankB - 2], shapeB[rankB - 1]);
        }
        if (shapeA[rankA - 1] != shapeB[rankB - 2]) {
            std::cerr << "ND-ND: last two indexes can not be multiplied\n";
            return std::nullopt;
        }
        if (rankA > rankB) {
            std::cout << "ND-ND: rankA>rankB\n";
            Shape result = shapeA;
            auto offset = rankA - rankB;
            for (auto i = offset, j = 0UL; i < rankA; ++i, ++j) {
                if (shapeB[j] != shapeA[i] && shapeB[j] != 1) {
                    return std::nullopt;
                }
            }
            result[rankA - 1] = shapeB[rankB - 1];
            return vector<Shape> { result };
        } else if (rankA < rankB) {
            std::cout << "ND-ND: rankA<=rankB\n";
            Shape result = shapeB;
            auto offset = rankB - rankA;
            for (auto i = offset, j = 0UL; i < rankA; ++i, ++j) {
                if (shapeB[i] != shapeA[j] && shapeA[j] != 1) {
                    return std::nullopt;
                }
            }
            result[rankB - 1] = shapeA[rankA - 1];
            return vector<Shape> { result };
        } else if (rankA == rankB) {
            std::cout << "ND-ND: rankA=rankB\n";
            if (shapeA[rankA - 1] != shapeB[rankB - 2]) {
                std::cerr << "last two indexes can not be multiplied\n";
                return std::nullopt;
            }
            Shape result = infer_broadcast(shapeA, shapeB);
            result[rankA - 2] = shapeA[rankA - 2];
            result[rankB - 1] = shapeB[rankB - 1];
            std::cout << "result shape: " << vecToString(result) << std::endl;
            return vector<Shape> { result };
        }
    }
    return std::nullopt;
}
} // namespace infini