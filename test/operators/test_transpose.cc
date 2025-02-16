#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "operators/transpose.h"

#include "test.h"
#include <iostream>

namespace infini {

TEST(Transpose, ShapeInference) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    {
        Graph g = make_ref<GraphObj>(runtime);
        Tensor i = g->addTensor({1, 2, 3, 4}, DataType::Float32);
        std::cout<<"Testing Transpose with permute {1.2.3.4}\n";
        auto op = g->addOp<TransposeObj>(i, nullptr, Shape{0, 1, 2, 3});
        std::cout<<"Transpose object: "<<op->toString();
        std::cout << "Expected shape: {1, 2, 3, 4}, Actual shape: ";
        for (auto dim : op->getOutput()->getDims()) {
            std::cout << dim << " ";
        }
        EXPECT_EQ(op->getOutput()->getDims(), (Shape{1, 2, 3, 4}));
    }
    {
        Graph g = make_ref<GraphObj>(runtime);
        Tensor i = g->addTensor({1, 2, 3, 4}, DataType::Float32);
        auto op = g->addOp<TransposeObj>(i, nullptr, Shape{0, 2, 1, 3});
        EXPECT_EQ(op->getOutput()->getDims(), (Shape{1, 3, 2, 4}));
    }
    {
        Graph g = make_ref<GraphObj>(runtime);
        Tensor i = g->addTensor({2, 3, 4}, DataType::Float32);
        auto op = g->addOp<TransposeObj>(i, nullptr, Shape{0, 2, 1});
        EXPECT_EQ(op->getOutput()->getDims(), (Shape{2, 4, 3}));
    }
}

} // namespace infini
