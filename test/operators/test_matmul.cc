#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "operators/matmul.h"

#include "test.h"
#include <iostream>
#include <ostream>

namespace infini {
using ExpectOutput = vector<float>;

TEST(Matmul, ShapeInference1)
{
    auto runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);

    auto A = g->addTensor(Shape { 1, 3, 5 });
    auto B = g->addTensor(Shape { 1, 5, 2 });

    //这里开始出问题
    auto matmul = g->addOp<MatmulObj>(A, B, nullptr);
    std::cout << "matmul: " << matmul->toString() << std::endl;
    auto C = matmul->getOutputs()[0];
    std::cout << "C dimensions: ";
    for (const auto& dim : C->getDims()) {
        std::cout << dim << " ";
    }
    std::cout << std::endl;
    EXPECT_EQ(C->getDims(), (Shape { 1, 3, 2 }));
}

TEST(Matmul, ShapeInference2)
{
    auto runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);
    auto A = g->addTensor(Shape { 3, 5, 4 });
    auto B = g->addTensor(Shape { 3, 5, 2 });
    auto matmul = g->addOp<MatmulObj>(A, B, nullptr, true, false);
    auto C = matmul->getOutputs()[0];
    std::cout << "matmul: " << matmul->toString() << std::endl;

    std::cout << "C dimensions: ";
    for (const auto& dim : C->getDims()) {
        std::cout << dim << " ";
    }
    std::cout << std::endl;
    EXPECT_EQ(C->getDims(), (Shape { 3, 4, 2 }));
}

TEST(Matmul, ShapeInference3)
{
    auto runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);
    auto A = g->addTensor(Shape { 1, 2, 3, 5 });
    auto B = g->addTensor(Shape { 1, 1, 5, 2 });
    auto matmul = g->addOp<MatmulObj>(A, B, nullptr);
    auto C = matmul->getOutputs()[0];
    std::cout << "matmul: " << matmul->toString() << std::endl;

    std::cout << "C dimensions: ";
    for (const auto& dim : C->getDims()) {
        std::cout << dim << " ";
    }
    std::cout << std::endl;
    EXPECT_EQ(C->getDims(), (Shape { 1, 2, 3, 2 }));
}

TEST(Matmul, ShapeInference4)
{
    auto runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);
    auto A = g->addTensor(Shape { 2, 3, 5, 4 });
    auto B = g->addTensor(Shape { 1, 3, 5, 2 });
    auto matmul = g->addOp<MatmulObj>(A, B, nullptr, true, false);
    auto C = matmul->getOutputs()[0];
    std::cout << "matmul: " << matmul->toString() << std::endl;

    std::cout << "C dimensions: ";
    for (const auto& dim : C->getDims()) {
        std::cout << dim << " ";
    }
    std::cout << std::endl;
    EXPECT_EQ(C->getDims(), (Shape { 2, 3, 4, 2 }));
}

TEST(Matmul, ShapeInference5)
{
    auto runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);
    auto A = g->addTensor(Shape { 2, 3, 5, 4 });
    auto B = g->addTensor(Shape { 1, 3, 2, 5 });
    auto matmul = g->addOp<MatmulObj>(A, B, nullptr, true, true);
    auto C = matmul->getOutputs()[0];
    std::cout << "matmul: " << matmul->toString() << std::endl;

    std::cout << "C dimensions: ";
    for (const auto& dim : C->getDims()) {
        std::cout << dim << " ";
    }
    std::cout << std::endl;
    EXPECT_EQ(C->getDims(), (Shape { 2, 3, 4, 2 }));
}

// TEST(Matmul, ShapeInference6)
// {
//     auto runtime = NativeCpuRuntimeObj::getInstance();
//     Graph g = make_ref<GraphObj>(runtime);
//     auto A = g->addTensor(Shape { 5, 4 });
//     auto B = g->addTensor(Shape { 4, 5 });
//     auto matmul = g->addOp<MatmulObj>(A, B, nullptr, true, true);
//     auto C = matmul->getOutputs()[0];
//     std::cout << "matmul: " << matmul->toString() << std::endl;
//     std::cout << "C dimensions: ";
//     std::cout << vecToString(C->getDims()) << std::endl;

//     EXPECT_EQ(C->getDims(), (Shape { 5, 5 }));
// }

// TEST(Matmul, ShapeInference7)
// {
//     auto runtime = NativeCpuRuntimeObj::getInstance();
//     Graph g = make_ref<GraphObj>(runtime);
//     auto A = g->addTensor(Shape { 5, 4 });
//     auto B = g->addTensor(Shape { 5, 5 });
//     auto matmul = g->addOp<MatmulObj>(A, B, nullptr, true, true);
//     auto C = matmul->getOutputs()[0];
//     std::cout << "matmul: " << matmul->toString() << std::endl;
//     std::cout << "C dimensions: ";
//     std::cout << vecToString(C->getDims()) << std::endl;

//     EXPECT_EQ(C->getDims(), (Shape {}));
// }

// TEST(Matmul, ShapeInference8)
// {
//     auto runtime = NativeCpuRuntimeObj::getInstance();
//     Graph g = make_ref<GraphObj>(runtime);
//     auto A = g->addTensor(Shape { 4 });
//     auto B = g->addTensor(Shape { 4, 5 });
//     auto matmul = g->addOp<MatmulObj>(A, B, nullptr, true, true);
//     auto C = matmul->getOutputs()[0];
//     std::cout << "matmul: " << matmul->toString() << std::endl;
//     std::cout << "C dimensions: ";
//     std::cout << vecToString(C->getDims()) << std::endl;

//     EXPECT_EQ(C->getDims(), (Shape { 1, 5 }));
// }
// TEST(Matmul, ShapeInference9)
// {
//     auto runtime = NativeCpuRuntimeObj::getInstance();
//     Graph g = make_ref<GraphObj>(runtime);
//     auto A = g->addTensor(Shape { 4 });
//     auto B = g->addTensor(Shape { 5, 5 });
//     auto matmul = g->addOp<MatmulObj>(A, B, nullptr, true, true);
//     auto C = matmul->getOutputs()[0];
//     std::cout << "matmul: " << matmul->toString() << std::endl;
//     std::cout << "C dimensions: ";
//     std::cout << vecToString(C->getDims()) << std::endl;

//     EXPECT_EQ(C->getDims(), (Shape {}));
// }
// TEST(Matmul, ShapeInference10)
// {
//     auto runtime = NativeCpuRuntimeObj::getInstance();
//     Graph g = make_ref<GraphObj>(runtime);
//     auto A = g->addTensor(Shape { 4, 5 });
//     auto B = g->addTensor(Shape { 5 });
//     auto matmul = g->addOp<MatmulObj>(A, B, nullptr, true, true);
//     auto C = matmul->getOutputs()[0];
//     std::cout << "matmul: " << matmul->toString() << std::endl;
//     std::cout << "C dimensions: ";
//     std::cout << vecToString(C->getDims()) << std::endl;

//     EXPECT_EQ(C->getDims(), (Shape { 4, 1 }));
// }

// TEST(Matmul, ShapeInference11)
// {
//     auto runtime = NativeCpuRuntimeObj::getInstance();
//     Graph g = make_ref<GraphObj>(runtime);
//     auto A = g->addTensor(Shape { 4, 5 });
//     auto B = g->addTensor(Shape { 4 });
//     auto matmul = g->addOp<MatmulObj>(A, B, nullptr, true, true);
//     auto C = matmul->getOutputs()[0];
//     std::cout << "matmul: " << matmul->toString() << std::endl;
//     std::cout << "C dimensions: ";
//     std::cout << vecToString(C->getDims()) << std::endl;

//     EXPECT_EQ(C->getDims(), (Shape {}));
// }

// TEST(Matmul, ShapeInference12)
// {
//     auto runtime = NativeCpuRuntimeObj::getInstance();
//     Graph g = make_ref<GraphObj>(runtime);
//     auto A = g->addTensor(Shape { 3 });
//     auto B = g->addTensor(Shape { 2, 3, 4 });
//     auto matmul = g->addOp<MatmulObj>(A, B, nullptr, true, true);
//     auto C = matmul->getOutputs()[0];
//     std::cout << "matmul: " << matmul->toString() << std::endl;
//     std::cout << "C dimensions: ";
//     std::cout << vecToString(C->getDims()) << std::endl;

//     EXPECT_EQ(C->getDims(), (Shape { 2, 4 }));
// }

// TEST(Matmul, ShapeInference13)
// {
//     auto runtime = NativeCpuRuntimeObj::getInstance();
//     Graph g = make_ref<GraphObj>(runtime);
//     auto A = g->addTensor(Shape { 4 });
//     auto B = g->addTensor(Shape { 2, 3, 4 });
//     auto matmul = g->addOp<MatmulObj>(A, B, nullptr, true, true);
//     auto C = matmul->getOutputs()[0];
//     std::cout << "matmul: " << matmul->toString() << std::endl;
//     std::cout << "C dimensions: ";
//     std::cout << vecToString(C->getDims()) << std::endl;

//     EXPECT_EQ(C->getDims(), (Shape {}));
// }

// TEST(Matmul, ShapeInference14)
// {
//     auto runtime = NativeCpuRuntimeObj::getInstance();
//     Graph g = make_ref<GraphObj>(runtime);
//     auto A = g->addTensor(Shape { 2, 3, 4 });
//     auto B = g->addTensor(Shape { 4 });
//     auto matmul = g->addOp<MatmulObj>(A, B, nullptr, true, true);
//     auto C = matmul->getOutputs()[0];
//     std::cout << "matmul: " << matmul->toString() << std::endl;
//     std::cout << "C dimensions: ";
//     std::cout << vecToString(C->getDims()) << std::endl;

//     EXPECT_EQ(C->getDims(), (Shape { 2, 3 }));
// }

// TEST(Matmul, ShapeInference15)
// {
//     auto runtime = NativeCpuRuntimeObj::getInstance();
//     Graph g = make_ref<GraphObj>(runtime);
//     auto A = g->addTensor(Shape { 2, 3, 4 });
//     auto B = g->addTensor(Shape { 5 });
//     auto matmul = g->addOp<MatmulObj>(A, B, nullptr, true, true);
//     auto C = matmul->getOutputs()[0];
//     std::cout << "matmul: " << matmul->toString() << std::endl;
//     std::cout << "C dimensions: ";
//     std::cout << vecToString(C->getDims()) << std::endl;

//     EXPECT_EQ(C->getDims(), (Shape {}));
// }
}; // namespace infini