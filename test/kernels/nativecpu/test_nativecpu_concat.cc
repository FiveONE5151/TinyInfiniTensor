#include "core/graph.h"
#include "core/runtime.h"
#include "core/tensor.h"
#include "operators/concat.h"

#include "operators/transpose.h"
#include "test.h"
#include "utils/data_generator.h"
#include <gtest/gtest.h>
#include <iostream>
#include <ostream>
#include <utility>

namespace infini {

TEST(Concat, NativeCpu)
{
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);

    auto t1 = g->addTensor({ 2, 2, 3, 1 }, DataType::Float32);
    auto t2 = g->addTensor({ 2, 2, 1, 1 }, DataType::Float32);
    auto t3 = g->addTensor({ 2, 2, 2, 1 }, DataType::Float32);
    auto op = g->addOp<ConcatObj>(TensorVec { std::move(t1), std::move(t2), std::move(t3) }, nullptr, 2);
    std::cout << "t1==nullptr?" << (t1 == nullptr) << std::endl;
    g->dataMalloc();
    std::cout << "space allocated" << std::endl;

    g->getInputs()[0]->setData(IncrementalGenerator());
    g->getInputs()[1]->setData(OneGenerator());
    g->getInputs()[2]->setData(OneGenerator());
    std::cout << "Data set" << std::endl;
    runtime->run(g);
    EXPECT_TRUE(op->getOutput()->equalData(
        vector<float> { 0, 1, 2, 1, 1, 1, 3, 4, 5, 1, 1, 1,
            6, 7, 8, 1, 1, 1, 9, 10, 11, 1, 1, 1 }));
}

TEST(Concat, NativeCpuSWallocator)
{
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);

    auto i1 = g->addTensor({ 1, 2, 3, 1 }, DataType::Float32);
    auto i2 = g->addTensor({ 2, 2, 1, 1 }, DataType::Float32);
    auto t1 = g->addTensor({ 2, 1, 3, 1 }, DataType::Float32);
    auto t2 = g->addTensor({ 2, 1, 1, 3 }, DataType::Float32);
    auto t3 = g->addTensor({ 2, 1, 1, 2 }, DataType::Float32);
    auto t4 = g->addTensor({ 2, 1, 1, 5 }, DataType::Float32);
    auto peak = t3->getBytes() + t2->getBytes() + i2->getBytes();
    std::cout << "peak used should be: " << peak << " bytes\n";
    auto op1 = g->addOpWithOutputs<TransposeObj>(i1, t1, Shape { 1, 0, 2, 3 });
    auto op2 = g->addOpWithOutputs<TransposeObj>(t1, t2, Shape { 0, 1, 3, 2 });
    auto op3 = g->addOpWithOutputs<TransposeObj>(i2, t3, Shape { 0, 2, 3, 1 });
    auto op4 = g->addOpWithOutputs<ConcatObj>(TensorVec { t2, t3 }, t4, 3);
    std::cout << "Graph: \n"
              << g->toString() << std::endl;

    g->dataMalloc();
    std::cout << "Graph: \n"
              << g->toString() << std::endl;

    i1->setData(IncrementalGenerator());
    i2->setData(IncrementalGenerator());
    t1->setData(IncrementalGenerator());
    t2->setData(IncrementalGenerator());
    t3->setData(IncrementalGenerator());
    runtime->run(g);

    std::cout << "output:\n "
              << g->getOutputs()[0] << std::endl;
}
} // namespace infini
