#include "core/graph.h"
#include "core/blob.h"
#include "core/common.h"
#include "core/object.h"
#include "core/op_type.h"
#include "core/ref.h"
#include "core/runtime.h"
#include "operators/matmul.h"
#include "operators/transpose.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <ios>
#include <iostream>
#include <memory>
#include <numeric>
#include <ostream>
#include <queue>

namespace infini {

void GraphObj::addOperatorAndConnect(const Operator& op)
{
    sorted = false;
    ops.push_back(op);
    for (auto& input : op->getInputs()) {
        if (input) {
            input->addTarget(op);
            if (auto pred = input->getSource()) {
                pred->addSuccessors(op);
                op->addPredecessors(pred);
            }
        }
    }
    for (auto& output : op->getOutputs()) {
        if (output) {
            output->setSource(op);
            for (auto& succ : output->getTargets()) {
                succ->addPredecessors(op);
                op->addSuccessors(succ);
            }
        }
    }
}

string GraphObj::toString() const
{
    //std::cout<<"Enter Graph.toString()"<<std::endl;
    std::ostringstream oss;
    oss << "Graph Tensors:\n";

    for (const auto& tensor : tensors) {
        oss << tensor << "\n";
    }

    // std::cout<<"Graph Tensors written"<<std::endl;

    oss << "Graph operators:\n";
    for (const auto& op : ops) {
        //  std::cout<<"Writing operator: "<<op->toString()<<std::endl;
        vector<UidBaseType> preds, succs;
        for (auto& o : op->getPredecessors())
            preds.emplace_back(o->getGuid());
        for (auto& o : op->getSuccessors())
            succs.emplace_back(o->getGuid());
        oss << "OP " << op->getGuid();
        oss << ", pred " << vecToString(preds);
        oss << ", succ " << vecToString(succs);
        oss << ", " << op << "\n";
    }

    //std::cout<<"Graph Operators written"<<std::endl;
    return oss.str();
}

bool GraphObj::topo_sort()
{
    if (this->sorted) {
        return true;
    }
    std::vector<Operator> sorted; // 排序后的operator
    std::unordered_set<OperatorObj*> flags; // 记录已经被排序的operator
    sorted.reserve(ops.size());
    flags.reserve(ops.size());
    while (sorted.size() < ops.size()) {
        // Any node is move to sorted in this loop.
        auto modified = false;
        for (auto const& op : ops) {
            // if初始化语句(cpp17), 获取当前operator的所有输入tensor
            // 条件1: 当前operator没有被处理过
            // 条件2: 输入tensor的来源operator为空或者已经被排序处理(即入度为0)
            if (auto const& inputs = op->getInputs();
                flags.find(op.get()) == flags.end() && std::all_of(inputs.begin(), inputs.end(), [&flags](auto const& input) {
                    auto ptr = input->getSource().get();
                    return !ptr || flags.find(ptr) != flags.end();
                })) {
                modified = true;
                sorted.emplace_back(op);
                flags.insert(op.get());
            }
        }
        if (!modified) // 没有入度为0的节点, 说明图有环
        {
            return false;
        }
    }
    this->ops = std::move(sorted); //修改成员变量为排序后的operators
    return this->sorted = true;
}

void GraphObj::optimize()
{
    // =================================== 作业 ===================================
    // TODO: 设计一个算法来实现指定的图优化规则
    // 图优化规则如下：
    // 1. 去除冗余的算子（例如，两个相邻的算子都是 transpose 算子，且做的是相反的操作，可以将其全部删除）
    // 2. 合并算子（例如，矩阵乘算子中含有属性transA、transB，如果其输入存在transpose，且对最后两个维度做交换，就可以将transpose融入到矩阵乘算子的属性中去）
    // =================================== 作业 ===================================

    // 1. 去除冗余的算子（例如，两个相邻的算子都是 transpose 算子，且做的是相反的操作，可以将其全部删除）
    // 遍历所有算子, 观察它的前几个和后几个相邻算子是否都是transpose
    std::unordered_set<Operator> flags; // 记录已经被删除的operator
    for (auto const& op : ops) {
        // 优化ranspose
        if (flags.find(op) == flags.end() and op->getOpType() == OpType::Transpose) {
            auto transposeOp = std::dynamic_pointer_cast<TransposeObj>(op);
            auto predecessors = transposeOp->getPredecessors();
            auto successors = transposeOp->getSuccessors();

            // 优化连续相反的transpose
            if (!predecessors.empty() && predecessors[0]->getOpType() == OpType::Transpose) {

                // TRANSPOSE只能有1个输入
                IT_ASSERT(predecessors.size() == 1);

                auto pre_op = std::dynamic_pointer_cast<TransposeObj>(predecessors[0]);
                if (!pre_op) {
                    continue;
                }
                auto perm = transposeOp->getPermute();
                auto pre_perm = pre_op->getPermute();

                // 判断是否是相反的操作
                // 即perm是否相等
                if (perm == pre_perm) {
                    // this->removeOperator(pre_op);
                    // this->removeOperator(op);
                    // 不能直接在循环内部remove, 因为循环的iterator下一个会迭代到被删除的operator, 应该先记录, 循环结束后再remove
                    flags.insert(pre_op);
                    flags.insert(op);
                }
            }
            // 2. 合并算子（例如，矩阵乘算子中含有属性transA、transB，如果其输入存在transpose，且对最后两个维度做交换，就可以将transpose融入到矩阵乘算子的属性中去）
            else if (!successors.empty() && successors[0]->getOpType() == OpType::MatMul) {
                auto matmulOp = std::dynamic_pointer_cast<MatmulObj>(successors[0]);
                auto perm = transposeOp->getPermute();

                int m = perm.size() - 2;
                int n = perm.size() - 1;
                bool legalTrans = false;

                for (int i = 0; i < (int)perm.size(); ++i) {
                    if (i < (int)perm.size() - 2) {
                        if (perm[i] != i)
                            break;
                    } else {
                        if (perm[i] == n && perm[i + 1] == m) {
                            legalTrans = true;
                            break;
                        } else
                            break;
                    }
                }

                if (!legalTrans) {
                    continue;
                }

                if (transposeOp->getOutputs()[0] == matmulOp->getInputs()[0]) {
                    matmulOp->setTransA(!matmulOp->getTransA());
                    flags.insert(transposeOp);
                } else if (transposeOp->getOutputs()[0] == matmulOp->getInputs()[1]) {
                    matmulOp->setTransB(!matmulOp->getTransB());
                    flags.insert(transposeOp);
                }
            }
        }
    }

    // 删除operator和其他所有相关信息
    for (auto& deleted : flags) {
        auto input = deleted->getInputs()[0];
        auto output = deleted->getOutputs()[0];

        // !! 从tensor出发, 来操作节点之间的连接 !!
        // 先把input tensor的target(即deleted op本身)删除
        input->removeTarget(deleted);
        bool has_successor = deleted->getSuccessors().size();

        // 如果deleted有后续节点
        // 那么把后续节点的input tensor替换
        // 并把input的target设置成后续节点
        // 再把当前节点从后续节点的predecessor删除
        if (has_successor) {
            for (auto& next_op : deleted->getSuccessors()) {
                next_op->replaceInput(output, input);
                input->addTarget(next_op);
                next_op->removePredecessors(deleted);
            }
        }

        // 删除output tensor
        removeTensor(output);

        // 如果deleted有前面节点
        // 那么要先把当前节点从前面节点的successor删除
        // 再根据是否有后续节点, 判断是否要把后续节点加入到前面节点的successor中
        if (deleted->getPredecessors().size()) {
            for (auto& prev_op : deleted->getPredecessors()) {
                prev_op->removeSuccessors(deleted);
                if (has_successor) {
                    for (auto& next_op : deleted->getSuccessors()) {
                        prev_op->addSuccessors(next_op);
                    }
                }
            }
        }

        // 从graph中删除节点
        removeOperator(deleted);
    }
}

Tensor GraphObj::getTensor(int fuid) const
{
    for (auto tensor : tensors) {
        if (tensor->getFuid() == fuid) {
            return tensor;
        }
    }
    return nullptr;
}

void GraphObj::shape_infer()
{
    for (auto& op : ops) {
        auto ans = op->inferShape();
        IT_ASSERT(ans.has_value());
        auto oldOutputs = op->getOutputs();
        IT_ASSERT(ans.value().size() == oldOutputs.size());
        // replace the old outputshape and size with new one
        for (int i = 0; i < (int)ans.value().size(); ++i) {
            auto newShape = ans.value()[i];
            auto oldShape = oldOutputs[i]->getDims();
            auto fuid = oldOutputs[i]->getFuid();
            if (newShape != oldShape) {
                auto tensor = this->getTensor(fuid);
                tensor->setShape(newShape);
            }
        }
    }
}

void GraphObj::dataMalloc()
{
    // topological sorting first
    IT_ASSERT(topo_sort() == true);

    // =================================== 作业 ===================================
    // TODO：利用 allocator 给计算图分配内存
    // HINT: 获取分配好的内存指针后，可以调用 tensor 的 setDataBlob 函数给 tensor 绑定内存
    // =================================== 作业 ===================================
    unordered_map<int, uint64_t> ref_count, offsets;
    auto graph_inputs = getInputs();
    auto graph_outputs = getOutputs();
    // 输入和输出tensor的空间不能被用于回收reused，所以先分配输入输出的tensor
    //获取每个tensor的引用计数，用于判断tensor在后续是否还会被使用到
    for (auto const& tensor : tensors) {
        auto fuid = tensor->getFuid();
        auto input_it = std::find(graph_inputs.begin(), graph_inputs.end(), tensor);
        auto output_it = std::find(graph_outputs.begin(), graph_outputs.end(), tensor);

        if (input_it == graph_inputs.end() and output_it == graph_outputs.end()) {
            // 直接使用缓存的数据，不用再遍历所有op，线性复杂度
            ref_count[fuid] = tensor->getTargets().size();
        } else { // graph input or output, allocate memory and cannot be reused
            auto size = tensor->getBytes();
            offsets[fuid] = allocator.alloc(size);
            //std::cout << "Allocating for input/output tensor " << fuid << std::endl;
        }
    }
    unordered_map<int, uint64_t> act_ref_count = ref_count;
    // std::cout << "Initial ref count:\n";
    // for (auto const& item : act_ref_count) {
    //     std::cout << "tensor " << item.first << ": " << item.second << std::endl;
    // }

    // 由于输入tensor已经被全部分配，所以首先分配op的output tensor，即计算图的中间tensor
    // 再看op的input需不需要重用
    for (auto const& op : ops) {
        auto const& inputs = op->getInputs();
        auto const& outputs = op->getOutputs();
        for (auto const& output : outputs) {
            auto fuid = output->getFuid();
            auto size = output->getBytes();

            // 如果是图的输出tensor，则不需要分配
            if (act_ref_count.find(fuid) == act_ref_count.end())
                continue;

            // sorted_tensors.push_back(fuid);
            // used += size;
            //std::cout << "Allocating for tensor " << fuid << std::endl;
            auto offset = allocator.alloc(size);
            offsets[fuid] = offset;
        }

        for (auto const& input : inputs) {
            auto fuid = input->getFuid();
            auto size = input->getBytes();
            act_ref_count[fuid] -= 1;

            // no longer be used, recycle the memory
            if (act_ref_count[fuid] == 0) {
                //std::cout << "releasing tensor " << fuid << std::endl;
                act_ref_count.erase(fuid);
                allocator.free(offsets[fuid], size);
            }
        }
    }

    auto hptr = allocator.getPtr();

    // 绑定datablob
    for (auto& tensor : tensors) {
        void* addr = static_cast<char*>(hptr) + offsets[tensor->getFuid()];
        tensor->setDataBlob(make_ref<BlobObj>(runtime, addr));
    }

    allocator.info();
}

Tensor GraphObj::addTensor(Shape dim, DataType dtype)
{
    return tensors.emplace_back(make_ref<TensorObj>(dim, dtype, runtime));
}

Tensor GraphObj::addTensor(const Tensor& tensor)
{
    IT_ASSERT(tensor->getRuntime() == runtime,
        std::string("Tensor runtime mismatch: cannot add a tenosr in ") + tensor->getRuntime()->toString() + " to " + runtime->toString());
    tensors.emplace_back(tensor);
    return tensor;
}

TensorVec GraphObj::addTensor(const TensorVec& tensors)
{
    for (auto& t : tensors)
        addTensor(t);
    return tensors;
}

// tensor's "source" and "target" must be in "ops".
// tensor has no "source" and no "target" must not exist.
// "inputs" or "outputs" of operators must be in "tensors"
// "predecessors" and "successors" of an operator of "ops" must be in "ops".
bool GraphObj::checkValid() const
{
    for (auto tensor : tensors) {
        IT_ASSERT(!(tensor->getTargets().size() == 0 && nullptr == tensor->getSource()));
        for (auto op : tensor->getTargets()) {
            IT_ASSERT(std::find(ops.begin(), ops.end(), op) != ops.end());
        }
        auto op = tensor->getSource();
        IT_ASSERT(!(op && std::find(ops.begin(), ops.end(), op) == ops.end()));
    }
    for (auto op : ops) {
        for (auto tensor : op->getInputs()) {
            IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) != tensors.end());
        }
        for (auto tensor : op->getOutputs()) {
            IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) != tensors.end());
        }
        for (auto pre : op->getPredecessors()) {
            IT_ASSERT(std::find(ops.begin(), ops.end(), pre) != ops.end());
        }
        for (auto suc : op->getSuccessors()) {
            IT_ASSERT(std::find(ops.begin(), ops.end(), suc) != ops.end());
        }
    }
    std::set<UidBaseType> s;
    // check whether two tensors with the same FUID exist
    for (auto tensor : tensors) {
        int cnt = s.count(tensor->getFuid());
        IT_ASSERT(cnt == 0, std::to_string(tensor->getFuid()));
        s.insert(tensor->getFuid());
    }
    return true;
}

} // namespace infini