#include "operators/concat.h"
#include "core/common.h"
#include "utils/operator_utils.h"

namespace infini {
ConcatObj::ConcatObj(GraphObj* graph, TensorVec inputs, Tensor output, int _dim)
    : OperatorObj(OpType::Concat, inputs, { output })
{
    int rank = inputs[0]->getRank();
    dim = get_real_axis(_dim, rank);
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> ConcatObj::inferShape(const TensorVec& inputs)
{
    Shape dims = inputs[0]->getDims();
    auto rank = inputs[0]->getRank();

    // =================================== 作业 ===================================
    // TODO：修改 dims，返回正确的 concat 后的 shape
    // REF: https://onnx.ai/onnx/operators/onnx__Concat.html#concat-13
    // =================================== 作业 ===================================
    dims[dim] = 0;
    for (auto& input : inputs) {
        //拼接的向量必须维度相同
        IT_ASSERT(rank == input->getRank());
        auto iDims = input->getDims();
        auto rank = iDims.size();
        for (int i = 0; i < (int)rank; ++i) {
            if (i == dim) {
                dims[dim] += iDims[i];
                continue;
            }
            //除了拼接的那一条轴其他轴必须相等
            IT_ASSERT(iDims[i] == dims[i]);
        }
    }
    return { { dims } };
}

std::string ConcatObj::toString() const
{
    std::ostringstream os;
    os << "Concat[" << getGuid() << "]";
    os << "(";
    for (auto input : inputs)
        os << vecToString(input->getDims()) << ",";
    os << "dim=" << dim << ",";
    os << "input=";
    for (auto input : inputs)
        os << input->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

} // namespace infini
