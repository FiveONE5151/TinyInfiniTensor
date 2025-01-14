#include "operators/transpose.h"
#include <algorithm>
#include <cstddef>
#include <iostream>
#include <ostream>
#include <vector>

namespace infini
{
    TransposeObj::TransposeObj(GraphObj *graph, Tensor input, Tensor output,
                               vector<int> permute)
        : OperatorObj(OpType::Transpose, {input}, {output})
    {
        auto rank = input->getRank();
        if (permute.empty())
        {
            for (size_t i = 0; i < rank; ++i)
            {
                transposePermute[i] = i;
            }
        }
        else
        {
            IT_ASSERT(rank == permute.size());
            transposePermute = std::move(permute);
        }
        IT_ASSERT(checkValid(graph));
    }
    /**
     vector<Shape> outputs_dim(inputs.size());
        for (auto &input : inputs) {
          auto input_dim = input->getDims();
          std::cout<< "input shape: ";
          std::for_each(input_dim.begin(), input_dim.end(), [](auto& a){std::cout<<a<<" ";});
          std::cout<<std::endl;
          auto output_dim = input_dim;
          for (size_t i = 0; i < this->getPermute().size(); ++i) {
            output_dim[i] = input_dim[this->getPermute()[i]];
          }
          outputs_dim.push_back(output_dim);
        }
        return outputs_dim;
     */
    optional<vector<Shape>> TransposeObj::inferShape(const TensorVec &inputs)
    {
      const auto A = inputs[0];
      auto input_dim = A->getDims();
      auto output_dim = input_dim;
      int rank = A->getRank();
      // =================================== 作业
      // ===================================
      // TODO：修改 output_dim，返回正确的 transpose 后的 shape
      // REF: https://onnx.ai/onnx/operators/onnx__Transpose.html#transpose-21
      // =================================== 作业
      // ===================================
      if (size_t(rank) != this->transposePermute.size()) {
        return std::nullopt;
      }
      for (size_t i = 0; i < this->getPermute().size(); ++i) {
        output_dim[i] = input_dim[this->getPermute()[i]];
      }

      return vector<vector<int>>{output_dim};
    }

    std::string TransposeObj::toString() const
    {
        std::ostringstream os;
        os << type.toString() << "[" << getGuid() << "]";
        os << "(";
        os << vecToString(inputs[0]->getDims()) << ",";
        os << "input=" << inputs[0]->getGuid() << ",";
        os << "output=" << outputs[0]->getGuid() << ")";
        return os.str();
    }
}; // namespace infini
