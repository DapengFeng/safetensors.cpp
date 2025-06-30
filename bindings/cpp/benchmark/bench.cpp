/*
 * Copyright (c) 2025 Dapeng Feng
 * All rights reserved.
 */

#include <chrono>

#include "safetensors/safetensors.hpp"
#include "torch/torch.h"

torch::Dtype to_torch_dtype(safetensors::Dtype dtype) {
  switch (dtype) {
    case safetensors::Dtype::F64:
      return torch::kDouble;
    case safetensors::Dtype::F32:
      return torch::kFloat;
    case safetensors::Dtype::F16:
      return torch::kHalf;
    case safetensors::Dtype::BF16:
      return torch::kBFloat16;
    case safetensors::Dtype::I64:
      return torch::kLong;
    case safetensors::Dtype::I32:
      return torch::kInt;
    case safetensors::Dtype::I16:
      return torch::kShort;
    case safetensors::Dtype::I8:
      return torch::kChar;
    case safetensors::Dtype::U64:
      return torch::kUInt64;
    case safetensors::Dtype::U32:
      return torch::kUInt32;
    case safetensors::Dtype::U16:
      return torch::kUInt16;
    case safetensors::Dtype::U8:
      return torch::kByte;
    case safetensors::Dtype::BOOL:
      return torch::kBool;
    default:
      throw std::runtime_error("Unsupported dtype");
  }
}

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <path_to_safetensors_file> [<loop_count>] [<deice>(default: cpu)]"
              << std::endl;
    return 1;
  }

  int loop_count = 1;
  bool cuda = false;
  if (argc > 2) {
    try {
      loop_count = std::stoi(argv[2]);
      if (loop_count < 1) {
        std::cerr << "Loop count must be a positive integer." << std::endl;
        return 1;
      }
    } catch (const std::invalid_argument&) {
      std::cerr << "Invalid loop count: " << argv[2] << std::endl;
      return 1;
    }
    if (argc > 3) {
      std::string device = argv[3];
      if (device == "cuda") {
        cuda = true;
      } else if (device == "cpu") {
        cuda = false;
      } else {
        std::cerr << "Invalid device: " << device << std::endl;
        return 1;
      }
    }
  }

  auto f = safetensors::SafeOpen(argv[1]);
  auto start = std::chrono::high_resolution_clock::now();
  for(int i = 0; i < loop_count; ++i) {
    torch::OrderedDict<std::string, torch::Tensor> tensors;
    for (const auto& key : f.keys()) {
      torch::NoGradGuard no_grad;
      auto tensor = f.get_tensor(key);
      std::vector<std::int64_t> shape;
      shape.reserve(tensor.shape.size());
      std::transform(
          tensor.shape.begin(), tensor.shape.end(), std::back_inserter(shape),
          [](const auto& dim) { return static_cast<std::int64_t>(dim); });
      auto blob = torch::from_blob(
                    const_cast<void*>(tensor.data_ptr), shape,
                    torch::TensorOptions().dtype(to_torch_dtype(tensor.dtype)).pinned_memory(true));
      if (cuda) {
        tensors.insert(
            std::move(key), std::move(blob.to(torch::kCUDA, true)));
      } else {
        // Use pinned memory for CPU tensors
      tensors.insert(
          std::move(key), std::move(blob));
      }
    }
  }
  

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;
  std::cout << "Benchmark completed in " << duration.count() / loop_count << " seconds."
            << std::endl;


  return 0;
}
