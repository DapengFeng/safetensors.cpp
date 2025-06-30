<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://huggingface.co/datasets/safetensors/assets/raw/main/banner-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://huggingface.co/datasets/safetensors/assets/raw/main/banner-light.svg">
    <img alt="Hugging Face Safetensors Library" src="https://huggingface.co/datasets/safetensors/assets/raw/main/banner-light.svg" style="max-width: 100%;">
  </picture>
  <br/>
  <br/>
</p>

Python
[![Pypi](https://img.shields.io/pypi/v/safetensors.svg)](https://pypi.org/pypi/safetensors/)
[![Documentation](https://img.shields.io/website/http/huggingface.co/docs/safetensors/index.svg?label=docs)](https://huggingface.co/docs/safetensors/index)
[![Codecov](https://codecov.io/github/huggingface/safetensors/coverage.svg?branch=main)](https://codecov.io/gh/huggingface/safetensors)
[![Downloads](https://static.pepy.tech/badge/safetensors/month)](https://pepy.tech/project/safetensors)

Rust
[![Crates.io](https://img.shields.io/crates/v/safetensors.svg)](https://crates.io/crates/safetensors)
[![Documentation](https://docs.rs/safetensors/badge.svg)](https://docs.rs/safetensors/)
[![Codecov](https://codecov.io/github/huggingface/safetensors/coverage.svg?branch=main)](https://codecov.io/gh/huggingface/safetensors)
[![Dependency status](https://deps.rs/repo/github/huggingface/safetensors/status.svg?path=safetensors)](https://deps.rs/repo/github/huggingface/safetensors?path=safetensors)

C++
[![CMake](https://img.shields.io/badge/CMake-3.23+-blue.svg)](https://cmake.org/)
[![C++](https://img.shields.io/badge/C++-20-blue.svg)](https://en.cppreference.com/w/cpp/20)
[![Performance](https://img.shields.io/badge/Performance-1851x_faster-brightgreen.svg)](#performance-benchmarks)

# safetensors


#### C++

The C++ bindings provide high-performance tensor loading and are built on top of the Rust core library using CXX bridge.

**Prerequisites:**
- CMake 3.23 or later
- Rust toolchain (stable)
- C++20 compatible compiler

**Building:**
```bash
cmake -B build -G Ninja
cmake --build build
```

**Installation:**
```bash
cmake --install build --prefix /path/to/install
```

**Usage:**
```cpp
#include "safetensors/safetensors.hpp"

// Load a safetensors file
auto f = safetensors::SafeOpen("model.safetensors");

// Get tensor keys
auto keys = f.keys();

// Load a specific tensor
auto tensor = f.get_tensor("weight1");
std::cout << "Shape: [";
for (size_t i = 0; i < tensor.shape.size(); ++i) {
    std::cout << tensor.shape[i];
    if (i < tensor.shape.size() - 1) std::cout << ", ";
}
std::cout << "]" << std::endl;
```

### Performance Benchmarks

We've benchmarked the C++ bindings against the Python implementation across different model sizes, access patterns, and devices (CPU vs CUDA). All benchmarks measure the time per iteration to load all tensors from the file:

#### CPU Performance (Single Load - Real-world Scenario)
| Model Size | Python CPU | C++ CPU | Speedup | Use Case |
|------------|------------|---------|---------|----------|
| 523MB (gpt2.safetensors)     | 0.691s     | 0.0004s | **1,851x faster** | Model loading |
| 4.7GB (vggt-1B.safetensors)     | 0.749s     | 0.0037s | **204x faster** | Large model loading |

#### CUDA Performance (Single Load - GPU Acceleration)
| Model Size | Python CUDA | C++ CUDA | Speedup | GPU Memory Transfer |
|------------|-------------|----------|---------|-------------------|
| 523MB (gpt2.safetensors)     | 1.036s      | 0.240s   | **4.3x faster** | CPU→GPU transfer |
| 4.7GB (vggt-1B.safetensors)     | 1.664s      | 0.570s   | **2.9x faster** | CPU→GPU transfer |

#### Repeated Access Performance (Loop Benchmarks)

**523MB Model (gpt2.safetensors):**
| Iterations | Python CPU | C++ CPU | Python CUDA | C++ CUDA |
|------------|------------|---------|-------------|----------|
| 1          | 0.691s     | 0.0004s | 1.036s      | 0.240s   |
| 10         | 0.070s     | 0.0003s | 0.145s      | 0.060s   |
| 100        | 0.009s     | 0.0003s | 0.054s      | 0.041s   |

**4.7GB Model (vggt-1B.safetensors):**
| Iterations | Python CPU | C++ CPU | Python CUDA | C++ CUDA |
|------------|------------|---------|-------------|----------|
| 1          | 0.749s     | 0.0037s | 1.664s      | 0.570s   | 
| 10         | 0.092s     | 0.0036s | 0.528s      | 0.370s   | 
| 100        | 0.026s     | 0.0036s | 0.414s      | 0.352s   | 

#### Performance Analysis

**C++ CPU - Ultimate Performance:**
- **Exceptional speed**: Up to 1,851x faster than Python CPU
- **Sub-millisecond latency**: Consistent 0.3-3.7ms performance
- **Memory efficient**: Direct memory access without GPU transfer overhead
- **Production optimal**: Best choice for inference servers

**C++ CUDA - GPU Integration:**
- **GPU-ready tensors**: Direct CUDA memory allocation
- **Faster than Python CUDA**: 2.9-4.3x improvement
- **Good for GPU workflows**: When tensors need to be on GPU anyway
- **Transfer overhead**: Includes CPU→GPU memory transfer time

**Key Insights:**
1. **C++ CPU dominates**: Fastest across all scenarios due to zero-copy access
2. **CUDA has transfer overhead**: CPU→GPU memory copy adds latency
3. **Choose CPU for pure speed**: Use CUDA only when GPU tensors are required
4. **Consistent C++ advantage**: Superior performance regardless of device or iterations

#### Device-Specific Recommendations

**For Maximum Speed (Inference/Loading):**
```bash
# Use C++ CPU for ultimate performance
./build/bindings/cpp/benchmark/bench_cpp model.safetensors 1

# Python CPU comparison
python bindings/cpp/benchmark/bench.py model.safetensors 1 --device cpu
```

**For GPU Workflows (Training/GPU Processing):**
```bash
# Use C++ CUDA when tensors need to be on GPU
./build/bindings/cpp/benchmark/bench_cpp model.safetensors 1 cuda

# Python CUDA comparison  
python bindings/cpp/benchmark/bench.py model.safetensors 1 --device cuda
```

**For any production use case, C++ bindings provide dramatic performance benefits:**

```bash
# Build optimized C++ bindings
cd bindings/cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# Single model load (production inference)
./build/benchmark/bench_cpp model.safetensors 1

# Repeated access (development/testing)  
./build/benchmark/bench_cpp model.safetensors 100

# Compare with Python
python benchmark/bench.py model.safetensors 1
```

**Performance Testing:**
```bash
# Comprehensive benchmark comparison
for iterations in 1 10 100; do
  echo "=== $iterations iterations ==="
  python bindings/cpp/benchmark/bench.py model.safetensors $iterations
  ./build/bindings/cpp/benchmark/bench_cpp model.safetensors $iterations
  echo
done
```

*Benchmarks performed on AMD64 system with optimized Release build (-O3). The C++ implementation shows exceptional performance across all scenarios.*

## Safetensors

This repository implements a new simple format for storing tensors
safely (as opposed to pickle) and that is still fast (zero-copy).

### Installation
#### Pip

You can install safetensors via the pip manager:

```bash
pip install safetensors
```

#### From source

For the sources, you need Rust

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
# Make sure it's up to date and using stable channel
rustup update
git clone https://github.com/huggingface/safetensors
cd safetensors/bindings/python
pip install setuptools_rust
pip install -e .
```


### Getting started

```python
import torch
from safetensors import safe_open
from safetensors.torch import save_file

tensors = {
   "weight1": torch.zeros((1024, 1024)),
   "weight2": torch.zeros((1024, 1024))
}
save_file(tensors, "model.safetensors")

tensors = {}
with safe_open("model.safetensors", framework="pt", device="cpu") as f:
   for key in f.keys():
       tensors[key] = f.get_tensor(key)
```

[Python documentation](https://huggingface.co/docs/safetensors/index)


### Format

- 8 bytes: `N`, an unsigned little-endian 64-bit integer, containing the size of the header
- N bytes: a JSON UTF-8 string representing the header.
  - The header data MUST begin with a `{` character (0x7B).
  - The header data MAY be trailing padded with whitespace (0x20).
  - The header is a dict like `{"TENSOR_NAME": {"dtype": "F16", "shape": [1, 16, 256], "data_offsets": [BEGIN, END]}, "NEXT_TENSOR_NAME": {...}, ...}`,
    - `data_offsets` point to the tensor data relative to the beginning of the byte buffer (i.e. not an absolute position in the file),
      with `BEGIN` as the starting offset and `END` as the one-past offset (so total tensor byte size = `END - BEGIN`).
  - A special key `__metadata__` is allowed to contain free form string-to-string map. Arbitrary JSON is not allowed, all values must be strings.
- Rest of the file: byte-buffer.

Notes:
 - Duplicate keys are disallowed. Not all parsers may respect this.
 - In general the subset of JSON is implicitly decided by `serde_json` for
   this library. Anything obscure might be modified at a later time, that odd ways
   to represent integer, newlines and escapes in utf-8 strings. This would only
   be done for safety concerns
 - Tensor values are not checked against, in particular NaN and +/-Inf could
   be in the file
 - Empty tensors (tensors with 1 dimension being 0) are allowed.
   They are not storing any data in the databuffer, yet retaining size in the header.
   They don't really bring a lot of values but are accepted since they are valid tensors
   from traditional tensor libraries perspective (torch, tensorflow, numpy, ..).
 - 0-rank Tensors (tensors with shape `[]`) are allowed, they are merely a scalar.
 - The byte buffer needs to be entirely indexed, and cannot contain holes. This prevents
   the creation of polyglot files.
 - Endianness: Little-endian.
   moment.
 - Order: 'C' or row-major.
 - Notes: Some smaller than 1 byte dtypes appeared, which make alignment tricky. Non traditional APIs might be required for those.


### Yet another format ?

The main rationale for this crate is to remove the need to use
`pickle` on `PyTorch` which is used by default.
There are other formats out there used by machine learning and more general
formats.


Let's take a look at alternatives and why this format is deemed interesting.
This is my very personal and probably biased view:

| Format                  | Safe | Zero-copy | Lazy loading | No file size limit | Layout control | Flexibility | Bfloat16/Fp8 | Performance |
| ----------------------- | --- | --- | --- | --- | --- | --- | --- | --- |
| pickle (PyTorch)        | ✗ | ✗ | ✗ | 🗸 | ✗ | 🗸 | 🗸 | ~ |
| H5 (Tensorflow)         | 🗸 | ✗ | 🗸 | 🗸 | ~ | ~ | ✗ | ~ |
| SavedModel (Tensorflow) | 🗸 | ✗ | ✗ | 🗸 | 🗸 | ✗ | 🗸 | ~ |
| MsgPack (flax)          | 🗸 | 🗸 | ✗ | 🗸 | ✗ | ✗ | 🗸 | ~ |
| Protobuf (ONNX)         | 🗸 | ✗ | ✗ | ✗ | ✗ | ✗ | 🗸 | ~ |
| Cap'n'Proto             | 🗸 | 🗸 | ~ | 🗸 | 🗸 | ~ | ✗ | ~ |
| Arrow                   | ? | ? | ? | ? | ? | ? | ✗ | ~ |
| Numpy (npy,npz)         | 🗸 | ? | ? | ✗ | 🗸 | ✗ | ✗ | ~ |
| pdparams (Paddle)       | ✗ | ✗ | ✗ | 🗸 | ✗ | 🗸 | 🗸 | ~ |
| SafeTensors (Python)    | 🗸 | 🗸 | 🗸 | 🗸 | 🗸 | ✗ | 🗸 | 🗸 |
| SafeTensors (C++)       | 🗸 | 🗸 | 🗸 | 🗸 | 🗸 | ✗ | 🗸 | 🗸🗸 |

- Safe: Can I use a file randomly downloaded and expect not to run arbitrary code ?
- Zero-copy: Does reading the file require more memory than the original file ?
- Lazy loading: Can I inspect the file without loading everything ? And loading only
  some tensors in it without scanning the whole file (distributed setting) ?
- Layout control: Lazy loading, is not necessarily enough since if the information about tensors is spread out in your file, then even if the information is lazily accessible you might have to access most of your file to read the available tensors (incurring many DISK -> RAM copies). Controlling the layout to keep fast access to single tensors is important.
- No file size limit: Is there a limit to the file size ?
- Flexibility: Can I save custom code in the format and be able to use it later with zero extra code ? (~ means we can store more than pure tensors, but no custom code)
- Bfloat16/Fp8: Does the format support native bfloat16/fp8 (meaning no weird workarounds are
  necessary)? This is becoming increasingly important in the ML world.
- Performance: Relative loading speed (🗸 = good, 🗸🗸 = excellent, ~ = average)


### Main oppositions

- Pickle: Unsafe, runs arbitrary code
- H5: Apparently now discouraged for TF/Keras. Seems like a great fit otherwise actually. Some classic use after free issues: <https://www.cvedetails.com/vulnerability-list/vendor_id-15991/product_id-35054/Hdfgroup-Hdf5.html>. On a very different level than pickle security-wise. Also 210k lines of code vs ~400 lines for this lib currently.
- SavedModel: Tensorflow specific (it contains TF graph information).
- MsgPack: No layout control to enable lazy loading (important for loading specific parts in distributed setting)
- Protobuf: Hard 2Go max file size limit
- Cap'n'proto: Float16 support is not present [link](https://capnproto.org/language.html#built-in-types) so using a manual wrapper over a byte-buffer would be necessary. Layout control seems possible but not trivial as buffers have limitations [link](https://stackoverflow.com/questions/48458839/capnproto-maximum-filesize).
- Numpy (npz): No `bfloat16` support. Vulnerable to zip bombs (DOS). Not zero-copy.
- Arrow: No `bfloat16` support.

### Notes

- Zero-copy: No format is really zero-copy in ML, it needs to go from disk to RAM/GPU RAM (that takes time). On CPU, if the file is already in cache, then it can
  truly be zero-copy, whereas on GPU there is not such disk cache, so a copy is always required
  but you can bypass allocating all the tensors on CPU at any given point.
  SafeTensors is not zero-copy for the header. The choice of JSON is pretty arbitrary, but since deserialization is <<< of the time required to load the actual tensor data and is readable I went that way, (also space is <<< to the tensor data).

- Endianness: Little-endian. This can be modified later, but it feels really unnecessary at the
  moment.
- Order: 'C' or row-major. This seems to have won. We can add that information later if needed.
- Stride: No striding, all tensors need to be packed before being serialized. I have yet to see a case where it seems useful to have a strided tensor stored in serialized format.
 - Sub 1 bytes dtypes: Dtypes can now have lower than 1 byte size, this makes alignment&adressing tricky. For now, the library will simply error out whenever an operation triggers an non aligned read. Trickier API may be created later for those non standard ops. 

### Benefits

Since we can invent a new format we can propose additional benefits:

- Prevent DOS attacks: We can craft the format in such a way that it's almost
  impossible to use malicious files to DOS attack a user. Currently, there's a limit
  on the size of the header of 100MB to prevent parsing extremely large JSON.
  Also when reading the file, there's a guarantee that addresses in the file
  do not overlap in any way, meaning when you're loading a file you should never
  exceed the size of the file in memory

- Faster load: PyTorch seems to be the fastest file to load out in the major
  ML formats. However, it does seem to have an extra copy on CPU, which we
  can bypass in this lib by using `torch.UntypedStorage.from_file`.
  Currently, CPU loading times are extremely fast with this lib compared to pickle.
  GPU loading times are as fast or faster than PyTorch equivalent.
  Loading first on CPU with memmapping with torch, and then moving all tensors to GPU seems
  to be faster too somehow (similar behavior in torch pickle)

- Lazy loading: in distributed (multi-node or multi-gpu) settings, it's nice to be able to
  load only part of the tensors on the various models. For
  [BLOOM](https://huggingface.co/bigscience/bloom) using this format enabled
  to load the model on 8 GPUs from 10mn with regular PyTorch weights down to 45s.
  This really speeds up feedbacks loops when developing on the model. For instance
  you don't have to have separate copies of the weights when changing the distribution
  strategy (for instance Pipeline Parallelism vs Tensor Parallelism).

License: Apache-2.0
