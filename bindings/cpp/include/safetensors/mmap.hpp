// MIT License

// Copyright (c) 2023-2024 The ggml authors
// Copyright (c) 2025 Dapeng Feng

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

namespace safetensors {

class File {
 public:
#if defined(_WIN32)
  explicit File(const std::filesystem::path &fname,
                const std::filesystem::path &mode = L"rb");
#else
  explicit File(const std::filesystem::path &fname,
                const std::filesystem::path &mode = "rb");
#endif

  File(const File &) = delete;
  File &operator=(const File &) = delete;

  File(File &&) = default;
  File &operator=(File &&) = default;

  ~File();

  int fileId() const;  // fileno overload

  std::size_t size() const;
  std::size_t tell() const;
  void seek(const std::size_t offset, const int whence) const;

  void readRaw(void *ptr, const std::size_t len) const;
  std::uint32_t readU32() const;

  void writeRaw(const void *ptr, const std::size_t len) const;
  void writeU32(const std::uint32_t val) const;

 private:
  struct impl;
  std::unique_ptr<impl> pimpl;
};

struct Mmap {
  explicit Mmap(File *file,
                const std::size_t prefetch = (std::size_t)-1,
                const bool numa = false);
  Mmap(const Mmap &) = delete;
  Mmap &operator=(const Mmap &) = delete;

  Mmap(Mmap &&) = default;
  Mmap &operator=(Mmap &&) = default;

  ~Mmap();

  std::size_t size() const;
  void *addr() const;
  std::uint8_t *data() const;

  void unmapFragment(const std::size_t first, const std::size_t last);

  static const bool SUPPORTED;

 private:
  struct impl;
  std::unique_ptr<impl> pimpl;
};

struct Mlock {
  Mlock();
  ~Mlock();

  Mlock(const Mlock &) = delete;
  Mlock &operator=(const Mlock &) = delete;

  Mlock(Mlock &&) = default;
  Mlock &operator=(Mlock &&) = default;

  void init(void *ptr);
  void growTo(const std::size_t target_size);

  static const bool SUPPORTED;

 private:
  struct impl;
  std::unique_ptr<impl> pimpl;
};

std::size_t path_max();

}  // namespace safetensors
