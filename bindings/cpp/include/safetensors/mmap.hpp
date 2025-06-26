// MIT License

// Copyright (c) 2023-2024 The ggml authors

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

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "fmt/format.h"
#include "rust/cxx.h"
#include "safetensors_abi/lib.h"

#if defined(__has_include)
// cppcheck-suppress preprocessorErrorDirective
#if __has_include(<unistd.h>)
#include <unistd.h>
#if defined(_POSIX_MAPPED_FILES)
#include <fcntl.h>
#include <sys/mman.h>
#endif
#if defined(_POSIX_MEMLOCK_RANGE)
#include <sys/resource.h>
#endif
#endif
#endif

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#ifndef PATH_MAX
#define PATH_MAX MAX_PATH
#endif
#include <io.h>
#endif

#if defined(__APPLE__)
#include <TargetConditionals.h>
#endif

namespace safetensors {

class File {
 public:
  inline static File SafeOpen(const std::string& filename,
                                    const std::string& mode = "rb") {
    FILE* file_ptr = std::fopen(filename.c_str(), mode.c_str());
    if (!file_ptr) {
      throw std::runtime_error(fmt::format("{}:{} can't open {}: {}", __FILE__,
                                           __LINE__, filename,
                                           std::strerror(errno)));
    }
    return File(std::make_unique<FileImpl>(file_ptr));
  }

  File(const File&) = delete;
  File& operator=(const File&) = delete;

  File(File&&) = default;
  File& operator=(File&&) = default;

  ~File() = default;

  int fileId() const;
  std::size_t size() const;
  std::size_t tell() const;
  void seek(const std::size_t offset, const int whence) const;
  void readRaw(void* ptr, const std::size_t len) const;
  std::uint32_t readU32() const;
  void writeRaw(const void* ptr, const std::size_t len) const;
  void writeU32(const std::uint32_t val) const;

 private:
  struct FileImpl;
  explicit File(std::unique_ptr<FileImpl> impl_ptr)
      : impl_ptr_(std::move(impl_ptr)) {}

  std::unique_ptr<FileImpl> impl_ptr_;
};

class Mmap {
 public:
  inline static Mmap SafeMap(File* file_ptr,
                             size_t prefetch = size_t(-1),
                             const bool numa = false) {
#if defined(_POSIX_MAPPED_FILES)
    int fd = file_ptr->fileId();
    int flags = MAP_SHARED;
    if (numa) {
      prefetch = 0;
    }

#if defined(__linux__)
    if (posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL)) {
      fmt::print("{}:{} posix_fadvise failed: {}", __FILE__, __LINE__,
                 std::strerror(errno));
    }
    if (prefetch) {
      flags |= MAP_POPULATE;
    }
#endif

    auto addr_ptr = mmap(nullptr, file_ptr->size(), PROT_READ, flags, fd, 0);
    if (addr_ptr == MAP_FAILED) {
      throw std::runtime_error(fmt::format("{}:{} mmap failed: {}", __FILE__,
                                           __LINE__, std::strerror(errno)));
    }

    if (prefetch > 0) {
      if (posix_madvise(addr_ptr, std::min(file_ptr->size(), prefetch),
                        POSIX_MADV_WILLNEED)) {
        fmt::print("{}:{} posix_madvise failed: {}", __FILE__, __LINE__,
                   std::strerror(errno));
      }
    }

    if (numa) {
      if (posix_madvise(addr_ptr, file_ptr->size(), POSIX_MADV_RANDOM)) {
        fmt::print("{}:{} posix_madvise failed: {}", __FILE__, __LINE__,
                   std::strerror(errno));
      }
    }

#elif defined(_WIN32)
    (void)numa;

    HANDLE hFile = (HANDLE)_get_osfhandle(file_ptr->fileId());

    HANDLE hMapping =
        CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);

    if (hMapping == NULL) {
      throw std::runtime_error(
          fmt::format("{}:{} CreateFileMappingA failed: {}", __FILE__, __LINE__,
                      win_err(GetLastError())));
    }

    auto addr_ptr = MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0);
    DWORD error = GetLastError();
    CloseHandle(hMapping);

    if (addr_ptr == NULL) {
      throw std::runtime_error(fmt::format("{}:{} MapViewOfFile failed: {}",
                                           __FILE__, __LINE__, win_err(error)));
    }

    if (prefetch > 0) {
#if _WIN32_WINNT >= 0x602
      BOOL(WINAPI * pPrefetchVirtualMemory)(HANDLE, ULONG_PTR,
                                            PWIN32_MEMORY_RANGE_ENTRY, ULONG);
      HMODULE hKernel32 = GetModuleHandleW(L"kernel32.dll");

      pPrefetchVirtualMemory =
          (decltype(pPrefetchVirtualMemory))(void*)GetProcAddress(  // NOLINT
              hKernel32, "PrefetchVirtualMemory");

      if (pPrefetchVirtualMemory) {
        WIN32_MEMORY_RANGE_ENTRY range;
        range.VirtualAddress = addr;
        range.NumberOfBytes = (SIZE_T)std::min(size, prefetch);
        if (!pPrefetchVirtualMemory(GetCurrentProcess(), 1, &range, 0)) {
          fmt::print("{}:{} PrefetchVirtualMemory failed: {}\n", __FILE__,
                     __LINE__, win_err(GetLastError()));
        }
      }
#else
#warning "skipping PrefetchVirtualMemory because _WIN32_WINNT < 0x602"
#endif

#else
#warning "mmap not supported on this platform"
    if (addr_ptr != nullptr) delete[] addr_ptr;
    addr_ptr = new std::uint8_t[file_ptr->size()];
    auto res = file_ptr->readRaw(addr_ptr, file_ptr->size());
    if (res.is_err()) {
      return result::Err(res.unwrap_err());
    }
#endif
    return Mmap(std::make_unique<MmapImpl>(addr_ptr, file_ptr->size()));
  }

  Mmap(const Mmap&) = delete;
  Mmap& operator=(const Mmap&) = delete;

  Mmap(Mmap&&) = default;
  Mmap& operator=(Mmap&&) = default;

  ~Mmap() = default;

  std::size_t size() const;
  std::uint8_t* data() const;
  void unmapFragment(const std::size_t first, const std::size_t last);

 private:
#if !defined(_POSIX_MAPPED_FILES) && !defined(_WIN32)
  static void* addr_ptr;
#endif
  struct MmapImpl;
  explicit Mmap(std::unique_ptr<MmapImpl> impl_ptr)
      : impl_ptr_(std::move(impl_ptr)) {}

  std::unique_ptr<MmapImpl> impl_ptr_;
};  // NOLINT

class Mlock {
 public:
  inline static Mlock SafeLock(void* addr_ptr) {
    return Mlock(std::make_unique<MlockImpl>(addr_ptr));
  }
  ~Mlock() = default;

  Mlock(const Mlock&) = delete;
  Mlock& operator=(const Mlock&) = delete;

  Mlock(Mlock&&) = default;
  Mlock& operator=(Mlock&&) = default;

  void growTo(const std::size_t target_size);

 private:
  struct MlockImpl;
  explicit Mlock(std::unique_ptr<MlockImpl> impl_ptr)
      : impl_ptr_(std::move(impl_ptr)) {}
  std::unique_ptr<MlockImpl> impl_ptr_;
};

// inline std::size_t pathMax()  { return PATH_MAX; }

}  // namespace safetensors
