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

#include "safetensors/mmap.hpp"

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstring>
#include <filesystem>
#include <stdexcept>

#include "fmt/format.h"

#ifdef __has_include
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

#if defined(_WIN32)
static std::string winErr(DWORD err) {
  LPSTR buf;
  std::size_t size = FormatMessageA(
      FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
          FORMAT_MESSAGE_IGNORE_INSERTS,
      NULL, err, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPSTR)&buf, 0,
      NULL);
  if (!size) {
    return fmt::format("Win32 error code: {}", err);
  }
  std::string ret(buf, size);
  LocalFree(buf);
  return ret;
}
#endif

// File

struct File::impl {
  impl(const std::filesystem::path& fname, const std::filesystem::path& mode) {
#if defined(_WIN32)
    fp = _wfopen(fname.c_str(), mode.c_str());
#else
    fp = std::fopen(fname.c_str(), mode.c_str());
#endif
    if (fp == NULL) {
      throw std::runtime_error(fmt::format("failed to open {}: {}",
                                           fname.string(), strerror(errno)));
    }
    size = std::filesystem::file_size(fname);
  }

  std::size_t tell() const {
#ifdef _WIN32
    __int64 ret = _ftelli64(fp);
#else
    auto ret = std::ftell(fp);
#endif
    if (ret == -1) {
      throw std::runtime_error(fmt::format("ftell error: {}", strerror(errno)));
    }

    return static_cast<std::size_t>(ret);
  }

  void seek(const std::size_t offset, const int whence) const {
#ifdef _WIN32
    int ret = _fseeki64(fp, (__int64)offset, whence);
#else
    int ret = std::fseek(fp, (long)offset, whence);  // NOLINT
#endif
    if (ret != 0) {
      throw std::runtime_error(fmt::format("seek error: {}", strerror(errno)));
    }
  }

  void readRaw(void* ptr, const std::size_t len) const {
    if (len == 0) {
      return;
    }
    errno = 0;
    std::size_t ret = std::fread(ptr, len, 1, fp);
    if (ferror(fp)) {
      throw std::runtime_error(fmt::format("read error: {}", strerror(errno)));
    }
    if (ret != 1) {
      throw std::runtime_error("unexpectedly reached end of file");
    }
  }

  std::uint32_t readU32() const {
    std::uint32_t ret;
    readRaw(&ret, sizeof(ret));
    return ret;
  }

  void writeRaw(const void* ptr, const std::size_t len) const {
    if (len == 0) {
      return;
    }
    errno = 0;
    std::size_t ret = std::fwrite(ptr, len, 1, fp);
    if (ret != 1) {
      throw std::runtime_error(fmt::format("write error: {}", strerror(errno)));
    }
  }

  void writeU32(const std::uint32_t val) const { writeRaw(&val, sizeof(val)); }

  ~impl() {
    if (fp) {
      std::fclose(fp);
    }
  }

  FILE* fp;
  std::size_t size;
};

File::File(const std::filesystem::path& fname,
           const std::filesystem::path& mode)
    : pimpl(std::make_unique<impl>(fname, mode)) {}
File::~File() = default;

int File::fileId() const {
#ifdef _WIN32
  return _fileno(pimpl->fp);
#else
#if defined(fileno)
  return fileno(pimpl->fp);
#else
  return ::fileno(pimpl->fp);
#endif
#endif
}

std::size_t File::size() const { return pimpl->size; }
std::size_t File::tell() const { return pimpl->tell(); }

void File::seek(const std::size_t offset, const int whence) const {
  pimpl->seek(offset, whence);
}
void File::readRaw(void* ptr, const std::size_t len) const {
  pimpl->readRaw(ptr, len);
}

std::uint32_t File::readU32() const { return pimpl->readU32(); }

void File::writeRaw(const void* ptr, const std::size_t len) const {
  pimpl->writeRaw(ptr, len);
}
void File::writeU32(const std::uint32_t val) const { pimpl->writeU32(val); }

// Mmap

struct Mmap::impl {
#if defined(_POSIX_MAPPED_FILES)
  static void alignRange(std::size_t* first,
                         std::size_t* last,
                         const std::size_t page_size) {
    std::size_t offset_in_page = *first & (page_size - 1);
    std::size_t offset_to_page =
        offset_in_page == 0 ? 0 : page_size - offset_in_page;
    *first += offset_to_page;

    *last = *last & ~(page_size - 1);

    if (*last <= *first) {
      *last = *first;
    }
  }
#endif

  impl(File* file, std::size_t prefetch, const bool numa) {
#ifdef _POSIX_MAPPED_FILES
    size = file->size();
    int fd = file->fileId();
    int flags = MAP_SHARED;
    if (numa) {
      prefetch = 0;
    }
#ifdef __linux__
    if (posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL)) {
      fmt::print(
          "warning: posix_fadvise(.., POSIX_FADV_SEQUENTIAL) failed: {}\n",
          strerror(errno));
    }
    if (prefetch) {
      flags |= MAP_POPULATE;
    }
#endif
    addr = mmap(NULL, file->size(), PROT_READ, flags, fd, 0);
    if (addr == MAP_FAILED) {
      throw std::runtime_error(fmt::format("mmap failed: {}", strerror(errno)));
    }

    if (prefetch > 0) {
      if (posix_madvise(addr, std::min(file->size(), prefetch),
                        POSIX_MADV_WILLNEED)) {
        fmt::print(
            "warning: posix_madvise(.., POSIX_MADV_WILLNEED) failed: {}\n",
            strerror(errno));
      }
    }
    if (numa) {
      if (posix_madvise(addr, file->size(), POSIX_MADV_RANDOM)) {
        fmt::print("warning: posix_madvise(.., POSIX_MADV_RANDOM) failed: {}\n",
                   strerror(errno));
      }
    }

    mapped_fragments.emplace_back(0, file->size());
#elif defined(_WIN32)
    void(numa);

    size = file->size();

    HANDLE hFile = (HANDLE)_get_osfhandle(file->fileId());

    HANDLE hMapping =
        CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);

    if (hMapping == NULL) {
      DWORD error = GetLastError();
      throw std::runtime_error(
          fmt::format("CreateFileMappingA failed: {}", win_err(error)));
    }

    addr = MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0);
    DWORD error = GetLastError();
    CloseHandle(hMapping);

    if (addr == NULL) {
      throw std::runtime_error(
          fmt::format("MapViewOfFile failed: {}", win_err(error)));
    }

    if (prefetch > 0) {
#if _WIN32_WINNT >= 0x602
      BOOL(WINAPI * pPrefetchVirtualMemory)(HANDLE, ULONG_PTR,
                                            PWIN32_MEMORY_RANGE_ENTRY, ULONG);
      HMODULE hKernel32 = GetModuleHandleW(L"kernel32.dll");

      pPrefetchVirtualMemory =
          (decltype(pPrefetchVirtualMemory))static_cast<void*>(
              GetProcAddress(hKernel32, "PrefetchVirtualMemory"));

      if (pPrefetchVirtualMemory) {
        WIN32_MEMORY_RANGE_ENTRY range;
        range.VirtualAddress = addr;
        range.NumberOfBytes = (SIZE_T)std::min(size, prefetch);
        if (!pPrefetchVirtualMemory(GetCurrentProcess(), 1, &range, 0)) {
          fmt::print("warning: PrefetchVirtualMemory failed: {}\n",
                     win_err(GetLastError()));
        }
      }
#else
      fmt::print(
          "skipping PrefetchVirtualMemory because _WIN32_WINNT < 0x602\n");
#endif
#else
    void(file);
    void(prefetch);
    void(numa);

    throw std::runtime_error("mmap not supported");
#endif
  }

  void unmapFragment(std::size_t first, std::size_t last) {
#if defined(_POSIX_MAPPED_FILES)
    int page_size = sysconf(_SC_PAGESIZE);
    alignRange(&first, &last, page_size);
    std::size_t len = last - first;

    if (len == 0) {
      return;
    }

    FMT_ASSERT(first % page_size == 0, "first is not page aligned");
    FMT_ASSERT(last % page_size == 0, "last is not page aligned");
    FMT_ASSERT(last > first, "last is not greater than first");

    void* next_page_start = static_cast<std::uint8_t*>(addr) + first;

    if (munmap(next_page_start, len)) {
      fmt::print("warning: munmap failed: {}\n", strerror(errno));
    }

    std::vector<std::pair<size_t, size_t>> new_mapped_fragments;
    for (const auto& frag : mapped_fragments) {
      if (frag.first < first && frag.second > last) {
        new_mapped_fragments.emplace_back(frag.first, first);
        new_mapped_fragments.emplace_back(last, frag.second);
      } else if (frag.first < first && frag.second > first) {
        new_mapped_fragments.emplace_back(frag.first, first);
      } else if (frag.first < last && frag.second > last) {
        new_mapped_fragments.emplace_back(last, frag.second);
      } else if (frag.first >= first && frag.second <= last) {
      } else {
        new_mapped_fragments.push_back(frag);
      }
    }
    mapped_fragments = std::move(new_mapped_fragments);
#elif defined(_WIN32)
      void(first);
      void(last);
#else
    void(first);
    void(last);

    throw std::runtime_error("mmap not supported");
#endif
  }

  ~impl() {
#if defined(_POSIX_MAPPED_FILES)
    for (const auto& frag : mapped_fragments) {
      if (munmap(static_cast<char*>(addr) + frag.first,
                 frag.second - frag.first)) {
        fmt::print("warning: munmap failed: {}\n", strerror(errno));
      }
    }
#elif defined(_WIN32)
      if (!UnmapViewOfFile(addr)) {
        fmt::print("warning: UnmapViewOfFile failed: {}\n",
                   win_err(GetLastError()));
      }
#endif
  }

#ifdef _POSIX_MAPPED_FILES
  std::vector<std::pair<std::size_t, std::size_t>> mapped_fragments;
#endif

  void* addr;
  std::size_t size;
};  // NOLINT

Mmap::Mmap(File* file, const std::size_t prefetch, const bool numa)
    : pimpl(std::make_unique<impl>(file, prefetch, numa)) {}
Mmap::~Mmap() = default;

std::size_t Mmap::size() const { return pimpl->size; }
void* Mmap::addr() const { return pimpl->addr; }
std::uint8_t* Mmap::data() const {
  return static_cast<std::uint8_t*>(pimpl->addr);
}

void Mmap::unmapFragment(const std::size_t first, const std::size_t last) {
  pimpl->unmapFragment(first, last);
}

#if defined(_POSIX_MEMLOCK_RANGE) || defined(_WIN32)
const bool Mmap::SUPPORTED = true;
#else
  const bool Mmap::SUPPORTED = false;
#endif

// Mlock

struct Mlock::impl {
  static std::size_t lockGranularity() {
#if defined(_POSIX_MAPPED_FILES)
    return static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
#elif defined(_WIN32)
      SYSTEM_INFO si;
      GetSystemInfo(&si);
      return static_cast<std::size_t>(si.dwPageSize);
#else
    return static_cast<std::size_t>(65536);
#endif
  }

  static void raw_unlock(void* addr, const std::size_t len) {
#if defined(_POSIX_MAPPED_FILES)
    if (munlock(addr, len)) {
      fmt::print("warning: failed to munlock buffer: {}\n",
                 std::strerror(errno));
    }
#elif defined(_WIN32)
      if (!VirtualUnlock(addr, len)) {
        fmt::print("warning: failed to VirtualUnlock buffer: {}\n",
                   llama_format_win_err(GetLastError()).c_str());
      }
#else
    void(addr);
    void(len);
    fmt::print("warning: munlock not supported on this system\n");
#endif
  }

  bool rawLock(const void* addr, const std::size_t len) const {
#if defined(_POSIX_MAPPED_FILES)
    if (!mlock(addr, len)) {
      return true;
    }

#ifdef __APPLE__
#define MLOCK_SUGGESTION                                              \
  "Try increasing the sysctl values 'vm.user_wire_limit' and "        \
  "'vm.global_user_wire_limit' and/or "                               \
  "decreasing 'vm.global_no_user_wire_amount'.  Also try increasing " \
  "RLIMIT_MEMLOCK (ulimit -l).\n"
#else
#define MLOCK_SUGGESTION \
  "Try increasing RLIMIT_MEMLOCK ('ulimit -l' as root).\n"
#endif

    char* errmsg = std::strerror(errno);
    bool suggest = (errno == ENOMEM);
#if defined(TARGET_OS_VISION) || defined(TARGET_OS_TV) || defined(_AIX)
    // visionOS/tvOS dont't support RLIMIT_MEMLOCK
    // Skip resource limit checks on visionOS/tvOS
    suggest = false;
#else
    struct rlimit lock_limit;
    if (suggest && getrlimit(RLIMIT_MEMLOCK, &lock_limit)) {
      suggest = false;
    }
    if (suggest && (lock_limit.rlim_max > lock_limit.rlim_cur + len)) {
      suggest = false;
    }
#endif

    fmt::print(
        "warning: failed to mlock {}-byte buffer (after previously locking "
        "{} bytes): {}\n{}",
        len, size, errmsg, suggest ? MLOCK_SUGGESTION : "");
    return false;
#elif defined(_WIN32)
      for (int tries = 1;; tries++) {
        if (VirtualLock(ptr, len)) {
          return true;
        }
        if (tries == 2) {
          fmt::print(
              "warning: failed to VirtualLock {}-byte buffer (after previously "
              "locking {} bytes): {}\n",
              len, size, llama_format_win_err(GetLastError()).c_str());
          return false;
        }

        SIZE_T min_ws_size, max_ws_size;
        if (!GetProcessWorkingSetSize(GetCurrentProcess(), &min_ws_size,
                                      &max_ws_size)) {
          fmt::print("warning: GetProcessWorkingSetSize failed: {}\n",
                     llama_format_win_err(GetLastError()).c_str());
          return false;
        }

        SIZE_T min_ws_size, max_ws_size;
        if (!GetProcessWorkingSetSize(GetCurrentProcess(), &min_ws_size,
                                      &max_ws_size)) {
          fmt::print("warning: GetProcessWorkingSetSize failed: {}\n",
                     llama_format_win_err(GetLastError()));
          return false;
        }
        std::size_t increment = len + 1048576;
        min_ws_size += increment;
        max_ws_size += increment;
        if (!SetProcessWorkingSetSize(GetCurrentProcess(), min_ws_size,
                                      max_ws_size)) {
          fmt::print("warning: SetProcessWorkingSetSize failed: {}\n",
                     llama_format_win_err(GetLastError()));
          return false;
        }
      }
#else
    void(addr);
    void(len);
    fmt::print("warning: mlock not supported on this system\n");
    return false;
#endif
  }

  impl() : addr(NULL), size(0), failed_already(false) {}

  void init(void* ptr) {
    FMT_ASSERT(addr == NULL && size == 0, "Memory region already initialized");
    addr = ptr;
  }

  void growTo(std::size_t target_size) {
    FMT_ASSERT(addr, "Memory region not initialized");
    if (failed_already) {
      return;
    }
    std::size_t granularity = lockGranularity();
    target_size = (target_size + granularity - 1) & ~(granularity - 1);
    if (target_size > size) {
      if (rawLock(static_cast<std::uint8_t*>(addr) + size,
                  target_size - size)) {
        size = target_size;
      } else {
        failed_already = true;
      }
    }
  }

  void* addr;
  std::size_t size;

  bool failed_already;
};

Mlock::Mlock() : pimpl(std::make_unique<impl>()) {}
Mlock::~Mlock() = default;

void Mlock::init(void* ptr) { pimpl->init(ptr); }
void Mlock::growTo(const std::size_t target_size) {
  pimpl->growTo(target_size);
}

#if defined(_POSIX_MEMLOCK_RANGE) || defined(_WIN32)
const bool Mlock::SUPPORTED = true;
#else
  const bool Mlock::SUPPORTED = false;
#endif

std::size_t path_max() { return PATH_MAX; }

}  // namespace safetensors
