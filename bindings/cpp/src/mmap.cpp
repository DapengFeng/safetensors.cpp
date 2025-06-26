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

#include "safetensors/mmap.hpp"

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace safetensors {

#if defined(_WIN32)
static std::string win_err(DWORD err) {
  LPSTR buf;
  size_t size = FormatMessageA(
      FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
          FORMAT_MESSAGE_IGNORE_INSERTS,
      NULL, err, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPSTR)&buf, 0,
      NULL);
  if (!size) {
    return "FormatMessageA failed";
  }
  std::string ret(buf, size);
  LocalFree(buf);
  return ret;
}
#endif

struct File::FileImpl {
  FILE* file_ptr_ = nullptr;
  std::size_t size_ = 0;

  explicit FileImpl(FILE* file_ptr) : file_ptr_(file_ptr) {}

  ~FileImpl() {
    if (file_ptr_) {
      std::fclose(file_ptr_);
    }
    file_ptr_ = nullptr;
  }

  inline std::size_t tell() const {
    auto ret = std::ftell(file_ptr_);
    if (ret == -1) {
      throw std::runtime_error(fmt::format("{}:{} tell error: {}", __FILE__,
                                           __LINE__, std::strerror(errno)));
    }
    return static_cast<std::size_t>(ret);
  }

  inline void seek(const std::size_t offset, const int whence) const {
    if (std::fseek(file_ptr_, offset, whence) != 0) {
      throw std::runtime_error(fmt::format("{}:{} seek error: {}", __FILE__,
                                           __LINE__, std::strerror(errno)));
    }
    return;
  }

  inline void readRaw(void* ptr, const std::size_t len) const {
    if (len == 0) return;

    errno = 0;
    std::size_t ret = std::fread(ptr, len, 1, file_ptr_);

    if (std::ferror(file_ptr_)) {
      throw std::runtime_error(fmt::format("{}:{} read error: {}", __FILE__,
                                           __LINE__, std::strerror(errno)));
    }

    if (ret != 1) {
      throw std::runtime_error(
          fmt::format("{}:{} unexpectedly reached end of file: {}", __FILE__,
                      __LINE__, std::strerror(errno)));
    }
    return;
  }
  inline std::uint32_t readU32() const {
    std::uint32_t ret;
    readRaw(&ret, sizeof(ret));
    return ret;
  }

  inline void writeRaw(const void* ptr, const std::size_t len) const {
    if (len == 0) return;

    errno = 0;
    std::size_t ret = std::fwrite(ptr, len, 1, file_ptr_);

    if (ret != 1) {
      throw std::runtime_error(fmt::format("{}:{} write error: {}", __FILE__,
                                           __LINE__, std::strerror(errno)));
    }
    return;
  }
  inline void writeU32(const std::uint32_t val) const {
    writeRaw(&val, sizeof(val));
  }
};

// File class method definitions
int File::fileId() const {
  return fileno(impl_ptr_->file_ptr_);
}

std::size_t File::size() const {
  return impl_ptr_->size_;
}

std::size_t File::tell() const {
  return impl_ptr_->tell();
}

void File::seek(const std::size_t offset, const int whence) const {
  impl_ptr_->seek(offset, whence);
}

void File::readRaw(void* ptr, const std::size_t len) const {
  impl_ptr_->readRaw(ptr, len);
}

std::uint32_t File::readU32() const {
  return impl_ptr_->readU32();
}

void File::writeRaw(const void* ptr, const std::size_t len) const {
  impl_ptr_->writeRaw(ptr, len);
}

void File::writeU32(const std::uint32_t val) const {
  impl_ptr_->writeU32(val);
}

struct Mmap::MmapImpl {
#if defined(_POSIX_MAPPED_FILES)
  inline static void align_range(size_t* first,
                                 size_t* last,
                                 size_t page_size) {
    size_t offset_in_page = *first & (page_size - 1);
    size_t offset_to_page =
        offset_in_page == 0 ? 0 : page_size - offset_in_page;
    *first += offset_to_page;

    *last = *last & ~(page_size - 1);

    if (*last <= *first) {
      *last = *first;
    }
  }
#endif

  void* addr_ptr_ = nullptr;
  std::size_t size_ = 0;

#if defined(_POSIX_MAPPED_FILES)
  std::vector<std::pair<size_t, size_t>> mapped_fragments_;
#endif

  explicit MmapImpl(void* addr_ptr, std::size_t size)
      : addr_ptr_(addr_ptr), size_(size) {
#if defined(_POSIX_MAPPED_FILES)
    mapped_fragments_.emplace_back(0, size);
#endif
  }

  ~MmapImpl() {
#if defined(_POSIX_MAPPED_FILES)
    for (const auto& frag : mapped_fragments_) {
      if (munmap(static_cast<uint8_t*>(addr_ptr_) + frag.first, frag.second - frag.first)) {
        fmt::print("{}:{} munmap failed: {}", __FILE__, __LINE__,
                   std::strerror(errno));
      }
    }
#elif defined(_WIN32)
    if (!UnmapViewOfFile(addr_ptr_)) {
      fmt::print("{}:{} UnmapViewOfFile failed: {}", __FILE__, __LINE__,
                 std::strerror(errno));
    }
#endif
    addr_ptr_ = nullptr;
  }

  void unmapFragment(size_t first, size_t last) {
#if defined(_POSIX_MAPPED_FILES)
    int page_size = sysconf(_SC_PAGESIZE);
    align_range(&first, &last, page_size);
    size_t len = last - first;

    if (len == 0) {
      return;
    }

    if (first % page_size != 0 || last % page_size != 0 || first >= last) {
      throw std::runtime_error(
          fmt::format("{}:{} Invalid range for unmapping: {}", __FILE__,
                      __LINE__, std::strerror(errno)));
    }

    void* next_page_start = static_cast<uint8_t*>(addr_ptr_) + first;

    if (munmap(next_page_start, len)) {
      fmt::print("{}:{} munmap failed: {}", __FILE__, __LINE__,
                 std::strerror(errno));
    }

    std::vector<std::pair<size_t, size_t>> new_mapped_fragments;
    for (const auto& frag : mapped_fragments_) {
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
    mapped_fragments_ = std::move(new_mapped_fragments);
#elif defined(_WIN32)
    (void)first;
    (void)last;
    return;
#endif
  }
};

// Mmap class method definitions
std::size_t Mmap::size() const {
  return impl_ptr_->size_;
}

std::uint8_t* Mmap::data() const {
  return static_cast<std::uint8_t*>(impl_ptr_->addr_ptr_);
}

void Mmap::unmapFragment(const std::size_t first, const std::size_t last) {
  impl_ptr_->unmapFragment(first, last);
}

struct Mlock::MlockImpl {
  inline static std::size_t lockGranularity() {
#if defined(_POSIX_MEMLOCK_RANGE)
    return static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
#elif defined(_WIN32)
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return std::static_cast<std::size_t>(si.dwPageSize);
#else
    return std::static_cast<std::size_t>(65536);
#endif
  }

  inline static void rawUnlock(void* addr_ptr, const std::size_t len) {
#if defined(_POSIX_MEMLOCK_RANGE)
    if (munlock(addr_ptr, len)) {
      fmt::print("{}:{} warning: failed to munlock buffer: {}\n", __FILE__,
                 __LINE__, std::strerror(errno));
    }
#elif defined(_WIN32)
    if (!VirtualUnlock(ptr, len)) {
      fmt::print("{}:{} warning: failed to VirtualUnlock buffer: {}\n",
                 __FILE__, __LINE__, win_err(GetLastError()));
    }
#else
#warning "munlock not supported on this platform"
#endif
  }

  explicit MlockImpl(void* addr_ptr)
      : addr_ptr_(addr_ptr), size_(0), failed_already_(false) {}

  ~MlockImpl() { addr_ptr_ = nullptr; }

  inline bool rawLock(void* ptr, std::size_t len) {
#if defined(_POSIX_MEMLOCK_RANGE)
    if (!mlock(addr_ptr_, len)) {
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
    rlimit lock_limit;
    if (suggest && getrlimit(RLIMIT_MEMLOCK, &lock_limit)) {
      suggest = false;
    }
    if (suggest && (lock_limit.rlim_max > lock_limit.rlim_cur + len)) {
      suggest = false;
    }
#endif
    fmt::print(
        "{}:{} warning: failed to mlock {}-byte buffer (after previously "
        "locking "
        "%{} bytes): {}\n{}",
        __FILE__, __LINE__, len, size_, errmsg,
        suggest ? MLOCK_SUGGESTION : "");
    return false;
#elif defined(_WIN32)
    for (int tries = 1;; tries++) {
      if (VirtualLock(ptr, len)) {
        return true;
      }
      if (tries == 2) {
        fmt::print(
            "{}:{} warning: failed to VirtualLock {}-byte buffer (after "
            "previously locking {} bytes): {}\n",
            __FILE__, __LINE__, len, size_, win_err(GetLastError()));
        return false;
      }

      SIZE_T min_ws_size, max_ws_size;
      if (!GetProcessWorkingSetSize(GetCurrentProcess(), &min_ws_size,
                                    &max_ws_size)) {
        fmt::print("{}:{} warning: GetProcessWorkingSetSize failed: {}\n",
                   __FILE__, __LINE__, win_err(GetLastError()));
        return false;
      }
      size_t increment = len + 1048576;
      min_ws_size += increment;
      max_ws_size += increment;
      if (!SetProcessWorkingSetSize(GetCurrentProcess(), min_ws_size,
                                    max_ws_size)) {
        fmt::print("{}:{} warning: SetProcessWorkingSetSize failed: {}\n",
                   __FILE__, __LINE__, win_err(GetLastError()));
        return false;
      }
    }
#else
#warning "mlock not supported on this system"
    return false;
#endif
  }

  inline void growTo(std::size_t target_size) {
    if (failed_already_) {
      return;
    }
    const std::size_t granularity = lockGranularity();
    target_size = (target_size + granularity - 1) & ~(granularity - 1);
    if (target_size > size_) {
      if (rawLock(static_cast<uint8_t*>(addr_ptr_) + size_,
                  target_size - size_)) {
        size_ = target_size;
      } else {
        rawUnlock(addr_ptr_, size_);
        failed_already_ = true;
      }
    }
  }

  void* addr_ptr_;
  std::size_t size_;
  bool failed_already_;
};

// Mlock class method definitions
void Mlock::growTo(const std::size_t target_size) {
  impl_ptr_->growTo(target_size);
}

}  // namespace safetensors
