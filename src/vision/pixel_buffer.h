// CPU-owned pixels that Metal can wrap without copying. Preparation stays CPU-only; the caller keeps the pixels
// alive until encode() has waited for its command buffers and released the Metal wrapper.
#pragma once
#include <unistd.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <new>
#include <vector>

namespace ni::vision {

inline size_t pixel_page_size() {
  static const size_t page = size_t(getpagesize());
  return page;
}

// newBufferWithBytesNoCopy requires a page-aligned pointer and length. Keep the vision scratch buffers' 16-byte tail.
inline size_t pixel_buffer_bytes(size_t bytes) {
  const size_t page = pixel_page_size();
  if (bytes > std::numeric_limits<size_t>::max() - 16 - (page - 1)) throw std::bad_array_new_length();
  return (bytes + 16 + page - 1) / page * page;
}

template <class T>
struct PixelAllocator {
  using value_type = T;
  PixelAllocator() = default;
  template <class U> PixelAllocator(const PixelAllocator<U>&) noexcept {}

  T* allocate(size_t n) {
    if (n > std::numeric_limits<size_t>::max() / sizeof(T)) throw std::bad_array_new_length();
    void* p = nullptr;
    if (posix_memalign(&p, pixel_page_size(), pixel_buffer_bytes(n * sizeof(T)))) throw std::bad_alloc();
    return static_cast<T*>(p);
  }
  void deallocate(T* p, size_t) noexcept { std::free(p); }
  template <class U> bool operator==(const PixelAllocator<U>&) const noexcept { return true; }
};

using PixelBuffer = std::vector<uint8_t, PixelAllocator<uint8_t>>;

}  // namespace ni::vision
