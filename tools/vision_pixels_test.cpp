// CPU-only checks of the storage wrapped by the Qwen vision encoder. No model or Metal device is needed.
#include "vision/encoder.h"
#include "vision/qwen_vit.h"

#include <algorithm>
#include <cstdio>
#include <limits>
#include <utility>

using namespace ni::vision;

namespace {
int fails = 0;
void check(bool ok, const char* what) {
  printf("%s %s\n", ok ? "ok  " : "FAIL", what);
  fails += !ok;
}
}

int main() {
  const size_t page = pixel_page_size();
  bool aligned = true, bounded = true, copied = true, moved = true;
  for (size_t n : {size_t(1), size_t(16), page - 16, page - 1, page, page + 1, size_t(1376 * 864 * 3)}) {
    Prepared p;
    p.pixels.resize(n);
    aligned &= uintptr_t(p.pixels.data()) % page == 0;
    const size_t bytes = pixel_buffer_bytes(n);
    bounded &= bytes % page == 0 && bytes >= n + 16 && bytes < n + 16 + page;
    // Touch the promised tail, including the last mapped byte (also checked under AddressSanitizer).
    auto* storage = p.pixels.data();
    std::fill(storage, storage + bytes, uint8_t(0x5a));
    Prepared copy = p;
    copied &= copy.pixels.data() != storage && copy.pixels == p.pixels;
    copy.pixels[0] = 0;
    copied &= p.pixels[0] == 0x5a;
    Prepared next = std::move(p);
    moved &= next.pixels.data() == storage && next.pixels.size() == n;
    PixelBuffer().swap(next.pixels);
    moved &= next.pixels.empty() && next.pixels.capacity() == 0;
  }
  check(aligned, "pixel addresses are page aligned, including non-page-sized images");
  check(bounded, "mapped lengths cover the image and padding, without another full page");
  check(copied, "Prepared copies own their pixels independently");
  check(moved, "Prepared moves keep the address; metadata-only reset releases storage");

  PixelBuffer grown(3, 17);
  grown.reserve(page + 1);
  check(uintptr_t(grown.data()) % page == 0 && grown[0] == 17 && grown[2] == 17,
        "vector reallocation preserves alignment and contents");
  bool overflow = false;
  try {
    (void)pixel_buffer_bytes(std::numeric_limits<size_t>::max());
  } catch (const std::bad_array_new_length&) {
    overflow = true;
  }
  check(overflow, "page rounding rejects size overflow");

  // Identity, one-axis, two-pass, and Pillow's vertical-first tall-image branch all return wrappable storage.
  struct Shape { int w, h, ow, oh; };
  bool resize_ok = true;
  for (const auto s : {Shape{32, 32, 32, 32}, Shape{32, 32, 64, 32}, Shape{32, 32, 32, 64},
                       Shape{37, 53, 32, 64}, Shape{2, 300, 32, 64}}) {
    const std::vector<uint8_t> rgb(size_t(s.w) * s.h * 3, 73);
    const auto out = pil_resize_bicubic(rgb, s.w, s.h, s.ow, s.oh);
    resize_ok &= uintptr_t(out.data()) % page == 0 && out.size() == size_t(s.ow) * s.oh * 3;
    resize_ok &= std::all_of(out.begin(), out.end(), [](uint8_t v) { return v == 73; });
  }
  check(resize_ok, "every bicubic resize branch writes directly into aligned pixel storage");
  return fails ? 1 : 0;
}
