// --gpu [--small]: direct one/two-row affine matvec vs qmm8g, fp64 references and output/stride guards.
// The direct path rounds inputs to half but changes the reduction: bitwise differences are reported,
// not accepted as evidence of model correctness. Use affine_model_check and real continuations too.
// --bench [N K bits]: interleaved timings over rotating weights, with and without X8 preparation.
// Takes the GPU lock. Run --gpu --small with MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1 first.
#include "bench/bench.h"
#include "metal/gpu.h"
#include "model/kparams.h"
#include "util/bf16.h"
#include "util/log.h"

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

using namespace ni;

namespace {
constexpr uint32_t guard = 16;
constexpr float sentinel = 12345.0f;
constexpr const char* usage = "affine_rows_test --gpu [--small] | --bench [N K bits]";

uint32_t positive(const char* arg) {
  uint32_t value = 0;
  const char* end = arg + strlen(arg);
  const auto parsed = std::from_chars(arg, end, value);
  if (parsed.ec != std::errc{} || parsed.ptr != end || !value) fatal("invalid positive integer: %s", arg);
  return value;
}

struct Matrix {
  Buffer w, s, b;
};

struct Case {
  Gpu& gpu = Gpu::get();
  uint32_t N, K, R, bits, nf, ks, cs, xld, yld;
  bool add;
  Buffer x, table, sums, storage[2], y[2];
  Pipeline prep, pipes[2];
  std::vector<Matrix> matrices;

  Case(uint32_t n, uint32_t k, uint32_t r, uint32_t f, uint32_t fragments, uint32_t slices,
       bool residual, bool stride, unsigned copies = 1)
      : N(n), K(k), R(r), bits(f), nf(fragments), ks(slices), cs(8 / slices),
        xld(k + (stride ? 16 : 0)), yld(n + (stride ? 11 : 0)), add(residual) {
    x = gpu.alloc(size_t(R) * xld * 4);
    table = gpu.alloc(size_t(K) * 8 * 2);
    sums = gpu.alloc(size_t(K / 64) * 8 * 4);
    std::mt19937 rng(42 + n + k + r + f);
    std::uniform_real_distribution<float> uniform(-1, 1);
    std::fill_n(x.as<float>(), size_t(R) * xld, std::numeric_limits<float>::quiet_NaN());
    for (uint32_t t = 0; t < R; ++t)
      for (uint32_t i = 0; i < K; ++i) x.as<float>()[t * xld + i] = uniform(rng);
    for (unsigned c = 0; c < copies; ++c) {
      Matrix m{gpu.alloc(size_t(N) * K * bits / 8 + 4, "weights", false),
               gpu.alloc(size_t(N) * (K / 64) * 2, "scales", false),
               gpu.alloc(size_t(N) * (K / 64) * 2, "biases", false)};
      for (size_t i = 0; i < m.w.size / 4; ++i) m.w.as<uint32_t>()[i] = rng();
      for (size_t i = 0; i < m.s.size / 2; ++i) {
        const float scale = 0.002f + 0.02f * std::fabs(uniform(rng));
        m.s.as<uint16_t>()[i] = f32_to_bf16(scale);
        m.b.as<uint16_t>()[i] = f32_to_bf16(-scale * float(1u << (bits - 1)) + 0.01f * uniform(rng));
      }
      matrices.push_back(m);
    }
    for (unsigned v = 0; v < 2; ++v) {
      storage[v] = gpu.alloc((size_t(R) * yld + 2 * guard) * 4);
      y[v] = storage[v].view(guard * 4, size_t(R) * yld * 4);
      reset(v);
      pipes[v] = gpu.pipeline(v ? "qmvgrc_f" + std::to_string(bits) + "g64" + (R == 2 ? "_t2" : "") :
                                 "qmm8g_f" + std::to_string(bits) + "g64_nf" + std::to_string(nf));
    }
    prep = gpu.pipeline(K % 256 == 0 ? "x8_prep2" : "x8_prep");
  }

  ~Case() {
    for (Matrix& m : matrices) for (Buffer* p : {&m.w, &m.s, &m.b}) gpu.release(*p);
    for (Buffer* p : {&x, &table, &sums, &storage[0], &storage[1]}) gpu.release(*p);
  }

  void reset(unsigned v) {
    std::fill_n(storage[v].as<float>(), storage[v].size / 4, sentinel);
    for (uint32_t t = 0; t < R; ++t)
      for (uint32_t n = 0; n < N; ++n) y[v].as<float>()[t * yld + n] = add ? float(int(n % 17) - 8) / 8 : 0;
  }

  void encode_prep(Encoder& e, unsigned v) {
    if (v) return;
    const kp::X8Prep p{K, R, xld, 0, 0, 1e-6f, K, 0, 0, 0};
    e.use(prep);
    e.buf(0, x); e.buf(1, x); e.buf(2, table);
    e.buf(3, sums); e.buf(4, x); e.val(5, p);
    e.buf(6, table); e.buf(7, sums); e.buf(8, table);
    e.dispatch(K % 256 == 0 ? K / 256 : 8u, 1, 1, 256);
  }

  void encode_mm(Encoder& e, unsigned v, const Matrix& m) {
    if (v) {
      const kp::Qmvr p{N, K, R, xld, yld, add ? 1u : 0u};
      e.use(pipes[v]);
      e.buf(0, m.w); e.buf(1, m.s); e.buf(2, m.b); e.buf(3, x); e.buf(4, y[v]); e.val(5, p);
      e.dispatch(1, (N + 7) / 8, 1, 64);
      return;
    }
    kp::Qmm8 p{};
    p.K = K; p.R = R; p.nseg = 1; p.ks = ks;
    p.N[0] = N; p.y_ld[0] = yld; p.add[0] = add;
    p.tg_begin[1] = (N + 8 * nf * cs - 1) / (8 * nf * cs);
    e.use(pipes[v]);
    for (int i = 0; i < 4; ++i) {
      e.buf(4 * i, m.w); e.buf(4 * i + 1, m.s); e.buf(4 * i + 2, m.b); e.buf(4 * i + 3, y[v]);
    }
    e.buf(16, table); e.buf(17, sums); e.val(18, p);
    e.dispatch(p.tg_begin[1], 1, 1, 32 * cs * ks);
  }

  double run(unsigned v, bool with_prep = true, unsigned repeats = 1) {
    auto e = gpu.begin();
    for (unsigned repeat = 0; repeat < repeats; ++repeat)
      for (const Matrix& m : matrices) {
        if (with_prep) encode_prep(*e, v);
        encode_mm(*e, v, m);
      }
    e->commit_and_wait();
    return e->gpu_ms() * 1000 / (repeats * matrices.size());
  }

  bool check() {
    reset(0); reset(1);
    run(0); run(1);
    size_t different = 0, nonfinite = 0, bad_guards = 0, reference_failures[2] = {};
    double max_error = 0, squared_error[2] = {}, squared_reference = 0;
    for (unsigned v = 0; v < 2; ++v) {
      for (uint32_t i = 0; i < guard; ++i) {
        bad_guards += storage[v].as<float>()[i] != sentinel;
        bad_guards += storage[v].as<float>()[guard + size_t(R) * yld + i] != sentinel;
      }
      for (uint32_t t = 0; t < R; ++t)
        for (uint32_t n = N; n < yld; ++n) bad_guards += y[v].as<float>()[t * yld + n] != sentinel;
    }
    const Matrix& m = matrices.back();
    for (uint32_t t = 0; t < R; ++t)
      for (uint32_t n = 0; n < N; ++n) {
        const size_t i = size_t(t) * yld + n;
        const float a = y[0].as<float>()[i], b = y[1].as<float>()[i];
        different += std::bit_cast<uint32_t>(a) != std::bit_cast<uint32_t>(b);
        nonfinite += !std::isfinite(a) || !std::isfinite(b);
        if (N > 64 && n % (N / 31) != 0) continue;
        double ref = add ? double(int(n % 17) - 8) / 8 : 0;
        double magnitude[2] = {std::fabs(ref), std::fabs(ref)};
        for (uint32_t k = 0; k < K; ++k) {
          const size_t bit = size_t(n) * K * bits + k * bits;
          uint32_t code = m.w.as<uint32_t>()[bit / 32] >> (bit % 32);
          if (bit % 32 + bits > 32) code |= m.w.as<uint32_t>()[bit / 32 + 1] << (32 - bit % 32);
          code &= (1u << bits) - 1;
          const size_t gi = size_t(n) * (K / 64) + k / 64;
          const double s = bf16_to_f32(m.s.as<uint16_t>()[gi]), bias = bf16_to_f32(m.b.as<uint16_t>()[gi]);
          const double input = float(__fp16(x.as<float>()[t * xld + k]));
          ref += (s * code + bias) * input;
          // Each bound follows its path's products, including the shipped path's cancellation.
          magnitude[0] += (std::fabs(s * (1024 + code)) + std::fabs(bias - 1024 * s)) * std::fabs(input);
          magnitude[1] += (std::fabs(s * code) + std::fabs(bias)) * std::fabs(input);
        }
        const double error = std::fabs(double(b) - ref);
        squared_error[0] += (double(a) - ref) * (double(a) - ref);
        squared_error[1] += (double(b) - ref) * (double(b) - ref);
        squared_reference += ref * ref;
        max_error = std::max(max_error, error);
        reference_failures[0] += std::fabs(double(a) - ref) > 64 * std::numeric_limits<float>::epsilon() * magnitude[0] + 1e-6;
        reference_failures[1] += error > 64 * std::numeric_limits<float>::epsilon() * magnitude[1] + 1e-6;
      }
    printf("check N=%u K=%u R=%u bits=%u nf=%u ks=%u add=%d stride=%d: different=%zu nonfinite=%zu "
           "guards=%zu fp64_failures_A=%zu fp64_failures_B=%zu max_abs_B=%.6g relative_l2_A=%.6g relative_l2_B=%.6g\n",
           N, K, R, bits, nf, ks, int(add), int(xld != K),
           different, nonfinite, bad_guards, reference_failures[0], reference_failures[1], max_error,
           std::sqrt(squared_error[0] / std::max(1e-30, squared_reference)),
           std::sqrt(squared_error[1] / std::max(1e-30, squared_reference)));
    return !nonfinite && !bad_guards && !reference_failures[0] && !reference_failures[1];
  }
};
}  // namespace

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IOLBF, 0);
  if (argc == 2 && !strcmp(argv[1], "--help")) { puts(usage); return 0; }
  const bool bench = argc >= 2 && !strcmp(argv[1], "--bench");
  const bool small = argc == 3 && !strcmp(argv[2], "--small");
  if ((!bench && (argc < 2 || strcmp(argv[1], "--gpu") || (argc != 2 && !small))) ||
      (bench && argc != 2 && argc != 5)) fatal("usage: %s", usage);
  const uint32_t N = bench && argc == 5 ? positive(argv[2]) : 8192;
  const uint32_t K = bench && argc == 5 ? positive(argv[3]) : 2048;
  const uint32_t bits = bench && argc == 5 ? positive(argv[4]) : 5;
  if (K % 64 || K > 65536 || N > 65536 || (bits != 5 && bits != 6)) fatal("unsupported benchmark shape");
  const size_t matrix_bytes = size_t(N) * K * bits / 8 + size_t(N) * (K / 64) * 4 + 4;
  if (matrix_bytes > (256u << 20)) fatal("benchmark matrix exceeds the 256 MiB allocation limit");
  ni::bench::GpuLock lock;
  if (!ni::bench::GpuLock::held_by_caller() && !lock.acquire(nullptr, 0)) fatal("GPU lock is busy");
  Gpu& gpu = Gpu::get();
  gpu.compile();
  printf("commit=%s GPU=%s cores=%d thermal=%s\n", KAPITSA_GIT, gpu.name().c_str(), gpu.core_count(), Gpu::thermal_state());
  if (!bench) {
    bool ok = true;
    for (uint32_t bits : {5u, 6u})
      for (uint32_t nf : {1u, 2u})
        for (uint32_t ks : {1u, 2u, 4u, 8u})
          for (uint32_t r : {1u, 2u}) {
            Case test(19, small ? 192 : 768, r, bits, nf, ks, ks % 4 == 0, true);
            ok = test.check() && ok;
          }
    for (uint32_t bits : {5u, 6u})
      for (uint32_t n : {1u, 7u, 8u})
        for (uint32_t r : {1u, 2u}) {
          Case test(n, 64, r, bits, 1, 1, true, true);
          ok = test.check() && ok;
        }
    if (!small)
      for (uint32_t bits : {5u, 6u})
        for (uint32_t r : {1u, 2u})
          for (bool output : {false, true}) {
            Case test(output ? 2048 : 8192, output ? 4096 : 2048, r, bits, 1, output ? 8 : 4, false, false);
            ok = test.check() && ok;
          }
    return ok ? 0 : 1;
  }
  const unsigned copies = unsigned(std::clamp<size_t>((128u << 20) / matrix_bytes, 1, 128));
  const uint32_t ks = K >= 2048 && ((N + 7) / 8) * 4 < 64u * uint32_t(gpu.core_count()) ? 8 : 4;
  for (uint32_t R : {1u, 2u}) {
    Case test(N, K, R, bits, 1, ks, false, false, copies);
    if (!test.check()) return 1;
    for (bool prep : {false, true}) {
      const double until = now_ms() + 300;
      do { test.run(0, prep); test.run(1, prep); } while (now_ms() < until);
      std::array<std::vector<double>, 2> times;
      for (unsigned pair = 0; pair < 12; ++pair)
        for (unsigned arm = 0; arm < 2; ++arm) {
          const unsigned v = arm ^ (pair & 1);
          const double us = test.run(v, prep);
          times[v].push_back(us);
          printf("run=%u arm=%c prep=%d N=%u K=%u R=%u bits=%u ks=%u copies=%u us=%.6f\n",
                 pair + 1, 'A' + v, int(prep), N, K, R, bits, ks, copies, us);
        }
      for (auto& ts : times) std::sort(ts.begin(), ts.end());
      const double a = (times[0][5] + times[0][6]) / 2, b = (times[1][5] + times[1][6]) / 2;
      printf("summary prep=%d N=%u K=%u R=%u bits=%u A=%.6f B=%.6f time_reduction=%.3f%% thermal=%s\n",
             int(prep), N, K, R, bits, a, b, 100 * (1 - b / a), Gpu::thermal_state());
    }
  }
}
