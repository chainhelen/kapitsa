#include "common.h"

// Experimental group-64 affine matvec for one or two live rows. K must be a multiple of 64,
// and x_ld a multiple of four floats. Two simdgroups compute four output channels each.
// Inputs retain X8's half rounding, but the dot product removes MMA's 1024-offset cancellation
// and changes the reduction order. This is not bitwise equivalent to qmm8g; test model outputs.
// NT=2 shares each decoded weight across two token rows.
template <int F, int NT = 1>
NI_INLINE void qmvgrc_impl(const device uint32_t* W, const device bfloat* S, const device bfloat* B,
                         const device float* X, device float* Y, constant QmvrParams& p, uint2 tg,
                         uint sg, uint lane) {
  const uint t0 = tg.x * NT, n0 = (tg.y * 2 + sg) * 4, wpr = p.K * F / 32, gpr = p.K / 64;
  uint rows[4];
  float acc[4][NT];
  for (int n = 0; n < 4; ++n) for (int t = 0; t < NT; ++t) acc[n][t] = 0.0f;
  for (int n = 0; n < 4; ++n) rows[n] = min(n0 + n, p.N - 1);
  for (uint k0 = lane * 16; k0 < p.K; k0 += 512) {
    float x[NT][16], sx[NT];
    for (int t = 0; t < NT; ++t) {
      sx[t] = 0.0f;
      for (int i = 0; i < 4; ++i) {
        const float4 v = float4(half4(*(const device float4*)(X + min(t0 + t, p.R - 1) * p.x_ld + k0 + 4 * i)));
        x[t][4 * i] = v.x; x[t][4 * i + 1] = v.y; x[t][4 * i + 2] = v.z; x[t][4 * i + 3] = v.w;
        sx[t] += v.x + v.y + v.z + v.w;
      }
    }
    for (int n = 0; n < 4; ++n) {
      uint code[16];
      wf_codes16<F>(W + rows[n] * wpr, k0, code);
      float dot[NT];
      for (int t = 0; t < NT; ++t) dot[t] = 0.0f;
      for (int i = 0; i < 16; ++i)
        for (int t = 0; t < NT; ++t) dot[t] = fma(float(code[i]), x[t][i], dot[t]);
      const uint gi = rows[n] * gpr + k0 / 64;
      const float scale = float(S[gi]), bias = float(B[gi]);
      for (int t = 0; t < NT; ++t) acc[n][t] += scale * dot[t] + bias * sx[t];
    }
  }
  for (int n = 0; n < 4; ++n)
    for (int t = 0; t < NT; ++t) {
      const float v = simd_sum(acc[n][t]);
      if (lane == 0 && n0 + n < p.N && t0 + t < p.R) {
        device float* dst = Y + (t0 + t) * p.y_ld + n0 + n;
        *dst = p.add ? *dst + v : v;
      }
    }
}

#define NI_QMVGRC(NAME, F, NT)                                                                              \
  [[kernel]] void NAME(const device uint32_t* W [[buffer(0)]], const device bfloat* S [[buffer(1)]],           \
                       const device bfloat* B [[buffer(2)]], const device float* X [[buffer(3)]],             \
                       device float* Y [[buffer(4)]], constant QmvrParams& p [[buffer(5)]],                    \
                       uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], \
                       uint lane [[thread_index_in_simdgroup]]) {                                           \
    qmvgrc_impl<F, NT>(W, S, B, X, Y, p, tg, sg, lane);                                                      \
  }

#if defined(NI_WF_INST)
NI_QMVGRC(qmvgrc_f5g64, 5, 1)
NI_QMVGRC(qmvgrc_f6g64, 6, 1)
NI_QMVGRC(qmvgrc_f5g64_t2, 5, 2)
NI_QMVGRC(qmvgrc_f6g64_t2, 6, 2)
#endif
