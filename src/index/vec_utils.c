#include "vec_utils.h"
#include <math.h>

#if defined(__AVX512F__) || defined(__AVX2__) || defined(__SSE__) || defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif
#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#endif

float vec_l2_distance_sq(const float *a, const float *b, uint32_t dim) {
  uint32_t i = 0;
#if defined(__AVX512F__)
  __m512 sum_vec = _mm512_setzero_ps();
  for (; i + 16 <= dim; i += 16) {
    __m512 va = _mm512_loadu_ps(a + i);
    __m512 vb = _mm512_loadu_ps(b + i);
    __m512 diff = _mm512_sub_ps(va, vb);
    sum_vec = _mm512_add_ps(sum_vec, _mm512_mul_ps(diff, diff));
  }
  float sum = _mm512_reduce_add_ps(sum_vec);
#elif defined(__AVX2__)
  __m256 sum_vec = _mm256_setzero_ps();
  for (; i + 8 <= dim; i += 8) {
    __m256 va = _mm256_loadu_ps(a + i);
    __m256 vb = _mm256_loadu_ps(b + i);
    __m256 diff = _mm256_sub_ps(va, vb);
    sum_vec = _mm256_add_ps(sum_vec, _mm256_mul_ps(diff, diff));
  }
  float tmp[8];
  _mm256_storeu_ps(tmp, sum_vec);
  float sum = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
#elif defined(__ARM_NEON)
  float32x4_t sum_vec = vdupq_n_f32(0.0f);
  for (; i + 4 <= dim; i += 4) {
    float32x4_t va = vld1q_f32(a + i);
    float32x4_t vb = vld1q_f32(b + i);
    float32x4_t diff = vsubq_f32(va, vb);
    sum_vec = vmlaq_f32(sum_vec, diff, diff);
  }
#if defined(__aarch64__)
  float sum = vaddvq_f32(sum_vec);
#else
  float t[4];
  vst1q_f32(t, sum_vec);
  float sum = t[0] + t[1] + t[2] + t[3];
#endif
#elif defined(__SSE__)
  __m128 sum_vec = _mm_setzero_ps();
  for (; i + 4 <= dim; i += 4) {
    __m128 va = _mm_loadu_ps(a + i);
    __m128 vb = _mm_loadu_ps(b + i);
    __m128 diff = _mm_sub_ps(va, vb);
    sum_vec = _mm_add_ps(sum_vec, _mm_mul_ps(diff, diff));
  }
  float tmp[4];
  _mm_storeu_ps(tmp, sum_vec);
  float sum = tmp[0] + tmp[1] + tmp[2] + tmp[3];
#else
  float sum = 0.0f;
#endif

  for (; i < dim; i++) {
    float d = a[i] - b[i];
    sum += d * d;
  }
  return sum;
}

float vec_l2_distance(const float *a, const float *b, uint32_t dim) {
  return sqrtf(vec_l2_distance_sq(a, b, dim));
}

float vec_cosine_distance(const float *a, const float *b, uint32_t dim) {
  uint32_t i = 0;
#if defined(__AVX512F__)
  __m512 dot_vec = _mm512_setzero_ps();
  __m512 norma_vec = _mm512_setzero_ps();
  __m512 normb_vec = _mm512_setzero_ps();
  for (; i + 16 <= dim; i += 16) {
    __m512 va = _mm512_loadu_ps(a + i);
    __m512 vb = _mm512_loadu_ps(b + i);
    dot_vec = _mm512_add_ps(dot_vec, _mm512_mul_ps(va, vb));
    norma_vec = _mm512_add_ps(norma_vec, _mm512_mul_ps(va, va));
    normb_vec = _mm512_add_ps(normb_vec, _mm512_mul_ps(vb, vb));
  }
  float dot = _mm512_reduce_add_ps(dot_vec);
  float norm_a = _mm512_reduce_add_ps(norma_vec);
  float norm_b = _mm512_reduce_add_ps(normb_vec);
#elif defined(__AVX2__)
  __m256 dot_vec = _mm256_setzero_ps();
  __m256 norma_vec = _mm256_setzero_ps();
  __m256 normb_vec = _mm256_setzero_ps();
  for (; i + 8 <= dim; i += 8) {
    __m256 va = _mm256_loadu_ps(a + i);
    __m256 vb = _mm256_loadu_ps(b + i);
    dot_vec = _mm256_add_ps(dot_vec, _mm256_mul_ps(va, vb));
    norma_vec = _mm256_add_ps(norma_vec, _mm256_mul_ps(va, va));
    normb_vec = _mm256_add_ps(normb_vec, _mm256_mul_ps(vb, vb));
  }
  float t_dot[8], t_na[8], t_nb[8];
  _mm256_storeu_ps(t_dot, dot_vec);
  _mm256_storeu_ps(t_na, norma_vec);
  _mm256_storeu_ps(t_nb, normb_vec);
  float dot = t_dot[0]+t_dot[1]+t_dot[2]+t_dot[3]+t_dot[4]+t_dot[5]+t_dot[6]+t_dot[7];
  float norm_a = t_na[0]+t_na[1]+t_na[2]+t_na[3]+t_na[4]+t_na[5]+t_na[6]+t_na[7];
  float norm_b = t_nb[0]+t_nb[1]+t_nb[2]+t_nb[3]+t_nb[4]+t_nb[5]+t_nb[6]+t_nb[7];
#elif defined(__ARM_NEON)
  float32x4_t dot_vec = vdupq_n_f32(0.0f);
  float32x4_t norma_vec = vdupq_n_f32(0.0f);
  float32x4_t normb_vec = vdupq_n_f32(0.0f);
  for (; i + 4 <= dim; i += 4) {
    float32x4_t va = vld1q_f32(a + i);
    float32x4_t vb = vld1q_f32(b + i);
    dot_vec = vmlaq_f32(dot_vec, va, vb);
    norma_vec = vmlaq_f32(norma_vec, va, va);
    normb_vec = vmlaq_f32(normb_vec, vb, vb);
  }
#if defined(__aarch64__)
  float dot = vaddvq_f32(dot_vec);
  float norm_a = vaddvq_f32(norma_vec);
  float norm_b = vaddvq_f32(normb_vec);
#else
  float t_dot[4], t_na[4], t_nb[4];
  vst1q_f32(t_dot, dot_vec);
  vst1q_f32(t_na, norma_vec);
  vst1q_f32(t_nb, normb_vec);
  float dot = t_dot[0]+t_dot[1]+t_dot[2]+t_dot[3];
  float norm_a = t_na[0]+t_na[1]+t_na[2]+t_na[3];
  float norm_b = t_nb[0]+t_nb[1]+t_nb[2]+t_nb[3];
#endif
#elif defined(__SSE__)
  __m128 dot_vec = _mm_setzero_ps();
  __m128 norma_vec = _mm_setzero_ps();
  __m128 normb_vec = _mm_setzero_ps();
  for (; i + 4 <= dim; i += 4) {
    __m128 va = _mm_loadu_ps(a + i);
    __m128 vb = _mm_loadu_ps(b + i);
    dot_vec = _mm_add_ps(dot_vec, _mm_mul_ps(va, vb));
    norma_vec = _mm_add_ps(norma_vec, _mm_mul_ps(va, va));
    normb_vec = _mm_add_ps(normb_vec, _mm_mul_ps(vb, vb));
  }
  float t_dot[4], t_na[4], t_nb[4];
  _mm_storeu_ps(t_dot, dot_vec);
  _mm_storeu_ps(t_na, norma_vec);
  _mm_storeu_ps(t_nb, normb_vec);
  float dot = t_dot[0]+t_dot[1]+t_dot[2]+t_dot[3];
  float norm_a = t_na[0]+t_na[1]+t_na[2]+t_na[3];
  float norm_b = t_nb[0]+t_nb[1]+t_nb[2]+t_nb[3];
#else
  float dot = 0.0f;
  float norm_a = 0.0f;
  float norm_b = 0.0f;
#endif

  for (; i < dim; i++) {
    dot += a[i] * b[i];
    norm_a += a[i] * a[i];
    norm_b += b[i] * b[i];
  }

  float denom = sqrtf(norm_a) * sqrtf(norm_b);
  if (denom < 1e-12f)
    return 1.0f; /* treat zero-vectors as maximally dissimilar */

  float cosine_sim = dot / denom;

  if (cosine_sim > 1.0f)
    cosine_sim = 1.0f;
  if (cosine_sim < -1.0f)
    cosine_sim = -1.0f;

  return 1.0f - cosine_sim;
}

float vec_inner_product_distance(const float *a, const float *b, uint32_t dim) {
  uint32_t i = 0;
#if defined(__AVX512F__)
  __m512 dot_vec = _mm512_setzero_ps();
  for (; i + 16 <= dim; i += 16) {
    __m512 va = _mm512_loadu_ps(a + i);
    __m512 vb = _mm512_loadu_ps(b + i);
    dot_vec = _mm512_add_ps(dot_vec, _mm512_mul_ps(va, vb));
  }
  float dot = _mm512_reduce_add_ps(dot_vec);
#elif defined(__AVX2__)
  __m256 dot_vec = _mm256_setzero_ps();
  for (; i + 8 <= dim; i += 8) {
    __m256 va = _mm256_loadu_ps(a + i);
    __m256 vb = _mm256_loadu_ps(b + i);
    dot_vec = _mm256_add_ps(dot_vec, _mm256_mul_ps(va, vb));
  }
  float tmp_dot[8];
  _mm256_storeu_ps(tmp_dot, dot_vec);
  float dot = tmp_dot[0]+tmp_dot[1]+tmp_dot[2]+tmp_dot[3]+tmp_dot[4]+tmp_dot[5]+tmp_dot[6]+tmp_dot[7];
#elif defined(__ARM_NEON)
  float32x4_t dot_vec = vdupq_n_f32(0.0f);
  for (; i + 4 <= dim; i += 4) {
    float32x4_t va = vld1q_f32(a + i);
    float32x4_t vb = vld1q_f32(b + i);
    dot_vec = vmlaq_f32(dot_vec, va, vb);
  }
#if defined(__aarch64__)
  float dot = vaddvq_f32(dot_vec);
#else
  float t_dot[4];
  vst1q_f32(t_dot, dot_vec);
  float dot = t_dot[0]+t_dot[1]+t_dot[2]+t_dot[3];
#endif
#elif defined(__SSE__)
  __m128 dot_vec = _mm_setzero_ps();
  for (; i + 4 <= dim; i += 4) {
    __m128 va = _mm_loadu_ps(a + i);
    __m128 vb = _mm_loadu_ps(b + i);
    dot_vec = _mm_add_ps(dot_vec, _mm_mul_ps(va, vb));
  }
  float t_dot[4];
  _mm_storeu_ps(t_dot, dot_vec);
  float dot = t_dot[0]+t_dot[1]+t_dot[2]+t_dot[3];
#else
  float dot = 0.0f;
#endif

  for (; i < dim; i++) {
    dot += a[i] * b[i];
  }
  return -dot;
}

void vec_normalize(float *v, uint32_t dim) {
  float norm = 0.0f;
  uint32_t i = 0;

#if defined(__AVX512F__)
  __m512 sum_vec = _mm512_setzero_ps();
  for (; i + 16 <= dim; i += 16) {
    __m512 vv = _mm512_loadu_ps(v + i);
    sum_vec = _mm512_add_ps(sum_vec, _mm512_mul_ps(vv, vv));
  }
  norm = _mm512_reduce_add_ps(sum_vec);
#elif defined(__AVX2__)
  __m256 sum_vec = _mm256_setzero_ps();
  for (; i + 8 <= dim; i += 8) {
    __m256 vv = _mm256_loadu_ps(v + i);
    sum_vec = _mm256_add_ps(sum_vec, _mm256_mul_ps(vv, vv));
  }
  float tmp[8];
  _mm256_storeu_ps(tmp, sum_vec);
  norm = tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
#elif defined(__ARM_NEON)
  float32x4_t sum_vec = vdupq_n_f32(0.0f);
  for (; i + 4 <= dim; i += 4) {
    float32x4_t vv = vld1q_f32(v + i);
    sum_vec = vmlaq_f32(sum_vec, vv, vv);
  }
#if defined(__aarch64__)
  norm = vaddvq_f32(sum_vec);
#else
  float t[4];
  vst1q_f32(t, sum_vec);
  norm = t[0]+t[1]+t[2]+t[3];
#endif
#elif defined(__SSE__)
  __m128 sum_vec = _mm_setzero_ps();
  for (; i + 4 <= dim; i += 4) {
    __m128 vv = _mm_loadu_ps(v + i);
    sum_vec = _mm_add_ps(sum_vec, _mm_mul_ps(vv, vv));
  }
  float t[4];
  _mm_storeu_ps(t, sum_vec);
  norm = t[0]+t[1]+t[2]+t[3];
#endif

  for (; i < dim; i++) {
    norm += v[i] * v[i];
  }
  
  norm = sqrtf(norm);

  if (norm < 1e-12f)
    return;

  float inv = 1.0f / norm;
  i = 0;

#if defined(__AVX512F__)
  __m512 inv_vec = _mm512_set1_ps(inv);
  for (; i + 16 <= dim; i += 16) {
    __m512 vv = _mm512_loadu_ps(v + i);
    _mm512_storeu_ps(v + i, _mm512_mul_ps(vv, inv_vec));
  }
#elif defined(__AVX2__)
  __m256 inv_vec = _mm256_set1_ps(inv);
  for (; i + 8 <= dim; i += 8) {
    __m256 vv = _mm256_loadu_ps(v + i);
    _mm256_storeu_ps(v + i, _mm256_mul_ps(vv, inv_vec));
  }
#elif defined(__ARM_NEON)
  float32x4_t inv_vec = vdupq_n_f32(inv);
  for (; i + 4 <= dim; i += 4) {
    float32x4_t vv = vld1q_f32(v + i);
    vst1q_f32(v + i, vmulq_f32(vv, inv_vec));
  }
#elif defined(__SSE__)
  __m128 inv_vec = _mm_set1_ps(inv);
  for (; i + 4 <= dim; i += 4) {
    __m128 vv = _mm_loadu_ps(v + i);
    _mm_storeu_ps(v + i, _mm_mul_ps(vv, inv_vec));
  }
#endif

  for (; i < dim; i++) {
    v[i] *= inv;
  }
}

vec_dist_fn_t vec_get_distance_fn(VecDistanceMetric metric) {
  switch (metric) {
    case VEC_DIST_L2:             return vec_l2_distance_sq;
    case VEC_DIST_COSINE:         return vec_cosine_distance;
    case VEC_DIST_INNER_PRODUCT:  return vec_inner_product_distance;
    default:                      return vec_l2_distance_sq;
  }
}
