#include "vec_utils.h"
#include <math.h>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define USE_NEON 1
#elif defined(__SSE__)
#include <xmmintrin.h>
#define USE_SSE 1
#endif

float vec_l2_distance_sq(const float *a, const float *b, uint32_t dim) {
#if defined(USE_NEON)
    float32x4_t sum_vec = vdupq_n_f32(0.0f);
    uint32_t i = 0;

    for (; i + 4 <= dim; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        float32x4_t diff = vsubq_f32(va, vb);
        sum_vec = vmlaq_f32(sum_vec, diff, diff);
    }

    float sum = vaddvq_f32(sum_vec);

    for (; i < dim; i++) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;

#elif defined(USE_SSE)
    __m128 sum_vec = _mm_setzero_ps();
    uint32_t i = 0;

    for (; i + 4 <= dim; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vb = _mm_loadu_ps(b + i);
        __m128 diff = _mm_sub_ps(va, vb);
        sum_vec = _mm_add_ps(sum_vec, _mm_mul_ps(diff, diff));
    }

    float tmp[4];
    _mm_storeu_ps(tmp, sum_vec);
    float sum = tmp[0] + tmp[1] + tmp[2] + tmp[3];

    for (; i < dim; i++) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;

#else
    float sum = 0.0f;
    for (uint32_t i = 0; i < dim; i++) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
#endif
}

float vec_l2_distance(const float *a, const float *b, uint32_t dim) {
    return sqrtf(vec_l2_distance_sq(a, b, dim));
}

float vec_cosine_distance(const float *a, const float *b, uint32_t dim) {
#if defined(USE_NEON)
    float32x4_t dot_vec   = vdupq_n_f32(0.0f);
    float32x4_t norma_vec = vdupq_n_f32(0.0f);
    float32x4_t normb_vec = vdupq_n_f32(0.0f);
    uint32_t i = 0;

    for (; i + 4 <= dim; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        dot_vec   = vmlaq_f32(dot_vec,   va, vb);
        norma_vec = vmlaq_f32(norma_vec, va, va);
        normb_vec = vmlaq_f32(normb_vec, vb, vb);
    }

    float dot    = vaddvq_f32(dot_vec);
    float norm_a = vaddvq_f32(norma_vec);
    float norm_b = vaddvq_f32(normb_vec);

    for (; i < dim; i++) {
        dot    += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

#elif defined(USE_SSE)
    __m128 dot_vec   = _mm_setzero_ps();
    __m128 norma_vec = _mm_setzero_ps();
    __m128 normb_vec = _mm_setzero_ps();
    uint32_t i = 0;

    for (; i + 4 <= dim; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vb = _mm_loadu_ps(b + i);
        dot_vec   = _mm_add_ps(dot_vec,   _mm_mul_ps(va, vb));
        norma_vec = _mm_add_ps(norma_vec, _mm_mul_ps(va, va));
        normb_vec = _mm_add_ps(normb_vec, _mm_mul_ps(vb, vb));
    }

    float tmp[4];
    _mm_storeu_ps(tmp, dot_vec);
    float dot = tmp[0] + tmp[1] + tmp[2] + tmp[3];
    _mm_storeu_ps(tmp, norma_vec);
    float norm_a = tmp[0] + tmp[1] + tmp[2] + tmp[3];
    _mm_storeu_ps(tmp, normb_vec);
    float norm_b = tmp[0] + tmp[1] + tmp[2] + tmp[3];

    for (; i < dim; i++) {
        dot    += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

#else
    float dot = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;

    for (uint32_t i = 0; i < dim; i++) {
        dot    += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
#endif

    float denom = sqrtf(norm_a) * sqrtf(norm_b);
    if (denom < 1e-12f)
        return 1.0f;

    float cosine_sim = dot / denom;

    if (cosine_sim > 1.0f)  cosine_sim = 1.0f;
    if (cosine_sim < -1.0f) cosine_sim = -1.0f;

    return 1.0f - cosine_sim;
}

float vec_inner_product_distance(const float *a, const float *b, uint32_t dim) {
#if defined(USE_NEON)
    float32x4_t dot_vec = vdupq_n_f32(0.0f);
    uint32_t i = 0;

    for (; i + 4 <= dim; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        dot_vec = vmlaq_f32(dot_vec, va, vb);
    }

    float dot = vaddvq_f32(dot_vec);

    for (; i < dim; i++)
        dot += a[i] * b[i];

    return -dot;

#elif defined(USE_SSE)
    __m128 dot_vec = _mm_setzero_ps();
    uint32_t i = 0;

    for (; i + 4 <= dim; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vb = _mm_loadu_ps(b + i);
        dot_vec = _mm_add_ps(dot_vec, _mm_mul_ps(va, vb));
    }

    float tmp[4];
    _mm_storeu_ps(tmp, dot_vec);
    float dot = tmp[0] + tmp[1] + tmp[2] + tmp[3];

    for (; i < dim; i++)
        dot += a[i] * b[i];

    return -dot;

#else
    float dot = 0.0f;
    for (uint32_t i = 0; i < dim; i++)
        dot += a[i] * b[i];
    return -dot;
#endif
}

void vec_normalize(float *v, uint32_t dim) {
    float norm = 0.0f;

#if defined(USE_NEON)
    float32x4_t sum_vec = vdupq_n_f32(0.0f);
    uint32_t i = 0;

    for (; i + 4 <= dim; i += 4) {
        float32x4_t vv = vld1q_f32(v + i);
        sum_vec = vmlaq_f32(sum_vec, vv, vv);
    }

    norm = vaddvq_f32(sum_vec);

    for (; i < dim; i++)
        norm += v[i] * v[i];

#elif defined(USE_SSE)
    __m128 sum_vec = _mm_setzero_ps();
    uint32_t i = 0;

    for (; i + 4 <= dim; i += 4) {
        __m128 vv = _mm_loadu_ps(v + i);
        sum_vec = _mm_add_ps(sum_vec, _mm_mul_ps(vv, vv));
    }

    float tmp[4];
    _mm_storeu_ps(tmp, sum_vec);
    norm = tmp[0] + tmp[1] + tmp[2] + tmp[3];

    for (; i < dim; i++)
        norm += v[i] * v[i];

#else
    for (uint32_t i = 0; i < dim; i++)
        norm += v[i] * v[i];
#endif

    norm = sqrtf(norm);

    if (norm < 1e-12f)
        return;

    float inv = 1.0f / norm;

#if defined(USE_NEON)
    float32x4_t inv_vec = vdupq_n_f32(inv);
    i = 0;

    for (; i + 4 <= dim; i += 4) {
        float32x4_t vv = vld1q_f32(v + i);
        vst1q_f32(v + i, vmulq_f32(vv, inv_vec));
    }

    for (; i < dim; i++)
        v[i] *= inv;

#elif defined(USE_SSE)
    __m128 inv_vec = _mm_set1_ps(inv);
    i = 0;

    for (; i + 4 <= dim; i += 4) {
        __m128 vv = _mm_loadu_ps(v + i);
        _mm_storeu_ps(v + i, _mm_mul_ps(vv, inv_vec));
    }

    for (; i < dim; i++)
        v[i] *= inv;

#else
    for (uint32_t i = 0; i < dim; i++)
        v[i] *= inv;
#endif
}
