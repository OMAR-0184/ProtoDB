#include "vec_utils.h"
#include <math.h>

float vec_l2_distance_sq(const float *a, const float *b, uint32_t dim) {
    float sum = 0.0f;
    for (uint32_t i = 0; i < dim; i++) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

float vec_l2_distance(const float *a, const float *b, uint32_t dim) {
    return sqrtf(vec_l2_distance_sq(a, b, dim));
}

float vec_cosine_distance(const float *a, const float *b, uint32_t dim) {
    float dot = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;

    for (uint32_t i = 0; i < dim; i++) {
        dot    += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    float denom = sqrtf(norm_a) * sqrtf(norm_b);
    if (denom < 1e-12f)
        return 1.0f;  /* treat zero-vectors as maximally dissimilar */

    float cosine_sim = dot / denom;

    /* Clamp to [-1, 1] to guard against floating-point drift */
    if (cosine_sim > 1.0f)  cosine_sim = 1.0f;
    if (cosine_sim < -1.0f) cosine_sim = -1.0f;

    return 1.0f - cosine_sim;
}

float vec_inner_product_distance(const float *a, const float *b, uint32_t dim) {
    float dot = 0.0f;
    for (uint32_t i = 0; i < dim; i++) {
        dot += a[i] * b[i];
    }
    /* Negate so that higher dot product → smaller "distance" */
    return -dot;
}

void vec_normalize(float *v, uint32_t dim) {
    float norm = 0.0f;
    for (uint32_t i = 0; i < dim; i++) {
        norm += v[i] * v[i];
    }
    norm = sqrtf(norm);

    if (norm < 1e-12f)
        return;  /* don't divide by zero */

    float inv = 1.0f / norm;
    for (uint32_t i = 0; i < dim; i++) {
        v[i] *= inv;
    }
}
