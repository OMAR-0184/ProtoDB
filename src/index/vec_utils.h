#ifndef VEC_UTILS_H
#define VEC_UTILS_H

#include <stdint.h>

#define VECTOR_MAX_DIM 1024

/*
 * Squared Euclidean (L2²) distance.
 * We skip the sqrt during scan for performance;
 * the caller can sqrt the final results if needed.
 */
float vec_l2_distance_sq(const float *a, const float *b, uint32_t dim);

/* Full Euclidean distance (with sqrt). */
float vec_l2_distance(const float *a, const float *b, uint32_t dim);

/*
 * Cosine distance = 1.0 - cosine_similarity.
 * Range: [0, 2].  0 = identical direction.
 */
float vec_cosine_distance(const float *a, const float *b, uint32_t dim);

/*
 * Negative inner product, so that smaller = more similar.
 * This lets the top-k heap use a uniform "min distance wins" convention.
 */
float vec_inner_product_distance(const float *a, const float *b, uint32_t dim);

/* In-place L2 normalization. */
void vec_normalize(float *v, uint32_t dim);

#endif
