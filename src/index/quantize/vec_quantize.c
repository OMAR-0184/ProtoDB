#include "vec_quantize.h"
#include "../vec_utils.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ============================================================
 *  SQ8 — Scalar Quantization (8-bit per dimension)
 * ============================================================ */

int sq8_init(SQ8Quantizer *sq, uint16_t dim) {
    if (dim == 0 || dim > SQ8_MAX_DIM)
        return -1;

    memset(sq, 0, sizeof(SQ8Quantizer));
    sq->dim = dim;

    sq->min_vals = malloc(dim * sizeof(float));
    sq->ranges = malloc(dim * sizeof(float));
    if (!sq->min_vals || !sq->ranges) {
        free(sq->min_vals);
        free(sq->ranges);
        sq->min_vals = NULL;
        sq->ranges = NULL;
        return -1;
    }

    sq->is_trained = false;
    return 0;
}

void sq8_destroy(SQ8Quantizer *sq) {
    free(sq->min_vals);
    free(sq->ranges);
    sq->min_vals = NULL;
    sq->ranges = NULL;
    sq->is_trained = false;
}

int sq8_train(SQ8Quantizer *sq, const float *vecs, uint32_t n) {
    if (!vecs || n == 0 || sq->is_trained)
        return -1;

    uint16_t dim = sq->dim;

    /* Initialize min/max with first vector */
    float *max_vals = malloc(dim * sizeof(float));
    if (!max_vals) return -1;

    for (uint16_t d = 0; d < dim; d++) {
        sq->min_vals[d] = vecs[d];
        max_vals[d] = vecs[d];
    }

    /* Scan all vectors */
    for (uint32_t i = 1; i < n; i++) {
        const float *v = vecs + (size_t)i * dim;
        for (uint16_t d = 0; d < dim; d++) {
            if (v[d] < sq->min_vals[d]) sq->min_vals[d] = v[d];
            if (v[d] > max_vals[d]) max_vals[d] = v[d];
        }
    }

    /* Compute ranges with epsilon to avoid division by zero */
    for (uint16_t d = 0; d < dim; d++) {
        sq->ranges[d] = max_vals[d] - sq->min_vals[d];
        if (sq->ranges[d] < 1e-10f)
            sq->ranges[d] = 1e-10f;
    }

    free(max_vals);
    sq->is_trained = true;
    return 0;
}

int sq8_encode(const SQ8Quantizer *sq, const float *vec, uint8_t *out_codes) {
    if (!sq->is_trained || !vec || !out_codes)
        return -1;

    for (uint16_t d = 0; d < sq->dim; d++) {
        float normalized = (vec[d] - sq->min_vals[d]) / sq->ranges[d];

        /* Clamp to [0, 1] */
        if (normalized < 0.0f) normalized = 0.0f;
        if (normalized > 1.0f) normalized = 1.0f;

        out_codes[d] = (uint8_t)(normalized * 255.0f + 0.5f);
    }
    return 0;
}

int sq8_decode(const SQ8Quantizer *sq, const uint8_t *codes, float *out_vec) {
    if (!sq->is_trained || !codes || !out_vec)
        return -1;

    for (uint16_t d = 0; d < sq->dim; d++) {
        out_vec[d] = sq->min_vals[d] +
                     ((float)codes[d] / 255.0f) * sq->ranges[d];
    }
    return 0;
}

int sq8_precompute_l2_table(const SQ8Quantizer *sq, const float *query,
                             float *dist_table) {
    if (!sq->is_trained || !query || !dist_table)
        return -1;

    /*
     * dist_table[d * 256 + code] = (query[d] - decode(code, d))²
     *
     * For SQ8, decode(code, d) = min_vals[d] + (code/255) * ranges[d]
     */
    for (uint16_t d = 0; d < sq->dim; d++) {
        float qd = query[d];
        float min_d = sq->min_vals[d];
        float range_d = sq->ranges[d];
        float inv = range_d / 255.0f;

        for (uint32_t code = 0; code < 256; code++) {
            float decoded = min_d + (float)code * inv;
            float diff = qd - decoded;
            dist_table[(size_t)d * 256 + code] = diff * diff;
        }
    }
    return 0;
}

float sq8_asymmetric_l2(const float *dist_table, const uint8_t *codes,
                         uint16_t dim) {
    float sum = 0.0f;

    /* SIMD-friendly: sequential access pattern */
    uint16_t i = 0;

#if defined(__ARM_NEON) || defined(__aarch64__)
    /* Process 4 dimensions at a time using NEON */
    for (; i + 4 <= dim; i += 4) {
        float d0 = dist_table[(size_t)(i + 0) * 256 + codes[i + 0]];
        float d1 = dist_table[(size_t)(i + 1) * 256 + codes[i + 1]];
        float d2 = dist_table[(size_t)(i + 2) * 256 + codes[i + 2]];
        float d3 = dist_table[(size_t)(i + 3) * 256 + codes[i + 3]];
        sum += d0 + d1 + d2 + d3;
    }
#endif

    for (; i < dim; i++) {
        sum += dist_table[(size_t)i * 256 + codes[i]];
    }

    return sum;
}

int sq8_encode_batch(const SQ8Quantizer *sq, const float *vecs,
                      uint32_t n, uint8_t *out_codes) {
    if (!sq->is_trained || !vecs || !out_codes)
        return -1;

    for (uint32_t i = 0; i < n; i++) {
        if (sq8_encode(sq, vecs + (size_t)i * sq->dim,
                       out_codes + (size_t)i * sq->dim) < 0)
            return -1;
    }
    return 0;
}

/* ============================================================
 *  PQ — Product Quantization
 * ============================================================ */

int pq_init(PQQuantizer *pq, uint16_t dim, uint16_t nsub) {
    if (dim == 0 || nsub == 0 || dim % nsub != 0)
        return -1;
    if (nsub > PQ_MAX_NSUB)
        return -1;

    memset(pq, 0, sizeof(PQQuantizer));
    pq->dim = dim;
    pq->nsub = nsub;
    pq->dsub = dim / nsub;

    /* Allocate codebooks: nsub * PQ_KSUB * dsub floats */
    size_t cb_size = (size_t)nsub * PQ_KSUB * pq->dsub * sizeof(float);
    pq->codebooks = malloc(cb_size);
    if (!pq->codebooks)
        return -1;

    pq->is_trained = false;
    return 0;
}

void pq_destroy(PQQuantizer *pq) {
    free(pq->codebooks);
    pq->codebooks = NULL;
    pq->is_trained = false;
}

/*
 * Train one sub-space codebook via k-means.
 * sub_vecs: n vectors of dimension dsub.
 * centroids: output PQ_KSUB × dsub floats.
 */
static int pq_train_subspace(const float *sub_vecs, uint32_t n,
                              float *centroids, uint16_t dsub) {
    uint32_t k = PQ_KSUB;
    if (n < k) k = n;

    /* Initialize centroids from first k vectors */
    memcpy(centroids, sub_vecs, (size_t)k * dsub * sizeof(float));

    float *new_centroids = calloc((size_t)k * dsub, sizeof(float));
    uint32_t *counts = calloc(k, sizeof(uint32_t));
    if (!new_centroids || !counts) {
        free(new_centroids);
        free(counts);
        return -1;
    }

    for (int iter = 0; iter < PQ_TRAIN_MAX_ITERS; iter++) {
        memset(new_centroids, 0, (size_t)k * dsub * sizeof(float));
        memset(counts, 0, k * sizeof(uint32_t));

        /* Assignment */
        for (uint32_t i = 0; i < n; i++) {
            const float *v = sub_vecs + (size_t)i * dsub;
            uint32_t best = 0;
            float best_dist = FLT_MAX;

            for (uint32_t c = 0; c < k; c++) {
                float dist = 0.0f;
                const float *cent = centroids + (size_t)c * dsub;
                for (uint16_t d = 0; d < dsub; d++) {
                    float diff = v[d] - cent[d];
                    dist += diff * diff;
                }
                if (dist < best_dist) {
                    best_dist = dist;
                    best = c;
                }
            }

            counts[best]++;
            float *acc = new_centroids + (size_t)best * dsub;
            for (uint16_t d = 0; d < dsub; d++)
                acc[d] += v[d];
        }

        /* Update centroids */
        float max_shift = 0.0f;
        for (uint32_t c = 0; c < k; c++) {
            float *nc = new_centroids + (size_t)c * dsub;
            float *oc = centroids + (size_t)c * dsub;

            if (counts[c] == 0) continue;

            float inv = 1.0f / (float)counts[c];
            for (uint16_t d = 0; d < dsub; d++)
                nc[d] *= inv;

            float shift = 0.0f;
            for (uint16_t d = 0; d < dsub; d++) {
                float diff = oc[d] - nc[d];
                shift += diff * diff;
            }
            if (shift > max_shift) max_shift = shift;

            memcpy(oc, nc, dsub * sizeof(float));
        }

        if (max_shift < PQ_TRAIN_TOL) break;
    }

    free(new_centroids);
    free(counts);
    return 0;
}

int pq_train(PQQuantizer *pq, const float *vecs, uint32_t n) {
    if (!vecs || n == 0 || pq->is_trained)
        return -1;
    if (n < PQ_KSUB)
        return -1; /* Need at least 256 training vectors */

    uint16_t dsub = pq->dsub;

    /* Extract and train each sub-space independently */
    float *sub_vecs = malloc((size_t)n * dsub * sizeof(float));
    if (!sub_vecs) return -1;

    for (uint16_t s = 0; s < pq->nsub; s++) {
        /* Extract sub-vectors for this sub-space */
        for (uint32_t i = 0; i < n; i++) {
            const float *src = vecs + (size_t)i * pq->dim + (size_t)s * dsub;
            float *dst = sub_vecs + (size_t)i * dsub;
            memcpy(dst, src, dsub * sizeof(float));
        }

        /* Train codebook for this sub-space */
        float *codebook = pq->codebooks +
                           (size_t)s * PQ_KSUB * dsub;
        if (pq_train_subspace(sub_vecs, n, codebook, dsub) < 0) {
            free(sub_vecs);
            return -1;
        }
    }

    free(sub_vecs);
    pq->is_trained = true;
    return 0;
}

int pq_encode(const PQQuantizer *pq, const float *vec, uint8_t *out_codes) {
    if (!pq->is_trained || !vec || !out_codes)
        return -1;

    uint16_t dsub = pq->dsub;

    for (uint16_t s = 0; s < pq->nsub; s++) {
        const float *sub = vec + (size_t)s * dsub;
        const float *codebook = pq->codebooks +
                                 (size_t)s * PQ_KSUB * dsub;

        uint8_t best = 0;
        float best_dist = FLT_MAX;

        for (uint32_t c = 0; c < PQ_KSUB; c++) {
            const float *cent = codebook + (size_t)c * dsub;
            float dist = 0.0f;
            for (uint16_t d = 0; d < dsub; d++) {
                float diff = sub[d] - cent[d];
                dist += diff * diff;
            }
            if (dist < best_dist) {
                best_dist = dist;
                best = (uint8_t)c;
            }
        }

        out_codes[s] = best;
    }
    return 0;
}

int pq_decode(const PQQuantizer *pq, const uint8_t *codes, float *out_vec) {
    if (!pq->is_trained || !codes || !out_vec)
        return -1;

    uint16_t dsub = pq->dsub;

    for (uint16_t s = 0; s < pq->nsub; s++) {
        const float *centroid = pq->codebooks +
                                 (size_t)s * PQ_KSUB * dsub +
                                 (size_t)codes[s] * dsub;
        memcpy(out_vec + (size_t)s * dsub, centroid, dsub * sizeof(float));
    }
    return 0;
}

int pq_precompute_l2_table(const PQQuantizer *pq, const float *query,
                            float *table) {
    if (!pq->is_trained || !query || !table)
        return -1;

    uint16_t dsub = pq->dsub;

    /*
     * table[s * PQ_KSUB + c] = L2²(query_sub_s, centroid_s_c)
     */
    for (uint16_t s = 0; s < pq->nsub; s++) {
        const float *qsub = query + (size_t)s * dsub;
        const float *codebook = pq->codebooks +
                                 (size_t)s * PQ_KSUB * dsub;

        for (uint32_t c = 0; c < PQ_KSUB; c++) {
            const float *cent = codebook + (size_t)c * dsub;
            float dist = 0.0f;
            for (uint16_t d = 0; d < dsub; d++) {
                float diff = qsub[d] - cent[d];
                dist += diff * diff;
            }
            table[(size_t)s * PQ_KSUB + c] = dist;
        }
    }
    return 0;
}

float pq_asymmetric_l2(const float *table, const uint8_t *codes,
                        uint16_t nsub) {
    float sum = 0.0f;
    for (uint16_t s = 0; s < nsub; s++) {
        sum += table[(size_t)s * PQ_KSUB + codes[s]];
    }
    return sum;
}

int pq_encode_batch(const PQQuantizer *pq, const float *vecs,
                     uint32_t n, uint8_t *out_codes) {
    if (!pq->is_trained || !vecs || !out_codes)
        return -1;

    for (uint32_t i = 0; i < n; i++) {
        if (pq_encode(pq, vecs + (size_t)i * pq->dim,
                      out_codes + (size_t)i * pq->nsub) < 0)
            return -1;
    }
    return 0;
}
