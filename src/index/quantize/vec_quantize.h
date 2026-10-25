#ifndef VEC_QUANTIZE_H
#define VEC_QUANTIZE_H

#include <stdint.h>
#include <stdbool.h>

/*
 * Scalar Quantization (SQ8) — compresses 32-bit float vectors to 8-bit
 * unsigned integers, achieving a 4× memory reduction.
 *
 * Each dimension is independently mapped from [min_val, max_val] → [0, 255].
 * The codebook stores per-dimension min and range values learned during
 * training, enabling encode/decode and distance computation in quantized space.
 *
 * Product Quantization (PQ) — divides the vector into M sub-vectors, each
 * quantized independently using a codebook of 256 centroids. Achieves
 * up to 16× compression for 128-d vectors with M=16 (1 byte per sub-vector).
 *
 * Both methods support asymmetric distance computation (ADC):
 * the query remains in full precision while database vectors are quantized.
 */

#define SQ8_MAX_DIM  2000
#define PQ_MAX_NSUB  64
#define PQ_KSUB      256    /* fixed codebook size per sub-space */
#define PQ_TRAIN_MAX_ITERS 25
#define PQ_TRAIN_TOL 1e-6f

/* ============================================================
 *  Scalar Quantization (SQ8)
 * ============================================================ */

typedef struct {
    uint16_t dim;
    float   *min_vals;       /* per-dimension minimum (dim floats)  */
    float   *ranges;         /* per-dimension range (dim floats)    */
    bool     is_trained;
} SQ8Quantizer;

/* Initialize an untrained SQ8 quantizer for the given dimension. */
int sq8_init(SQ8Quantizer *sq, uint16_t dim);

/* Free all internal memory. */
void sq8_destroy(SQ8Quantizer *sq);

/*
 * Train the quantizer by computing per-dimension min/max from training data.
 * vecs: n vectors of dimension dim, row-major.
 * Returns 0 on success, -1 on failure.
 */
int sq8_train(SQ8Quantizer *sq, const float *vecs, uint32_t n);

/*
 * Encode a single float vector into an SQ8 code.
 * out_codes must point to at least `dim` bytes.
 * Returns 0 on success, -1 if not trained.
 */
int sq8_encode(const SQ8Quantizer *sq, const float *vec, uint8_t *out_codes);

/*
 * Decode an SQ8 code back to approximate float vector.
 * out_vec must point to at least `dim` floats.
 */
int sq8_decode(const SQ8Quantizer *sq, const uint8_t *codes, float *out_vec);

/*
 * Compute asymmetric L2² distance between a float query and an SQ8-encoded vector.
 * This avoids decoding the database vector fully — precomputes a lookup table.
 *
 * dist_table: precomputed per-dimension distance table (dim × 256 floats).
 * Call sq8_precompute_l2_table() once per query, then sq8_asymmetric_l2() per vector.
 */
int sq8_precompute_l2_table(const SQ8Quantizer *sq, const float *query,
                             float *dist_table);

float sq8_asymmetric_l2(const float *dist_table, const uint8_t *codes,
                         uint16_t dim);

/*
 * Batch encode n vectors. out_codes must be at least n*dim bytes.
 */
int sq8_encode_batch(const SQ8Quantizer *sq, const float *vecs,
                      uint32_t n, uint8_t *out_codes);

/* ============================================================
 *  Product Quantization (PQ)
 * ============================================================ */

typedef struct {
    uint16_t dim;
    uint16_t nsub;           /* number of sub-vectors (M in PQ literature)  */
    uint16_t dsub;           /* sub-vector dimensionality = dim / nsub      */
    uint16_t _pad;
    float   *codebooks;     /* nsub codebooks, each ksub × dsub floats     */
                             /* layout: codebooks[sub][code][d]             */
                             /* flat: nsub * PQ_KSUB * dsub floats          */
    bool     is_trained;
} PQQuantizer;

/* Initialize an untrained PQ quantizer. nsub must divide dim evenly. */
int pq_init(PQQuantizer *pq, uint16_t dim, uint16_t nsub);

/* Free all internal memory. */
void pq_destroy(PQQuantizer *pq);

/*
 * Train the PQ codebooks via k-means on each sub-space.
 * vecs: n vectors of dimension dim, row-major.
 * Returns 0 on success, -1 on failure.
 */
int pq_train(PQQuantizer *pq, const float *vecs, uint32_t n);

/*
 * Encode a single vector into a PQ code (nsub bytes).
 * out_codes must point to at least `nsub` bytes.
 */
int pq_encode(const PQQuantizer *pq, const float *vec, uint8_t *out_codes);

/*
 * Decode a PQ code back to approximate float vector.
 * out_vec must point to at least `dim` floats.
 */
int pq_decode(const PQQuantizer *pq, const uint8_t *codes, float *out_vec);

/*
 * Precompute asymmetric distance table for a query vector.
 * table: nsub × PQ_KSUB floats. table[sub][code] = L2²(query_sub, centroid).
 */
int pq_precompute_l2_table(const PQQuantizer *pq, const float *query,
                            float *table);

/*
 * Compute asymmetric L2² distance from precomputed table and a PQ code.
 */
float pq_asymmetric_l2(const float *table, const uint8_t *codes,
                        uint16_t nsub);

/*
 * Batch encode n vectors. out_codes must be at least n*nsub bytes.
 */
int pq_encode_batch(const PQQuantizer *pq, const float *vecs,
                     uint32_t n, uint8_t *out_codes);

#endif
