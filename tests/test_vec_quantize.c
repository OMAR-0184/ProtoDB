#include "../src/index/quantize/vec_quantize.h"
#include "../src/index/vec_utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

/* ============================================================
 *  SQ8 Tests
 * ============================================================ */

static void test_sq8_init_destroy(void) {
    SQ8Quantizer sq;
    assert(sq8_init(&sq, 64) == 0);
    assert(sq.dim == 64);
    assert(!sq.is_trained);
    assert(sq.min_vals != NULL);
    assert(sq.ranges != NULL);

    sq8_destroy(&sq);
    assert(sq.min_vals == NULL);
    assert(sq.ranges == NULL);
    printf("  PASS: test_sq8_init_destroy\n");
}

static void test_sq8_train_and_bounds(void) {
    SQ8Quantizer sq;
    uint16_t dim = 4;
    assert(sq8_init(&sq, dim) == 0);

    /* 3 training vectors:
     * v0: [ 0.0, 10.0, -5.0,  2.0 ]
     * v1: [ 5.0, 20.0,  0.0,  2.0 ]
     * v2: [10.0, 15.0,  5.0,  2.0 ]
     */
    float vecs[] = {
        0.0f, 10.0f, -5.0f, 2.0f,
        5.0f, 20.0f,  0.0f, 2.0f,
       10.0f, 15.0f,  5.0f, 2.0f
    };

    assert(sq8_train(&sq, vecs, 3) == 0);
    assert(sq.is_trained);

    /* Dimension 0: min 0.0, max 10.0, range 10.0 */
    assert(fabsf(sq.min_vals[0] - 0.0f) < 1e-5f);
    assert(fabsf(sq.ranges[0] - 10.0f) < 1e-5f);

    /* Dimension 1: min 10.0, max 20.0, range 10.0 */
    assert(fabsf(sq.min_vals[1] - 10.0f) < 1e-5f);
    assert(fabsf(sq.ranges[1] - 10.0f) < 1e-5f);

    /* Dimension 2: min -5.0, max 5.0, range 10.0 */
    assert(fabsf(sq.min_vals[2] - (-5.0f)) < 1e-5f);
    assert(fabsf(sq.ranges[2] - 10.0f) < 1e-5f);

    /* Dimension 3: constant 2.0 -> min 2.0, range clamped to 1e-6f */
    assert(fabsf(sq.min_vals[3] - 2.0f) < 1e-5f);
    assert(sq.ranges[3] > 0.0f);

    sq8_destroy(&sq);
    printf("  PASS: test_sq8_train_and_bounds\n");
}

static void test_sq8_encode_decode(void) {
    SQ8Quantizer sq;
    uint16_t dim = 8;
    assert(sq8_init(&sq, dim) == 0);

    /* Train with uniformly spaced values [0, 100] */
    uint32_t n = 200;
    float *train_data = malloc((size_t)n * dim * sizeof(float));
    for (uint32_t i = 0; i < n * dim; i++) {
        train_data[i] = ((float)rand() / (float)RAND_MAX) * 100.0f;
    }
    assert(sq8_train(&sq, train_data, n) == 0);

    /* Encode a test vector */
    float original[8] = {0.0f, 25.0f, 50.0f, 75.0f, 100.0f, 33.3f, 66.6f, 12.5f};
    uint8_t codes[8];
    assert(sq8_encode(&sq, original, codes) == 0);

    /* Decode back */
    float decoded[8];
    assert(sq8_decode(&sq, codes, decoded) == 0);

    /* Verify error on each dimension is within 1 quantization step: range / 255 */
    for (uint16_t d = 0; d < dim; d++) {
        float step = sq.ranges[d] / 255.0f;
        float err = fabsf(original[d] - decoded[d]);
        assert(err <= step + 1e-3f);
    }

    free(train_data);
    sq8_destroy(&sq);
    printf("  PASS: test_sq8_encode_decode (4x compression, bounded error)\n");
}

static void test_sq8_asymmetric_l2(void) {
    SQ8Quantizer sq;
    uint16_t dim = 16;
    assert(sq8_init(&sq, dim) == 0);

    uint32_t n = 300;
    float *train_data = malloc((size_t)n * dim * sizeof(float));
    for (uint32_t i = 0; i < n * dim; i++) {
        train_data[i] = ((float)rand() / (float)RAND_MAX) * 10.0f;
    }
    assert(sq8_train(&sq, train_data, n) == 0);

    /* Test vectors */
    float query[16];
    float db_vec[16];
    for (uint16_t d = 0; d < dim; d++) {
        query[d] = 5.0f;
        db_vec[d] = 7.0f;
    }

    uint8_t codes[16];
    assert(sq8_encode(&sq, db_vec, codes) == 0);

    /* Precompute table */
    float dist_table[16 * 256];
    assert(sq8_precompute_l2_table(&sq, query, dist_table) == 0);

    /* Asymmetric distance */
    float approx_d2 = sq8_asymmetric_l2(dist_table, codes, dim);
    float exact_d2 = vec_l2_distance_sq(query, db_vec, dim);

    /* The difference should be very small (within 5% for quantized representation) */
    float rel_diff = fabsf(approx_d2 - exact_d2) / exact_d2;
    assert(rel_diff < 0.05f);

    free(train_data);
    sq8_destroy(&sq);
    printf("  PASS: test_sq8_asymmetric_l2 (ADC matches true L2 within 5%%)\n");
}

static void test_sq8_batch_encode(void) {
    SQ8Quantizer sq;
    uint16_t dim = 4;
    assert(sq8_init(&sq, dim) == 0);

    float train_data[] = {
        0.0f, 0.0f, 0.0f, 0.0f,
        10.0f, 10.0f, 10.0f, 10.0f
    };
    assert(sq8_train(&sq, train_data, 2) == 0);

    float test_vecs[] = {
        0.0f, 5.0f, 10.0f, 2.5f,
        10.0f, 5.0f, 0.0f, 7.5f
    };
    uint8_t batch_codes[8];
    assert(sq8_encode_batch(&sq, test_vecs, 2, batch_codes) == 0);

    uint8_t single_0[4], single_1[4];
    assert(sq8_encode(&sq, test_vecs, single_0) == 0);
    assert(sq8_encode(&sq, test_vecs + 4, single_1) == 0);

    assert(memcmp(batch_codes, single_0, 4) == 0);
    assert(memcmp(batch_codes + 4, single_1, 4) == 0);

    sq8_destroy(&sq);
    printf("  PASS: test_sq8_batch_encode\n");
}

/* ============================================================
 *  PQ Tests
 * ============================================================ */

static void test_pq_init_destroy(void) {
    PQQuantizer pq;
    assert(pq_init(&pq, 16, 4) == 0);
    assert(pq.dim == 16);
    assert(pq.nsub == 4);
    assert(pq.dsub == 4);
    assert(!pq.is_trained);
    assert(pq.codebooks != NULL);

    pq_destroy(&pq);
    assert(pq.codebooks == NULL);

    /* Incompatible nsub should fail */
    assert(pq_init(&pq, 15, 4) == -1);
    printf("  PASS: test_pq_init_destroy\n");
}

static void test_pq_train_and_encode(void) {
    PQQuantizer pq;
    uint16_t dim = 16;
    uint16_t nsub = 4; /* 4 sub-vectors of 4 floats each -> 4 bytes per vector! 16x compression */
    assert(pq_init(&pq, dim, nsub) == 0);

    /* Generate synthetic training data */
    uint32_t n = 400;
    srand(42);
    float *train_data = malloc((size_t)n * dim * sizeof(float));
    for (uint32_t i = 0; i < n * dim; i++) {
        train_data[i] = ((float)rand() / (float)RAND_MAX) * 10.0f;
    }

    assert(pq_train(&pq, train_data, n) == 0);
    assert(pq.is_trained);

    /* Test encode: 16 floats (64 bytes) -> 4 bytes (16x compression) */
    float test_vec[16];
    for (int i = 0; i < 16; i++) test_vec[i] = train_data[i];

    uint8_t code[4];
    assert(pq_encode(&pq, test_vec, code) == 0);

    /* Test decode */
    float decoded[16];
    assert(pq_decode(&pq, code, decoded) == 0);

    /* Verify decoded vector has reasonable values within range [0, 10] */
    for (int i = 0; i < 16; i++) {
        assert(decoded[i] >= -1.0f && decoded[i] <= 11.0f);
    }

    /* Test batch encode */
    uint8_t batch_codes[4 * 5];
    assert(pq_encode_batch(&pq, train_data, 5, batch_codes) == 0);
    assert(memcmp(batch_codes, code, 4) == 0);

    free(train_data);
    pq_destroy(&pq);
    printf("  PASS: test_pq_train_and_encode (16x compression from 64B to 4B)\n");
}

static void test_pq_asymmetric_l2(void) {
    PQQuantizer pq;
    uint16_t dim = 8;
    uint16_t nsub = 2; /* 2 sub-vectors of 4 floats */
    assert(pq_init(&pq, dim, nsub) == 0);

    uint32_t n = 300;
    srand(123);
    float *train_data = malloc((size_t)n * dim * sizeof(float));
    for (uint32_t i = 0; i < n * dim; i++) {
        train_data[i] = ((float)rand() / (float)RAND_MAX) * 5.0f;
    }
    assert(pq_train(&pq, train_data, n) == 0);

    float query[8] = {1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f};

    /* Precompute distance table */
    float dist_table[2 * PQ_KSUB];
    assert(pq_precompute_l2_table(&pq, query, dist_table) == 0);

    /* Compute asymmetric distance for two database vectors: one close, one far */
    float close_vec[8] = {1.1f, 2.1f, 2.9f, 4.1f, 1.0f, 2.0f, 3.1f, 3.9f};
    float far_vec[8]   = {9.0f, 9.0f, 9.0f, 9.0f, 9.0f, 9.0f, 9.0f, 9.0f};

    uint8_t close_code[2], far_code[2];
    assert(pq_encode(&pq, close_vec, close_code) == 0);
    assert(pq_encode(&pq, far_vec, far_code) == 0);

    float d_close = pq_asymmetric_l2(dist_table, close_code, nsub);
    float d_far   = pq_asymmetric_l2(dist_table, far_code, nsub);

    /* Close vector distance must be significantly smaller than far vector distance */
    assert(d_close < d_far);
    assert(d_close < 5.0f);
    assert(d_far > 10.0f);

    free(train_data);
    pq_destroy(&pq);
    printf("  PASS: test_pq_asymmetric_l2 (ADC preserves ranking)\n");
}

/* ============================================================
 *  Main
 * ============================================================ */

int main(void) {
    printf("--- Running Vector Quantization Tests ---\n");

    test_sq8_init_destroy();
    test_sq8_train_and_bounds();
    test_sq8_encode_decode();
    test_sq8_asymmetric_l2();
    test_sq8_batch_encode();

    test_pq_init_destroy();
    test_pq_train_and_encode();
    test_pq_asymmetric_l2();

    printf("=== All Vector Quantization Tests PASSED! ===\n\n");
    return 0;
}
