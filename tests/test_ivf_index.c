#include "../src/index/vec_utils.h"
#include "../src/index/vec_topk.h"
#include "../src/index/vec_page.h"
#include "../src/index/flat/flat_index.h"
#include "../src/index/ivf/ivf_index.h"
#include "../src/pages/storage_mgr.h"
#include "../src/buffer/buffer_pool.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <unistd.h>

#define TEST_DB_PATH "/tmp/test_ivf_index.db"

/* ============================================================
 *  Test environment helpers
 * ============================================================ */

typedef struct {
    StorageManager sm;
    BufferPool     bp;
    IvfIndex       idx;
} IvfTestEnv;

static void ivf_env_setup(IvfTestEnv *env, uint16_t dim,
                           VecDistanceMetric metric,
                           uint32_t nlist, uint32_t nprobe,
                           uint32_t bp_frames) {
    unlink(TEST_DB_PATH);
    assert(storage_open(&env->sm, TEST_DB_PATH) == 0);
    assert(bp_init(&env->bp, &env->sm, bp_frames) == 0);
    assert(ivf_index_create(&env->idx, &env->bp, dim, metric,
                            nlist, nprobe) == 0);
}

static void ivf_env_teardown(IvfTestEnv *env) {
    ivf_index_destroy(&env->idx);
    bp_flush_all(&env->bp);
    bp_destroy(&env->bp);
    storage_close(&env->sm);
    unlink(TEST_DB_PATH);
}

/* ============================================================
 *  Helper: generate clustered synthetic data
 *
 *  Creates `n_clusters` clusters, each with `per_cluster` vectors.
 *  Cluster i has center at (i*spread, i*spread, ..., i*spread)
 *  with small Gaussian-ish noise.
 * ============================================================ */

static float *generate_clustered_data(uint16_t dim, uint32_t n_clusters,
                                       uint32_t per_cluster, float spread,
                                       uint32_t *total_out) {
    uint32_t total = n_clusters * per_cluster;
    *total_out = total;
    float *data = malloc((size_t)total * dim * sizeof(float));
    assert(data != NULL);

    srand(42);  /* deterministic for reproducibility */

    for (uint32_t c = 0; c < n_clusters; c++) {
        float center = (float)c * spread;
        for (uint32_t v = 0; v < per_cluster; v++) {
            float *vec = data + (size_t)(c * per_cluster + v) * dim;
            for (uint16_t d = 0; d < dim; d++) {
                /* Small noise: [-0.5, 0.5] */
                float noise = ((float)rand() / (float)RAND_MAX) - 0.5f;
                vec[d] = center + noise * 0.5f;
            }
        }
    }
    return data;
}

/* ============================================================
 *  Test: index lifecycle before training
 * ============================================================ */

static void test_ivf_untrained(void) {
    IvfTestEnv env;
    ivf_env_setup(&env, 4, VEC_DIST_L2, 4, 2, 256);

    /* Should not be trained */
    assert(!ivf_index_is_trained(&env.idx));
    assert(ivf_index_count(&env.idx) == 0);

    /* Insert should fail before training */
    float vec[] = {1.0f, 2.0f, 3.0f, 4.0f};
    assert(ivf_index_insert(&env.idx, vec, NULL, NULL) == -1);

    /* Search should fail before training */
    VecResult results[1];
    uint32_t num;
    assert(ivf_index_search(&env.idx, vec, 1, results, &num) == -1);

    ivf_env_teardown(&env);
    printf("  PASS: test_ivf_untrained\n");
}

/* ============================================================
 *  Test: k-means training converges on well-separated clusters
 * ============================================================ */

static void test_ivf_train_basic(void) {
    IvfTestEnv env;
    uint16_t dim = 4;
    uint32_t nlist = 4;
    ivf_env_setup(&env, dim, VEC_DIST_L2, nlist, 2, 256);

    /* Generate 4 well-separated clusters */
    uint32_t total;
    float *data = generate_clustered_data(dim, nlist, 50, 100.0f, &total);

    /* Reseed PRNG for deterministic k-means initialization */
    srand(12345);
    assert(ivf_index_train(&env.idx, data, total) == 0);
    assert(ivf_index_is_trained(&env.idx));

    /* Training a second time should fail */
    assert(ivf_index_train(&env.idx, data, total) == -1);

    /* Verify centroids roughly match cluster centers.
     * Sort centroid first-components and compare to expected {0, 100, 200, 300}. */
    float centroid_firsts[4];
    for (uint32_t i = 0; i < nlist; i++)
        centroid_firsts[i] = env.idx.centroids[i * dim];

    /* Simple bubble sort */
    for (int i = 0; i < 3; i++)
        for (int j = i + 1; j < 4; j++)
            if (centroid_firsts[i] > centroid_firsts[j]) {
                float tmp = centroid_firsts[i];
                centroid_firsts[i] = centroid_firsts[j];
                centroid_firsts[j] = tmp;
            }

    /* Each centroid's first dimension should be near its cluster center */
    for (uint32_t i = 0; i < nlist; i++) {
        float expected = (float)i * 100.0f;
        assert(fabsf(centroid_firsts[i] - expected) < 10.0f);
    }

    free(data);
    ivf_env_teardown(&env);
    printf("  PASS: test_ivf_train_basic\n");
}

/* ============================================================
 *  Test: train with insufficient data should fail
 * ============================================================ */

static void test_ivf_train_insufficient(void) {
    IvfTestEnv env;
    ivf_env_setup(&env, 4, VEC_DIST_L2, 10, 2, 64);

    /* Only 5 vectors for nlist=10 — must fail */
    float data[5 * 4];
    memset(data, 0, sizeof(data));
    assert(ivf_index_train(&env.idx, data, 5) == -1);
    assert(!ivf_index_is_trained(&env.idx));

    ivf_env_teardown(&env);
    printf("  PASS: test_ivf_train_insufficient\n");
}

/* ============================================================
 *  Test: insert routing and search accuracy
 * ============================================================ */

static void test_ivf_insert_and_search(void) {
    IvfTestEnv env;
    uint16_t dim = 3;
    uint32_t nlist = 4;
    uint32_t nprobe = 2;
    ivf_env_setup(&env, dim, VEC_DIST_L2, nlist, nprobe, 256);

    /* Train on 4 well-separated clusters of 20 vectors each */
    uint32_t total;
    float *train_data = generate_clustered_data(dim, nlist, 20, 100.0f, &total);
    assert(ivf_index_train(&env.idx, train_data, total) == 0);

    /* Insert the training data */
    for (uint32_t i = 0; i < total; i++)
        assert(ivf_index_insert(&env.idx, train_data + (size_t)i * dim,
                                NULL, NULL) == 0);

    assert(ivf_index_count(&env.idx) == total);

    /* Search near the first cluster center (≈ 0,0,0) */
    float query[] = {0.0f, 0.0f, 0.0f};
    VecResult results[5];
    uint32_t num;
    assert(ivf_index_search(&env.idx, query, 5, results, &num) == 0);
    assert(num == 5);

    /* All 5 results should be from cluster 0 (near origin), so
     * their distances should be small (< some threshold) */
    for (uint32_t i = 0; i < num; i++)
        assert(results[i].distance < 2.0f);  /* well within cluster radius */

    /* Results should be sorted ascending */
    for (uint32_t i = 1; i < num; i++)
        assert(results[i].distance >= results[i - 1].distance - 1e-6f);

    free(train_data);
    ivf_env_teardown(&env);
    printf("  PASS: test_ivf_insert_and_search\n");
}

/* ============================================================
 *  Test: nprobe = nlist should give same results as flat scan
 * ============================================================ */

static void test_ivf_full_probe(void) {
    uint16_t dim = 4;
    uint32_t nlist = 4;
    uint32_t n_vecs = 80;

    /* Setup IVF index */
    IvfTestEnv ivf_env;
    ivf_env_setup(&ivf_env, dim, VEC_DIST_L2, nlist, nlist, 256);  /* nprobe=nlist */

    /* Setup flat index for comparison */
    unlink("/tmp/test_ivf_flat_cmp.db");
    StorageManager sm2;
    BufferPool bp2;
    FlatIndex flat;
    assert(storage_open(&sm2, "/tmp/test_ivf_flat_cmp.db") == 0);
    assert(bp_init(&bp2, &sm2, 64) == 0);
    assert(flat_index_create(&flat, &bp2, dim, VEC_DIST_L2) == 0);

    /* Generate training and insert data */
    srand(123);
    float *data = malloc((size_t)n_vecs * dim * sizeof(float));
    assert(data != NULL);
    for (uint32_t i = 0; i < n_vecs; i++)
        for (uint16_t d = 0; d < dim; d++)
            data[i * dim + d] = (float)rand() / (float)RAND_MAX * 100.0f;

    /* Train IVF */
    assert(ivf_index_train(&ivf_env.idx, data, n_vecs) == 0);

    /* Insert into both */
    for (uint32_t i = 0; i < n_vecs; i++) {
        assert(ivf_index_insert(&ivf_env.idx, data + (size_t)i * dim,
                                NULL, NULL) == 0);
        assert(flat_index_insert(&flat, data + (size_t)i * dim,
                                 NULL, NULL) == 0);
    }

    /* Query */
    float query[4] = {50.0f, 50.0f, 50.0f, 50.0f};
    uint32_t k = 5;

    VecResult ivf_res[5], flat_res[5];
    uint32_t ivf_num, flat_num;

    assert(ivf_index_search(&ivf_env.idx, query, k, ivf_res, &ivf_num) == 0);
    assert(flat_index_search(&flat, query, k, flat_res, &flat_num) == 0);

    assert(ivf_num == flat_num);

    /* With nprobe = nlist, IVF searches everything, so distances
     * should match flat search exactly. */
    for (uint32_t i = 0; i < ivf_num; i++)
        assert(fabsf(ivf_res[i].distance - flat_res[i].distance) < 1e-4f);

    /* Cleanup */
    free(data);
    flat_index_destroy(&flat);
    bp_flush_all(&bp2);
    bp_destroy(&bp2);
    storage_close(&sm2);
    unlink("/tmp/test_ivf_flat_cmp.db");
    ivf_env_teardown(&ivf_env);
    printf("  PASS: test_ivf_full_probe\n");
}

/* ============================================================
 *  Test: cosine metric through IVF
 * ============================================================ */

static void test_ivf_cosine(void) {
    IvfTestEnv env;
    uint16_t dim = 3;
    uint32_t nlist = 2;
    ivf_env_setup(&env, dim, VEC_DIST_COSINE, nlist, nlist, 128);

    /* Two directional clusters */
    float train_data[] = {
        /* Cluster 0: pointing roughly along +x */
         1.0f, 0.1f, 0.0f,
         1.0f, 0.0f, 0.1f,
         1.0f, 0.05f, 0.05f,
         0.9f, 0.1f, 0.1f,
        /* Cluster 1: pointing roughly along +y */
         0.1f, 1.0f, 0.0f,
         0.0f, 1.0f, 0.1f,
         0.05f, 1.0f, 0.05f,
         0.1f, 0.9f, 0.1f,
    };
    uint32_t n_train = 8;

    assert(ivf_index_train(&env.idx, train_data, n_train) == 0);

    for (uint32_t i = 0; i < n_train; i++)
        assert(ivf_index_insert(&env.idx, train_data + (size_t)i * dim,
                                NULL, NULL) == 0);

    /* Query along +x → closest should be cluster-0 vectors */
    float query[] = {1.0f, 0.0f, 0.0f};
    VecResult results[3];
    uint32_t num;
    assert(ivf_index_search(&env.idx, query, 3, results, &num) == 0);
    assert(num == 3);

    /* All top-3 should have small cosine distance (< 0.1) */
    for (uint32_t i = 0; i < num; i++)
        assert(results[i].distance < 0.15f);

    ivf_env_teardown(&env);
    printf("  PASS: test_ivf_cosine\n");
}

/* ============================================================
 *  Test: inner product metric through IVF
 * ============================================================ */

static void test_ivf_inner_product(void) {
    IvfTestEnv env;
    uint16_t dim = 3;
    uint32_t nlist = 2;
    ivf_env_setup(&env, dim, VEC_DIST_INNER_PRODUCT, nlist, nlist, 128);

    /* Two clusters with different magnitudes */
    float train_data[] = {
        /* Cluster 0: small vectors */
         0.1f, 0.1f, 0.1f,
         0.2f, 0.1f, 0.0f,
         0.0f, 0.2f, 0.1f,
         0.1f, 0.0f, 0.2f,
        /* Cluster 1: large vectors */
         5.0f, 5.0f, 5.0f,
         6.0f, 5.0f, 4.0f,
         4.0f, 6.0f, 5.0f,
         5.0f, 4.0f, 6.0f,
    };
    uint32_t n_train = 8;

    assert(ivf_index_train(&env.idx, train_data, n_train) == 0);

    for (uint32_t i = 0; i < n_train; i++)
        assert(ivf_index_insert(&env.idx, train_data + (size_t)i * dim,
                                NULL, NULL) == 0);

    /* Query (1,1,1): IP distances are negative dot products.
     * Large vectors in cluster 1 have highest dot product → most negative distance. */
    float query[] = {1.0f, 1.0f, 1.0f};
    VecResult results[3];
    uint32_t num;
    assert(ivf_index_search(&env.idx, query, 3, results, &num) == 0);
    assert(num == 3);

    /* All top-3 should be from cluster 1 (large vectors), so distance < -10 */
    for (uint32_t i = 0; i < num; i++)
        assert(results[i].distance < -10.0f);

    ivf_env_teardown(&env);
    printf("  PASS: test_ivf_inner_product\n");
}

/* ============================================================
 *  Test: multi-page partitions with many vectors
 * ============================================================ */

static void test_ivf_multipage(void) {
    IvfTestEnv env;
    uint16_t dim = 64;
    uint32_t nlist = 4;
    uint32_t nprobe = 2;
    /* Generous buffer pool for many pages */
    ivf_env_setup(&env, dim, VEC_DIST_L2, nlist, nprobe, 512);

    /* Generate clustered data — enough to fill multiple pages per partition */
    uint32_t per_cluster = 200;
    uint32_t total;
    float *data = generate_clustered_data(dim, nlist, per_cluster, 1000.0f, &total);

    assert(ivf_index_train(&env.idx, data, total) == 0);

    for (uint32_t i = 0; i < total; i++)
        assert(ivf_index_insert(&env.idx, data + (size_t)i * dim,
                                NULL, NULL) == 0);

    assert(ivf_index_count(&env.idx) == total);

    /* Verify that at least one partition spans multiple pages */
    bool found_multipage = false;
    for (uint32_t i = 0; i < nlist; i++) {
        if (env.idx.partitions[i].num_pages > 1) {
            found_multipage = true;
            break;
        }
    }
    assert(found_multipage);

    /* Search near cluster 0 center */
    float *query = calloc(dim, sizeof(float));  /* origin ≈ cluster 0 center */
    assert(query != NULL);

    VecResult results[10];
    uint32_t num;
    assert(ivf_index_search(&env.idx, query, 10, results, &num) == 0);
    assert(num == 10);

    /* Results should be sorted */
    for (uint32_t i = 1; i < num; i++)
        assert(results[i].distance >= results[i - 1].distance - 1e-6f);

    free(query);
    free(data);
    ivf_env_teardown(&env);
    printf("  PASS: test_ivf_multipage\n");
}

/* ============================================================
 *  Test: k > total vectors
 * ============================================================ */

static void test_ivf_k_exceeds_total(void) {
    IvfTestEnv env;
    uint16_t dim = 3;
    uint32_t nlist = 2;
    ivf_env_setup(&env, dim, VEC_DIST_L2, nlist, nlist, 64);

    /* Train with minimal data */
    float train_data[] = {
        0.0f, 0.0f, 0.0f,
        10.0f, 10.0f, 10.0f,
        1.0f, 0.0f, 0.0f,
        9.0f, 10.0f, 10.0f,
    };
    assert(ivf_index_train(&env.idx, train_data, 4) == 0);

    /* Insert only 3 vectors */
    float vecs[][3] = {
        {0.0f, 0.0f, 0.0f},
        {1.0f, 0.0f, 0.0f},
        {10.0f, 10.0f, 10.0f},
    };
    for (int i = 0; i < 3; i++)
        assert(ivf_index_insert(&env.idx, vecs[i], NULL, NULL) == 0);

    /* Ask for k=10 but only 3 exist */
    VecResult results[10];
    uint32_t num;
    assert(ivf_index_search(&env.idx, vecs[0], 10, results, &num) == 0);
    assert(num == 3);

    /* First result should be exact match */
    assert(fabsf(results[0].distance) < 1e-6f);

    ivf_env_teardown(&env);
    printf("  PASS: test_ivf_k_exceeds_total\n");
}

/* ============================================================
 *  Main
 * ============================================================ */

int main(void) {
    printf("=== IVF Vector Index Tests ===\n");

    test_ivf_untrained();
    test_ivf_train_basic();
    test_ivf_train_insufficient();
    test_ivf_insert_and_search();
    test_ivf_full_probe();
    test_ivf_cosine();
    test_ivf_inner_product();
    test_ivf_multipage();
    test_ivf_k_exceeds_total();

    printf("All IVF vector index tests passed.\n\n");
    return 0;
}
