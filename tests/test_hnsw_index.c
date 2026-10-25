#include "../src/index/hnsw/hnsw_index.h"
#include "../src/index/vec_utils.h"
#include "../src/pages/storage_mgr.h"
#include "../src/buffer/buffer_pool.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <unistd.h>

#define TEST_DB_PATH "/tmp/test_hnsw_index.db"

typedef struct {
    StorageManager sm;
    BufferPool     bp;
    HnswIndex      idx;
} HnswTestEnv;

static void hnsw_env_setup(HnswTestEnv *env, uint16_t dim,
                           VecDistanceMetric metric,
                           uint32_t M, uint32_t ef_construction,
                           uint32_t bp_frames) {
    unlink(TEST_DB_PATH);
    assert(storage_open(&env->sm, TEST_DB_PATH) == 0);
    assert(bp_init(&env->bp, &env->sm, bp_frames) == 0);
    assert(hnsw_index_create(&env->idx, &env->bp, dim, metric,
                             M, ef_construction) == 0);
}

static void hnsw_env_teardown(HnswTestEnv *env) {
    hnsw_index_destroy(&env->idx);
    bp_flush_all(&env->bp);
    bp_destroy(&env->bp);
    storage_close(&env->sm);
    unlink(TEST_DB_PATH);
}

/* ============================================================
 *  Tests
 * ============================================================ */

static void test_hnsw_create_destroy(void) {
    HnswTestEnv env;
    hnsw_env_setup(&env, 32, VEC_DIST_L2, 16, 100, 64);

    assert(hnsw_index_count(&env.idx) == 0);
    assert(env.idx.dim == 32);
    assert(env.idx.M == 16);
    assert(env.idx.M_max0 == 32);
    assert(env.idx.ef_construction == 100);

    hnsw_env_teardown(&env);
    printf("  PASS: test_hnsw_create_destroy\n");
}

static void test_hnsw_empty_search(void) {
    HnswTestEnv env;
    hnsw_env_setup(&env, 8, VEC_DIST_L2, 8, 50, 64);

    float q[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    VecResult results[10];
    uint32_t num_results = 999;

    assert(hnsw_index_search(&env.idx, q, 5, 0, results, &num_results) == 0);
    assert(num_results == 0);

    hnsw_env_teardown(&env);
    printf("  PASS: test_hnsw_empty_search\n");
}

static void test_hnsw_insert_single(void) {
    HnswTestEnv env;
    hnsw_env_setup(&env, 4, VEC_DIST_L2, 8, 50, 64);

    float v[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    page_id_t pid;
    uint16_t slot;
    assert(hnsw_index_insert(&env.idx, v, &pid, &slot) == 0);
    assert(hnsw_index_count(&env.idx) == 1);

    VecResult results[5];
    uint32_t num_results = 0;
    assert(hnsw_index_search(&env.idx, v, 1, 0, results, &num_results) == 0);
    assert(num_results == 1);
    assert(results[0].page_id == pid);
    assert(results[0].slot_index == slot);
    assert(fabsf(results[0].distance) < 1e-5f);

    hnsw_env_teardown(&env);
    printf("  PASS: test_hnsw_insert_single\n");
}

static void test_hnsw_exact_recall(void) {
    uint16_t dim = 16;
    uint32_t num_vecs = 100;
    HnswTestEnv env;
    hnsw_env_setup(&env, dim, VEC_DIST_L2, 16, 100, 128);

    srand(42);
    float *vectors = malloc((size_t)num_vecs * dim * sizeof(float));
    for (uint32_t i = 0; i < num_vecs * dim; i++) {
        vectors[i] = (float)rand() / (float)RAND_MAX * 10.0f;
    }

    for (uint32_t i = 0; i < num_vecs; i++) {
        page_id_t pid;
        uint16_t slot;
        assert(hnsw_index_insert(&env.idx, vectors + (size_t)i * dim, &pid, &slot) == 0);
    }
    assert(hnsw_index_count(&env.idx) == num_vecs);

    /* Test 1-NN recall: searching for each inserted vector should find itself */
    uint32_t recall_hits = 0;
    for (uint32_t i = 0; i < num_vecs; i++) {
        VecResult res[1];
        uint32_t num_res = 0;
        assert(hnsw_index_search(&env.idx, vectors + (size_t)i * dim, 1, 64, res, &num_res) == 0);
        assert(num_res == 1);
        if (fabsf(res[0].distance) < 1e-4f) {
            recall_hits++;
        }
    }

    /* 100% 1-NN recall on 100 vectors */
    assert(recall_hits == num_vecs);

    free(vectors);
    hnsw_env_teardown(&env);
    printf("  PASS: test_hnsw_exact_recall (100/100 1-NN exact recall)\n");
}

static void test_hnsw_topk_ranking(void) {
    uint16_t dim = 8;
    uint32_t num_vecs = 150;
    uint32_t k = 10;
    HnswTestEnv env;
    hnsw_env_setup(&env, dim, VEC_DIST_L2, 12, 80, 128);

    srand(1234);
    float *vectors = malloc((size_t)num_vecs * dim * sizeof(float));
    for (uint32_t i = 0; i < num_vecs * dim; i++) {
        vectors[i] = ((float)rand() / (float)RAND_MAX) * 5.0f;
    }

    for (uint32_t i = 0; i < num_vecs; i++) {
        assert(hnsw_index_insert(&env.idx, vectors + (size_t)i * dim, NULL, NULL) == 0);
    }

    float q[8] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    VecResult results[10];
    uint32_t num_res = 0;
    assert(hnsw_index_search(&env.idx, q, k, 64, results, &num_res) == 0);
    assert(num_res == k);

    /* Verify monotonicity: dist[0] <= dist[1] <= ... <= dist[k-1] */
    for (uint32_t i = 1; i < num_res; i++) {
        assert(results[i].distance >= results[i-1].distance);
    }

    free(vectors);
    hnsw_env_teardown(&env);
    printf("  PASS: test_hnsw_topk_ranking (monotonic top-%u distances)\n", k);
}

static void test_hnsw_cosine_metric(void) {
    uint16_t dim = 4;
    HnswTestEnv env;
    hnsw_env_setup(&env, dim, VEC_DIST_COSINE, 8, 50, 64);

    float v1[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    float v2[4] = {0.0f, 1.0f, 0.0f, 0.0f};
    float v3[4] = {0.7071f, 0.7071f, 0.0f, 0.0f};

    assert(hnsw_index_insert(&env.idx, v1, NULL, NULL) == 0);
    assert(hnsw_index_insert(&env.idx, v2, NULL, NULL) == 0);
    assert(hnsw_index_insert(&env.idx, v3, NULL, NULL) == 0);

    /* Query with v1: nearest should be v1 with distance ~0, then v3 with ~0.293, then v2 with ~1.0 */
    VecResult res[3];
    uint32_t num_res = 0;
    assert(hnsw_index_search(&env.idx, v1, 3, 0, res, &num_res) == 0);
    assert(num_res == 3);
    assert(fabsf(res[0].distance) < 1e-4f);
    assert(res[1].distance > 0.1f && res[1].distance < 0.5f);
    assert(fabsf(res[2].distance - 1.0f) < 1e-4f);

    hnsw_env_teardown(&env);
    printf("  PASS: test_hnsw_cosine_metric\n");
}

static void test_hnsw_persistence(void) {
    uint16_t dim = 8;
    uint32_t num_vecs = 100;
    HnswTestEnv env;
    hnsw_env_setup(&env, dim, VEC_DIST_L2, 8, 60, 128);

    srand(999);
    float *vectors = malloc((size_t)num_vecs * dim * sizeof(float));
    for (uint32_t i = 0; i < num_vecs * dim; i++) {
        vectors[i] = ((float)rand() / (float)RAND_MAX) * 10.0f;
    }

    for (uint32_t i = 0; i < num_vecs; i++) {
        assert(hnsw_index_insert(&env.idx, vectors + (size_t)i * dim, NULL, NULL) == 0);
    }

    /* Save index to stream page */
    page_id_t save_pid = INVALID_PAGE_ID;
    assert(hnsw_index_save(&env.idx, &save_pid) == 0);
    assert(save_pid != INVALID_PAGE_ID);

    /* Run a query before destroying */
    float q[8] = {5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f};
    VecResult orig_results[5];
    uint32_t orig_count = 0;
    assert(hnsw_index_search(&env.idx, q, 5, 64, orig_results, &orig_count) == 0);
    assert(orig_count == 5);

    /* Destroy the in-memory index */
    hnsw_index_destroy(&env.idx);

    /* Load back from stream page */
    HnswIndex loaded_idx;
    assert(hnsw_index_load(&loaded_idx, &env.bp, save_pid) == 0);
    assert(hnsw_index_count(&loaded_idx) == num_vecs);
    assert(loaded_idx.dim == dim);
    assert(loaded_idx.M == env.idx.M || loaded_idx.M == 8);

    /* Query the loaded index and verify identical top-5 results */
    VecResult loaded_results[5];
    uint32_t loaded_count = 0;
    assert(hnsw_index_search(&loaded_idx, q, 5, 64, loaded_results, &loaded_count) == 0);
    assert(loaded_count == 5);

    for (uint32_t i = 0; i < 5; i++) {
        assert(loaded_results[i].page_id == orig_results[i].page_id);
        assert(loaded_results[i].slot_index == orig_results[i].slot_index);
        assert(fabsf(loaded_results[i].distance - orig_results[i].distance) < 1e-4f);
    }

    hnsw_index_destroy(&loaded_idx);
    free(vectors);
    bp_flush_all(&env.bp);
    bp_destroy(&env.bp);
    storage_close(&env.sm);
    unlink(TEST_DB_PATH);
    printf("  PASS: test_hnsw_persistence (save, load, and query match)\n");
}

/* ============================================================
 *  Main
 * ============================================================ */

int main(void) {
    printf("--- Running HNSW Index Tests ---\n");

    test_hnsw_create_destroy();
    test_hnsw_empty_search();
    test_hnsw_insert_single();
    test_hnsw_exact_recall();
    test_hnsw_topk_ranking();
    test_hnsw_cosine_metric();
    test_hnsw_persistence();

    printf("=== All HNSW Index Tests PASSED! ===\n\n");
    return 0;
}
