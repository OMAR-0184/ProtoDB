#include "../src/index/ivf/ivf_dynamic.h"
#include "../src/index/flat/flat_index.h"
#include "../src/index/vec_utils.h"
#include "../src/pages/storage_mgr.h"
#include "../src/buffer/buffer_pool.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <unistd.h>

#define TEST_DB_PATH "/tmp/test_ivf_dynamic.db"

typedef struct {
    StorageManager sm;
    BufferPool     bp;
    IvfIndex       idx;
} IvfDynTestEnv;

static void ivf_dyn_setup(IvfDynTestEnv *env, uint16_t dim,
                          VecDistanceMetric metric,
                          uint32_t nlist, uint32_t nprobe,
                          uint32_t bp_frames) {
    unlink(TEST_DB_PATH);
    assert(storage_open(&env->sm, TEST_DB_PATH) == 0);
    assert(bp_init(&env->bp, &env->sm, bp_frames) == 0);
    assert(ivf_index_create(&env->idx, &env->bp, dim, metric,
                            nlist, nprobe) == 0);
}

static void ivf_dyn_teardown(IvfDynTestEnv *env) {
    ivf_index_destroy(&env->idx);
    bp_flush_all(&env->bp);
    bp_destroy(&env->bp);
    storage_close(&env->sm);
    unlink(TEST_DB_PATH);
}

/* ============================================================
 *  Tests
 * ============================================================ */

static void test_dynamic_config_default(void) {
    IvfDynamicConfig cfg;
    ivf_dynamic_config_default(&cfg);

    assert(cfg.split_threshold == IVF_DYN_DEFAULT_SPLIT_THRESHOLD);
    assert(cfg.merge_threshold == IVF_DYN_DEFAULT_MERGE_THRESHOLD);
    assert(cfg.min_cluster_size == IVF_DYN_DEFAULT_MIN_CLUSTER_SIZE);
    assert(cfg.centroid_alpha == IVF_DYN_DEFAULT_CENTROID_ALPHA);
    assert(cfg.update_interval == IVF_DYN_DEFAULT_UPDATE_INTERVAL);

    printf("  PASS: test_dynamic_config_default\n");
}

static void test_dynamic_health_check(void) {
    IvfDynTestEnv env;
    uint16_t dim = 4;
    uint32_t nlist = 3;
    uint32_t nprobe = 2;
    ivf_dyn_setup(&env, dim, VEC_DIST_L2, nlist, nprobe, 64);

    /* Setup 3 distinct centroids manually */
    float train_data[] = {
        0.0f, 0.0f, 0.0f, 0.0f,
        10.0f, 10.0f, 10.0f, 10.0f,
        20.0f, 20.0f, 20.0f, 20.0f
    };
    assert(ivf_index_train(&env.idx, train_data, 3) == 0);
    for (int d = 0; d < 4; d++) env.idx.centroids[d] = 0.0f;
    for (int d = 0; d < 4; d++) env.idx.centroids[4 + d] = 10.0f;
    for (int d = 0; d < 4; d++) env.idx.centroids[8 + d] = 20.0f;

    /* Insert 10 vectors near cluster 0, 20 near cluster 1, and 2 near cluster 2 */
    for (int i = 0; i < 10; i++) {
        float v[4] = {0.1f * (float)i, 0.1f * (float)i, 0.0f, 0.0f};
        assert(ivf_index_insert(&env.idx, v, NULL, NULL) == 0);
    }
    for (int i = 0; i < 20; i++) {
        float v[4] = {10.0f + 0.1f * (float)i, 10.0f, 10.0f, 10.0f};
        assert(ivf_index_insert(&env.idx, v, NULL, NULL) == 0);
    }
    for (int i = 0; i < 2; i++) {
        float v[4] = {20.0f, 20.0f, 20.0f, 20.0f};
        assert(ivf_index_insert(&env.idx, v, NULL, NULL) == 0);
    }

    IvfDynamicConfig cfg;
    ivf_dynamic_config_default(&cfg);
    cfg.split_threshold = 1.8f;  /* mean is ~10.6, 20 > 10.6*1.8 (~19.1) -> oversized */
    cfg.merge_threshold = 0.3f;  /* 2 < 10.6*0.3 (~3.2) -> undersized */

    ClusterHealthReport report;
    assert(ivf_dynamic_health_check(&env.idx, &cfg, &report) == 0);

    assert(report.num_clusters == 3);
    assert(report.total_vectors == 32);
    assert(report.num_oversized >= 1);
    assert(report.num_undersized >= 1);

    ivf_dynamic_health_free(&report);
    ivf_dyn_teardown(&env);
    printf("  PASS: test_dynamic_health_check\n");
}

static void test_dynamic_update_centroid_ema(void) {
    IvfDynTestEnv env;
    uint16_t dim = 2;
    ivf_dyn_setup(&env, dim, VEC_DIST_L2, 2, 1, 64);

    float train_data[] = {
        0.0f, 0.0f,
        10.0f, 10.0f
    };
    assert(ivf_index_train(&env.idx, train_data, 2) == 0);
    env.idx.centroids[0] = 0.0f; env.idx.centroids[1] = 0.0f;
    env.idx.centroids[2] = 10.0f; env.idx.centroids[3] = 10.0f;

    /* Initial centroid 0 is near (0, 0) */
    float c0_x = env.idx.centroids[0];
    float c0_y = env.idx.centroids[1];

    /* Update centroid 0 with a vector at (10, 10) with alpha = 0.5 */
    float sample[2] = {10.0f, 10.0f};
    assert(ivf_dynamic_update_centroid(&env.idx, 0, sample, 0.5f) == 0);

    /* New centroid should have moved halfway towards (10, 10) */
    float expected_x = 0.5f * 10.0f + 0.5f * c0_x;
    float expected_y = 0.5f * 10.0f + 0.5f * c0_y;
    assert(fabsf(env.idx.centroids[0] - expected_x) < 1e-4f);
    assert(fabsf(env.idx.centroids[1] - expected_y) < 1e-4f);

    ivf_dyn_teardown(&env);
    printf("  PASS: test_dynamic_update_centroid_ema\n");
}

static void test_dynamic_split_cluster(void) {
    IvfDynTestEnv env;
    uint16_t dim = 2;
    uint32_t nlist = 2;
    ivf_dyn_setup(&env, dim, VEC_DIST_L2, nlist, 2, 64);

    float train_data[] = {
        0.0f, 0.0f,
        100.0f, 100.0f
    };
    assert(ivf_index_train(&env.idx, train_data, 2) == 0);
    env.idx.centroids[0] = 0.0f; env.idx.centroids[1] = 0.0f;
    env.idx.centroids[2] = 100.0f; env.idx.centroids[3] = 100.0f;

    /* Insert 30 vectors into cluster 0 with two sub-groups: around (0,0) and (20,20) */
    for (int i = 0; i < 15; i++) {
        float v[2] = {0.1f * (float)i, 0.1f * (float)i};
        assert(ivf_index_insert(&env.idx, v, NULL, NULL) == 0);
    }
    for (int i = 0; i < 15; i++) {
        float v[2] = {20.0f + 0.1f * (float)i, 20.0f + 0.1f * (float)i};
        assert(ivf_index_insert(&env.idx, v, NULL, NULL) == 0);
    }

    uint32_t count_before = flat_index_count(&env.idx.partitions[0]);
    assert(count_before == 30);

    /* Split cluster 0 */
    assert(ivf_dynamic_split_cluster(&env.idx, 0) == 0);

    /* nlist should have increased by 1 */
    assert(env.idx.nlist == 3);

    /* Both parts should now have vectors */
    uint32_t count_p0 = flat_index_count(&env.idx.partitions[0]);
    uint32_t count_p2 = flat_index_count(&env.idx.partitions[2]);
    assert(count_p0 > 0);
    assert(count_p2 > 0);
    assert(count_p0 + count_p2 == 30);

    /* Search still finds nearest vectors */
    float q[2] = {0.0f, 0.0f};
    VecResult results[3];
    uint32_t num_res = 0;
    assert(ivf_index_search(&env.idx, q, 3, NULL, NULL, results, &num_res) == 0);
    assert(num_res == 3);
    assert(results[0].distance < 0.1f);

    ivf_dyn_teardown(&env);
    printf("  PASS: test_dynamic_split_cluster (nlist 2 -> 3, balanced partition)\n");
}

static void test_dynamic_merge_and_compact(void) {
    IvfDynTestEnv env;
    uint16_t dim = 2;
    uint32_t nlist = 3;
    ivf_dyn_setup(&env, dim, VEC_DIST_L2, nlist, 3, 64);

    float train_data[] = {
        0.0f, 0.0f,
        1.0f, 1.0f,
        100.0f, 100.0f
    };
    assert(ivf_index_train(&env.idx, train_data, 3) == 0);
    env.idx.centroids[0] = 0.0f; env.idx.centroids[1] = 0.0f;
    env.idx.centroids[2] = 1.0f; env.idx.centroids[3] = 1.0f;
    env.idx.centroids[4] = 100.0f; env.idx.centroids[5] = 100.0f;

    /* Insert 10 vectors near cluster 0, 1 vector near cluster 1, and 1 near cluster 2 */
    for (int i = 0; i < 10; i++) {
        float v[2] = {0.05f * (float)i, 0.05f * (float)i};
        assert(ivf_index_insert(&env.idx, v, NULL, NULL) == 0);
    }
    float v_near1[2] = {1.05f, 1.05f};
    assert(ivf_index_insert(&env.idx, v_near1, NULL, NULL) == 0);
    float v_near2[2] = {100.0f, 100.0f};
    assert(ivf_index_insert(&env.idx, v_near2, NULL, NULL) == 0);

    assert(flat_index_count(&env.idx.partitions[0]) == 10);
    assert(flat_index_count(&env.idx.partitions[1]) == 1);
    assert(flat_index_count(&env.idx.partitions[2]) == 1);

    /* Merge cluster 1 into its nearest neighbor (cluster 0) */
    assert(ivf_dynamic_merge_cluster(&env.idx, 1) == 0);
    assert(flat_index_count(&env.idx.partitions[1]) == 0);
    assert(flat_index_count(&env.idx.partitions[0]) == 11);

    /* Compact index: remove the empty partition 1 */
    assert(ivf_dynamic_compact(&env.idx) == 0);
    assert(env.idx.nlist == 2);
    assert(flat_index_count(&env.idx.partitions[0]) == 11);

    ivf_dyn_teardown(&env);
    printf("  PASS: test_dynamic_merge_and_compact (nlist 3 -> 2 after merge + compact)\n");
}

static void test_dynamic_rebalance(void) {
    IvfDynTestEnv env;
    uint16_t dim = 2;
    uint32_t nlist = 3;
    ivf_dyn_setup(&env, dim, VEC_DIST_L2, nlist, 3, 64);

    float train_data[] = {
        0.0f, 0.0f,
        50.0f, 50.0f,
        100.0f, 100.0f
    };
    assert(ivf_index_train(&env.idx, train_data, 3) == 0);
    env.idx.centroids[0] = 0.0f; env.idx.centroids[1] = 0.0f;
    env.idx.centroids[2] = 50.0f; env.idx.centroids[3] = 50.0f;
    env.idx.centroids[4] = 100.0f; env.idx.centroids[5] = 100.0f;

    /* Intentionally skewed data: 40 vectors near cluster 0, 1 near cluster 2 */
    for (int i = 0; i < 40; i++) {
        float v[2] = {0.2f * (float)i, 0.2f * (float)i};
        assert(ivf_index_insert(&env.idx, v, NULL, NULL) == 0);
    }
    float v2[2] = {100.1f, 100.1f};
    assert(ivf_index_insert(&env.idx, v2, NULL, NULL) == 0);

    IvfDynamicConfig cfg;
    ivf_dynamic_config_default(&cfg);
    cfg.split_threshold = 1.5f;
    cfg.merge_threshold = 0.2f;
    cfg.min_cluster_size = 5;

    int ops = ivf_dynamic_rebalance(&env.idx, &cfg);
    assert(ops > 0);

    ivf_dyn_teardown(&env);
    printf("  PASS: test_dynamic_rebalance (performed %d rebalance ops)\n", ops);
}

static void test_dynamic_retrain(void) {
    IvfDynTestEnv env;
    uint16_t dim = 2;
    uint32_t nlist = 2;
    ivf_dyn_setup(&env, dim, VEC_DIST_L2, nlist, 2, 64);

    /* Initial training centered around (0,0) and (10,10) */
    float train_data[] = {
        0.0f, 0.0f,
        10.0f, 10.0f
    };
    assert(ivf_index_train(&env.idx, train_data, 2) == 0);

    /* Distribution drifts: all new data is at (50,50) and (100,100) */
    for (int i = 0; i < 20; i++) {
        float v[2] = {50.0f + 0.1f * (float)i, 50.0f};
        assert(ivf_index_insert(&env.idx, v, NULL, NULL) == 0);
    }
    for (int i = 0; i < 20; i++) {
        float v[2] = {100.0f + 0.1f * (float)i, 100.0f};
        assert(ivf_index_insert(&env.idx, v, NULL, NULL) == 0);
    }

    /* Retrain centroids */
    assert(ivf_dynamic_retrain(&env.idx) == 0);
    assert(env.idx.total_vectors == 40);

    /* Verify centroids shifted into new data territory [50, 100] */
    float c0_x = env.idx.centroids[0];
    float c1_x = env.idx.centroids[2];
    assert(c0_x >= 40.0f && c0_x <= 110.0f);
    assert(c1_x >= 40.0f && c1_x <= 110.0f);

    /* Search still finds nearest vectors */
    float q[2] = {50.0f, 50.0f};
    VecResult results[3];
    uint32_t num_res = 0;
    assert(ivf_index_search(&env.idx, q, 3, NULL, NULL, results, &num_res) == 0);
    assert(num_res == 3);
    assert(results[0].distance < 0.2f);

    ivf_dyn_teardown(&env);
    printf("  PASS: test_dynamic_retrain (centroids adapted to drifted distribution)\n");
}

/* ============================================================
 *  Main
 * ============================================================ */

int main(void) {
    printf("--- Running IVF Dynamic Re-Clustering Tests ---\n");

    test_dynamic_config_default();
    test_dynamic_health_check();
    test_dynamic_update_centroid_ema();
    test_dynamic_split_cluster();
    test_dynamic_merge_and_compact();
    test_dynamic_rebalance();
    test_dynamic_retrain();

    printf("=== All IVF Dynamic Tests PASSED! ===\n\n");
    return 0;
}
