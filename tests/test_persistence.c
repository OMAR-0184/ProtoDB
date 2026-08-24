#include "../src/pages/page.h"
#include "../src/pages/storage_mgr.h"
#include "../src/buffer/buffer_pool.h"
#include "../src/index/flat/flat_index.h"
#include "../src/index/ivf/ivf_index.h"
#include <stdio.h>
#include <assert.h>
#include <unistd.h>
#include <math.h>

#define TEST_DB "build/test_persistence.db"

static void test_flat_persistence(void) {
    unlink(TEST_DB);

    /* 1. Create and populate index */
    StorageManager sm;
    assert(storage_open(&sm, TEST_DB) == 0);
    BufferPool bp;
    bp_init(&bp, &sm, 16);

    FlatIndex idx;
    assert(flat_index_create(&idx, &bp, 4, VEC_DIST_L2) == 0);

    float v1[4] = {1.0, 1.0, 1.0, 1.0};
    float v2[4] = {2.0, 2.0, 2.0, 2.0};
    assert(flat_index_insert(&idx, v1, NULL, NULL) == 0);
    assert(flat_index_insert(&idx, v2, NULL, NULL) == 0);

    /* Search to verify */
    float q[4] = {2.1, 2.1, 2.1, 2.1};
    VecResult res[2];
    uint32_t n;
    assert(flat_index_search(&idx, q, 2, NULL, NULL, res, &n) == 0);
    assert(n == 2);
    float orig_dist = res[0].distance;

    /* Save index to storage */
    page_id_t root_pid;
    assert(flat_index_save(&idx, &root_pid) == 0);

    /* Save root PID to meta page */
    sm.meta.root_index_pid = root_pid;
    sm.meta.root_index_type = 1; /* flat */

    flat_index_destroy(&idx);
    bp_flush_all(&bp);
    bp_destroy(&bp);
    storage_close(&sm);

    /* 2. Reload and verify */
    StorageManager sm2;
    assert(storage_open(&sm2, TEST_DB) == 0);
    BufferPool bp2;
    bp_init(&bp2, &sm2, 16);

    assert(sm2.meta.root_index_type == 1);
    page_id_t loaded_root = sm2.meta.root_index_pid;

    FlatIndex loaded_idx;
    assert(flat_index_load(&loaded_idx, &bp2, loaded_root) == 0);

    assert(flat_index_count(&loaded_idx) == 2);
    
    assert(flat_index_search(&loaded_idx, q, 2, NULL, NULL, res, &n) == 0);
    assert(n == 2);
    assert(fabs(res[0].distance - orig_dist) < 1e-5);

    flat_index_destroy(&loaded_idx);
    bp_destroy(&bp2);
    storage_close(&sm2);
    unlink(TEST_DB);
    printf("  PASS: test_flat_persistence\n");
}

static void test_ivf_persistence(void) {
    unlink(TEST_DB);

    StorageManager sm;
    assert(storage_open(&sm, TEST_DB) == 0);
    BufferPool bp;
    bp_init(&bp, &sm, 16);

    IvfIndex idx;
    assert(ivf_index_create(&idx, &bp, 4, VEC_DIST_L2, 2, 2) == 0);

    float train[8] = {1, 1, 1, 1, 2, 2, 2, 2};
    assert(ivf_index_train(&idx, train, 2) == 0);

    float v1[4] = {1.0, 1.0, 1.0, 1.0};
    float v2[4] = {2.0, 2.0, 2.0, 2.0};
    assert(ivf_index_insert(&idx, v1, NULL, NULL) == 0);
    assert(ivf_index_insert(&idx, v2, NULL, NULL) == 0);

    float q[4] = {2.1, 2.1, 2.1, 2.1};
    VecResult res[2];
    uint32_t n;
    assert(ivf_index_search(&idx, q, 2, NULL, NULL, res, &n) == 0);
    assert(n == 2);
    float orig_dist = res[0].distance;

    page_id_t root_pid;
    assert(ivf_index_save(&idx, &root_pid) == 0);
    
    sm.meta.root_index_pid = root_pid;
    sm.meta.root_index_type = 2; /* ivf */

    ivf_index_destroy(&idx);
    assert(bp_flush_all(&bp) == 0);
    bp_destroy(&bp);
    storage_close(&sm);

    /* Reload */
    StorageManager sm2;
    assert(storage_open(&sm2, TEST_DB) == 0);
    BufferPool bp2;
    bp_init(&bp2, &sm2, 16);

    assert(sm2.meta.root_index_type == 2);
    page_id_t loaded_root = sm2.meta.root_index_pid;

    IvfIndex loaded_idx;
    assert(ivf_index_load(&loaded_idx, &bp2, loaded_root) == 0);

    assert(ivf_index_count(&loaded_idx) == 2);
    
    assert(ivf_index_search(&loaded_idx, q, 2, NULL, NULL, res, &n) == 0);
    assert(n == 2);
    assert(fabs(res[0].distance - orig_dist) < 1e-5);

    ivf_index_destroy(&loaded_idx);
    bp_destroy(&bp2);
    storage_close(&sm2);
    unlink(TEST_DB);
    printf("  PASS: test_ivf_persistence\n");
}

int main(void) {
    printf("\n=== Persistence Tests ===\n");
    test_flat_persistence();
    test_ivf_persistence();
    printf("All persistence tests passed.\n\n");
    return 0;
}
