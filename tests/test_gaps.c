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

#define TEST_DB_PATH "/tmp/test_gaps.db"

static void cleanup(void) { unlink(TEST_DB_PATH); }

/* ============================================================
 *  vec_page_compact tests
 * ============================================================ */

static void test_vec_page_compact_basic(void) {
    cleanup();
    StorageManager sm;
    assert(storage_open(&sm, TEST_DB_PATH) == 0);
    BufferPool bp;
    assert(bp_init(&bp, &sm, 16) == 0);

    page_id_t pid;
    Page *p = bp_new_page(&bp, &pid);
    assert(p != NULL);
    vec_page_init(p, pid, 4);

    /* Insert 5 vectors */
    float v0[4] = {0, 0, 0, 0};
    float v1[4] = {1, 1, 1, 1};
    float v2[4] = {2, 2, 2, 2};
    float v3[4] = {3, 3, 3, 3};
    float v4[4] = {4, 4, 4, 4};
    assert(vec_page_append(p, v0) == 0);
    assert(vec_page_append(p, v1) == 0);
    assert(vec_page_append(p, v2) == 0);
    assert(vec_page_append(p, v3) == 0);
    assert(vec_page_append(p, v4) == 0);
    assert(vec_page_count(p) == 5);

    /* Delete indices 1 and 3 */
    assert(vec_page_delete(p, 1) == 0);
    assert(vec_page_delete(p, 3) == 0);
    assert(vec_page_live_count(p) == 3);

    /* Compact */
    uint16_t reclaimed = vec_page_compact(p);
    assert(reclaimed == 2);
    assert(vec_page_count(p) == 3);
    assert(vec_page_live_count(p) == 3);

    /* Verify the remaining vectors are v0, v2, v4 */
    const float *r0 = vec_page_get(p, 0);
    const float *r1 = vec_page_get(p, 1);
    const float *r2 = vec_page_get(p, 2);
    assert(r0[0] == 0.0f);
    assert(r1[0] == 2.0f);
    assert(r2[0] == 4.0f);

    /* No deletions remain */
    assert(!vec_page_is_deleted(p, 0));
    assert(!vec_page_is_deleted(p, 1));
    assert(!vec_page_is_deleted(p, 2));

    bp_unpin_page(&bp, pid, false);
    bp_destroy(&bp);
    storage_close(&sm);
    cleanup();

    printf("  PASS: test_vec_page_compact_basic\n");
}

static void test_vec_page_compact_all_deleted(void) {
    cleanup();
    StorageManager sm;
    assert(storage_open(&sm, TEST_DB_PATH) == 0);
    BufferPool bp;
    assert(bp_init(&bp, &sm, 16) == 0);

    page_id_t pid;
    Page *p = bp_new_page(&bp, &pid);
    assert(p != NULL);
    vec_page_init(p, pid, 4);

    float v[4] = {1, 2, 3, 4};
    assert(vec_page_append(p, v) == 0);
    assert(vec_page_append(p, v) == 0);
    assert(vec_page_append(p, v) == 0);

    assert(vec_page_delete(p, 0) == 0);
    assert(vec_page_delete(p, 1) == 0);
    assert(vec_page_delete(p, 2) == 0);

    uint16_t reclaimed = vec_page_compact(p);
    assert(reclaimed == 3);
    assert(vec_page_count(p) == 0);
    assert(vec_page_live_count(p) == 0);

    bp_unpin_page(&bp, pid, false);
    bp_destroy(&bp);
    storage_close(&sm);
    cleanup();

    printf("  PASS: test_vec_page_compact_all_deleted\n");
}

static void test_vec_page_compact_none_deleted(void) {
    cleanup();
    StorageManager sm;
    assert(storage_open(&sm, TEST_DB_PATH) == 0);
    BufferPool bp;
    assert(bp_init(&bp, &sm, 16) == 0);

    page_id_t pid;
    Page *p = bp_new_page(&bp, &pid);
    assert(p != NULL);
    vec_page_init(p, pid, 4);

    float v[4] = {1, 2, 3, 4};
    assert(vec_page_append(p, v) == 0);
    assert(vec_page_append(p, v) == 0);

    uint16_t reclaimed = vec_page_compact(p);
    assert(reclaimed == 0);
    assert(vec_page_count(p) == 2);

    bp_unpin_page(&bp, pid, false);
    bp_destroy(&bp);
    storage_close(&sm);
    cleanup();

    printf("  PASS: test_vec_page_compact_none_deleted\n");
}

/* ============================================================
 *  flat_index_compact tests
 * ============================================================ */

static void test_flat_compact(void) {
    cleanup();
    StorageManager sm;
    assert(storage_open(&sm, TEST_DB_PATH) == 0);
    BufferPool bp;
    assert(bp_init(&bp, &sm, 32) == 0);

    FlatIndex idx;
    assert(flat_index_create(&idx, &bp, 4, VEC_DIST_L2) == 0);

    /* Insert 10 vectors */
    page_id_t pids[10];
    uint16_t slots[10];
    for (int i = 0; i < 10; i++) {
        float v[4] = {(float)i, 0, 0, 0};
        assert(flat_index_insert(&idx, v, &pids[i], &slots[i]) == 0);
    }
    assert(flat_index_count(&idx) == 10);

    /* Delete half */
    for (int i = 0; i < 10; i += 2) {
        assert(flat_index_delete(&idx, pids[i], slots[i]) == 0);
    }
    assert(flat_index_count(&idx) == 5);

    /* Compact */
    uint32_t reclaimed = 0;
    assert(flat_index_compact(&idx, &reclaimed) == 0);
    assert(reclaimed == 5);

    /* Search should still work for the remaining vectors */
    float query[4] = {1.0f, 0, 0, 0};
    VecResult results[5];
    uint32_t num = 0;
    assert(flat_index_search(&idx, query, 5, NULL, NULL, results, &num) == 0);
    assert(num == 5);
    /* Closest should be v1 (distance 0) */
    assert(results[0].distance < 1e-6f);

    flat_index_destroy(&idx);
    bp_destroy(&bp);
    storage_close(&sm);
    cleanup();

    printf("  PASS: test_flat_compact\n");
}

/* ============================================================
 *  flat_index_insert_batch tests
 * ============================================================ */

static void test_flat_insert_batch(void) {
    cleanup();
    StorageManager sm;
    assert(storage_open(&sm, TEST_DB_PATH) == 0);
    BufferPool bp;
    assert(bp_init(&bp, &sm, 32) == 0);

    FlatIndex idx;
    assert(flat_index_create(&idx, &bp, 4, VEC_DIST_L2) == 0);

    /* Create 100 vectors */
    uint32_t n = 100;
    float *vecs = malloc(n * 4 * sizeof(float));
    assert(vecs != NULL);
    for (uint32_t i = 0; i < n; i++) {
        vecs[i * 4 + 0] = (float)i;
        vecs[i * 4 + 1] = 0.0f;
        vecs[i * 4 + 2] = 0.0f;
        vecs[i * 4 + 3] = 0.0f;
    }

    assert(flat_index_insert_batch(&idx, vecs, n) == 0);
    assert(flat_index_count(&idx) == n);

    /* Search for the first vector */
    float query[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    VecResult results[1];
    uint32_t num = 0;
    assert(flat_index_search(&idx, query, 1, NULL, NULL, results, &num) == 0);
    assert(num == 1);
    assert(results[0].distance < 1e-6f);

    /* Search for last vector */
    float query2[4] = {99.0f, 0.0f, 0.0f, 0.0f};
    assert(flat_index_search(&idx, query2, 1, NULL, NULL, results, &num) == 0);
    assert(num == 1);
    assert(results[0].distance < 1e-6f);

    free(vecs);
    flat_index_destroy(&idx);
    bp_destroy(&bp);
    storage_close(&sm);
    cleanup();

    printf("  PASS: test_flat_insert_batch\n");
}

static void test_flat_insert_batch_spanning_pages(void) {
    cleanup();
    StorageManager sm;
    assert(storage_open(&sm, TEST_DB_PATH) == 0);
    BufferPool bp;
    assert(bp_init(&bp, &sm, 64) == 0);

    FlatIndex idx;
    assert(flat_index_create(&idx, &bp, 4, VEC_DIST_L2) == 0);

    /* 502 vectors per page with dim=4, so 1000 vectors should span 2 pages */
    uint32_t n = 1000;
    float *vecs = malloc(n * 4 * sizeof(float));
    assert(vecs != NULL);
    for (uint32_t i = 0; i < n; i++) {
        vecs[i * 4] = (float)i;
        vecs[i * 4 + 1] = 0;
        vecs[i * 4 + 2] = 0;
        vecs[i * 4 + 3] = 0;
    }

    assert(flat_index_insert_batch(&idx, vecs, n) == 0);
    assert(flat_index_count(&idx) == n);
    assert(idx.num_pages >= 2); /* Should span multiple pages */

    free(vecs);
    flat_index_destroy(&idx);
    bp_destroy(&bp);
    storage_close(&sm);
    cleanup();

    printf("  PASS: test_flat_insert_batch_spanning_pages\n");
}

/* ============================================================
 *  ivf_index_compact tests
 * ============================================================ */

static void test_ivf_compact(void) {
    cleanup();
    StorageManager sm;
    assert(storage_open(&sm, TEST_DB_PATH) == 0);
    BufferPool bp;
    assert(bp_init(&bp, &sm, 128) == 0);

    IvfIndex idx;
    assert(ivf_index_create(&idx, &bp, 4, VEC_DIST_L2, 4, 4) == 0);

    /* Generate training vectors */
    uint32_t ntrain = 100;
    float *train = malloc(ntrain * 4 * sizeof(float));
    assert(train != NULL);
    srand(42);
    for (uint32_t i = 0; i < ntrain; i++) {
        train[i * 4 + 0] = (float)(rand() % 100);
        train[i * 4 + 1] = (float)(rand() % 100);
        train[i * 4 + 2] = (float)(rand() % 100);
        train[i * 4 + 3] = (float)(rand() % 100);
    }
    assert(ivf_index_train(&idx, train, ntrain) == 0);

    /* Insert 50 vectors */
    page_id_t pids[50];
    uint16_t slots[50];
    for (int i = 0; i < 50; i++) {
        float v[4] = {(float)(rand() % 100), (float)(rand() % 100),
                      (float)(rand() % 100), (float)(rand() % 100)};
        assert(ivf_index_insert(&idx, v, &pids[i], &slots[i]) == 0);
    }
    assert(ivf_index_count(&idx) == 50);

    /* Delete 25 */
    for (int i = 0; i < 50; i += 2) {
        assert(ivf_index_delete(&idx, pids[i], slots[i]) == 0);
    }
    assert(ivf_index_count(&idx) == 25);

    /* Compact */
    uint32_t reclaimed = 0;
    assert(ivf_index_compact(&idx, &reclaimed) == 0);
    assert(reclaimed == 25);

    free(train);
    ivf_index_destroy(&idx);
    bp_destroy(&bp);
    storage_close(&sm);
    cleanup();

    printf("  PASS: test_ivf_compact\n");
}

/* ============================================================
 *  ivf_index_insert_batch tests
 * ============================================================ */

static void test_ivf_insert_batch(void) {
    cleanup();
    StorageManager sm;
    assert(storage_open(&sm, TEST_DB_PATH) == 0);
    BufferPool bp;
    assert(bp_init(&bp, &sm, 128) == 0);

    IvfIndex idx;
    assert(ivf_index_create(&idx, &bp, 4, VEC_DIST_L2, 4, 4) == 0);

    /* Training data */
    uint32_t ntrain = 100;
    float *train = malloc(ntrain * 4 * sizeof(float));
    assert(train != NULL);
    srand(42);
    for (uint32_t i = 0; i < ntrain; i++) {
        train[i * 4 + 0] = (float)(rand() % 100);
        train[i * 4 + 1] = (float)(rand() % 100);
        train[i * 4 + 2] = (float)(rand() % 100);
        train[i * 4 + 3] = (float)(rand() % 100);
    }
    assert(ivf_index_train(&idx, train, ntrain) == 0);

    /* Batch insert 200 vectors */
    uint32_t n = 200;
    float *vecs = malloc(n * 4 * sizeof(float));
    assert(vecs != NULL);
    for (uint32_t i = 0; i < n; i++) {
        vecs[i * 4 + 0] = (float)(rand() % 100);
        vecs[i * 4 + 1] = (float)(rand() % 100);
        vecs[i * 4 + 2] = (float)(rand() % 100);
        vecs[i * 4 + 3] = (float)(rand() % 100);
    }

    assert(ivf_index_insert_batch(&idx, vecs, n) == 0);
    assert(ivf_index_count(&idx) == n);

    /* Verify search still works */
    float query[4] = {50.0f, 50.0f, 50.0f, 50.0f};
    VecResult results[5];
    uint32_t num = 0;
    assert(ivf_index_search(&idx, query, 5, NULL, NULL, results, &num) == 0);
    assert(num == 5);

    free(train);
    free(vecs);
    ivf_index_destroy(&idx);
    bp_destroy(&bp);
    storage_close(&sm);
    cleanup();

    printf("  PASS: test_ivf_insert_batch\n");
}

/* ============================================================
 *  bp_destroy WAL integration test
 * ============================================================ */

static void test_bp_destroy_wal_integration(void) {
    cleanup();
    const char *wal_path = "/tmp/test_gaps.db.wal";
    unlink(wal_path);

    StorageManager sm;
    assert(storage_open(&sm, TEST_DB_PATH) == 0);
    BufferPool bp;
    assert(bp_init(&bp, &sm, 4) == 0);

    Wal wal;
    assert(wal_open(&wal, TEST_DB_PATH, &sm) == 0);
    bp.wal = &wal;

    /* Create and dirty a page */
    page_id_t pid;
    Page *p = bp_new_page(&bp, &pid);
    assert(p != NULL);
    memset(p->data + PAGE_DATA_OFFSET, 0xAB, 100);
    bp_unpin_page(&bp, pid, true);

    /* bp_destroy should log to WAL before flushing */
    bp_destroy(&bp);

    /* WAL should have records (at least the dirty page we just created) */
    assert(wal.num_records > 0);

    wal_close(&wal);
    storage_close(&sm);
    cleanup();
    unlink(wal_path);

    printf("  PASS: test_bp_destroy_wal_integration\n");
}

/* ============================================================
 *  main
 * ============================================================ */

int main(void) {
    printf("=== Gap Fix Tests ===\n");

    /* vec_page_compact */
    test_vec_page_compact_basic();
    test_vec_page_compact_all_deleted();
    test_vec_page_compact_none_deleted();

    /* flat_index_compact */
    test_flat_compact();

    /* flat_index_insert_batch */
    test_flat_insert_batch();
    test_flat_insert_batch_spanning_pages();

    /* ivf_index_compact */
    test_ivf_compact();

    /* ivf_index_insert_batch */
    test_ivf_insert_batch();

    /* bp_destroy WAL */
    test_bp_destroy_wal_integration();

    printf("All gap fix tests passed.\n\n");
    return 0;
}
