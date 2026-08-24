#include "../src/pages/storage_mgr.h"
#include "../src/buffer/buffer_pool.h"
#include "../src/metadata/metadata_store.h"
#include "../src/index/flat/flat_index.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>

static const char *TEST_DB = "build/protodb_metadata_test.db";

static void cleanup(void) {
    unlink(TEST_DB);
}

static void test_metadata_crud(void) {
    cleanup();

    StorageManager sm;
    assert(storage_open(&sm, TEST_DB) == 0);

    BufferPool bp;
    assert(bp_init(&bp, &sm, 16) == 0);

    MetadataStore ms;
    assert(meta_store_init(&ms, &bp) == 0);

    /* Set */
    assert(meta_store_set(&ms, 1, 0, "{\"category\": \"document\", \"id\": 42}") == 0);
    assert(meta_store_set(&ms, 1, 1, "{\"category\": \"image\", \"id\": 99}") == 0);
    assert(meta_store_set(&ms, 2, 5, "{\"category\": \"document\", \"id\": 100}") == 0);

    assert(ms.count == 3);

    /* Get */
    const char *m1 = meta_store_get(&ms, 1, 0);
    assert(m1 != NULL);
    assert(strcmp(m1, "{\"category\": \"document\", \"id\": 42}") == 0);

    const char *m2 = meta_store_get(&ms, 1, 1);
    assert(m2 != NULL);
    assert(strcmp(m2, "{\"category\": \"image\", \"id\": 99}") == 0);

    const char *m3 = meta_store_get(&ms, 5, 5);
    assert(m3 == NULL);

    /* Update */
    assert(meta_store_set(&ms, 1, 1, "updated") == 0);
    assert(strcmp(meta_store_get(&ms, 1, 1), "updated") == 0);
    assert(ms.count == 3);

    /* Delete */
    assert(meta_store_delete(&ms, 1, 0) == 0);
    assert(ms.count == 2);
    assert(meta_store_get(&ms, 1, 0) == NULL);

    meta_store_destroy(&ms);
    bp_destroy(&bp);
    storage_close(&sm);
    cleanup();

    printf("  PASS: test_metadata_crud\n");
}

static void test_metadata_persistence(void) {
    cleanup();

    /* 1. Create and save */
    StorageManager sm;
    assert(storage_open(&sm, TEST_DB) == 0);

    BufferPool bp;
    assert(bp_init(&bp, &sm, 16) == 0);

    MetadataStore ms;
    assert(meta_store_init(&ms, &bp) == 0);

    /* Insert 100 items */
    for (int i = 0; i < 100; i++) {
        char buf[64];
        snprintf(buf, sizeof(buf), "payload-%d", i);
        assert(meta_store_set(&ms, (page_id_t)(i % 10 + 1), (uint16_t)i, buf) == 0);
    }

    page_id_t root_pid;
    assert(meta_store_save(&ms, &root_pid) == 0);
    
    sm.meta.root_meta_store_pid = root_pid;

    meta_store_destroy(&ms);
    bp_destroy(&bp);
    storage_close(&sm);

    /* 2. Re-open and load */
    StorageManager sm2;
    assert(storage_open(&sm2, TEST_DB) == 0);

    BufferPool bp2;
    assert(bp_init(&bp2, &sm2, 16) == 0);

    MetadataStore ms2;
    assert(meta_store_load(&ms2, &bp2, sm2.meta.root_meta_store_pid) == 0);

    assert(ms2.count == 100);
    
    for (int i = 0; i < 100; i++) {
        char buf[64];
        snprintf(buf, sizeof(buf), "payload-%d", i);
        const char *val = meta_store_get(&ms2, (page_id_t)(i % 10 + 1), (uint16_t)i);
        assert(val != NULL);
        assert(strcmp(val, buf) == 0);
    }

    meta_store_destroy(&ms2);
    bp_destroy(&bp2);
    storage_close(&sm2);
    cleanup();

    printf("  PASS: test_metadata_persistence\n");
}

/* A filter function that matches a string inside the JSON */
static bool category_filter(page_id_t pid, uint16_t slot, void *user_data) {
    MetadataStore *ms = (MetadataStore *)user_data;
    const char *meta = meta_store_get(ms, pid, slot);
    if (!meta) return false;
    
    return strstr(meta, "\"category\": \"document\"") != NULL;
}

static void test_filtered_search(void) {
    cleanup();

    StorageManager sm;
    assert(storage_open(&sm, TEST_DB) == 0);
    BufferPool bp;
    assert(bp_init(&bp, &sm, 16) == 0);

    FlatIndex idx;
    assert(flat_index_create(&idx, &bp, 4, VEC_DIST_L2) == 0);

    MetadataStore ms;
    assert(meta_store_init(&ms, &bp) == 0);

    float v1[4] = {1.0, 0.0, 0.0, 0.0};
    float v2[4] = {1.1, 0.0, 0.0, 0.0};
    float v3[4] = {1.2, 0.0, 0.0, 0.0};

    page_id_t p;
    uint16_t s;

    assert(flat_index_insert(&idx, v1, &p, &s) == 0);
    meta_store_set(&ms, p, s, "{\"category\": \"image\"}");

    assert(flat_index_insert(&idx, v2, &p, &s) == 0);
    meta_store_set(&ms, p, s, "{\"category\": \"document\"}");

    assert(flat_index_insert(&idx, v3, &p, &s) == 0);
    meta_store_set(&ms, p, s, "{\"category\": \"document\"}");

    float query[4] = {1.0, 0.0, 0.0, 0.0};
    VecResult results[3];
    uint32_t num = 0;

    /* Without filter, v1 is closest (dist=0) */
    assert(flat_index_search(&idx, query, 3, NULL, NULL, results, &num) == 0);
    assert(num == 3);
    assert(results[0].distance == 0.0f); /* v1 */

    /* With filter for "document", v1 is skipped, v2 is closest */
    assert(flat_index_search(&idx, query, 3, category_filter, &ms, results, &num) == 0);
    assert(num == 2);
    /* Dist to v2 (1.1) is 0.1^2 = 0.01 */
    assert(results[0].distance > 0.009f && results[0].distance < 0.011f); 

    meta_store_destroy(&ms);
    flat_index_destroy(&idx);
    bp_destroy(&bp);
    storage_close(&sm);
    cleanup();

    printf("  PASS: test_filtered_search\n");
}

int main(void) {
    printf("=== Metadata & Filtering Tests ===\n");
    test_metadata_crud();
    test_metadata_persistence();
    test_filtered_search();
    printf("All metadata & filtering tests passed.\n\n");
    return 0;
}
