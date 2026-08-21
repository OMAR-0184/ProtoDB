#include "../src/pages/storage_mgr.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>

static const char *TEST_DB = "build/protodb_bulk_test.db";

static void cleanup(void) {
    unlink(TEST_DB);
}

static void test_bulk_read(void) {
    cleanup();

    StorageManager sm;
    assert(storage_open(&sm, TEST_DB) == 0);

    /* Allocate 5 pages and write distinct data to each */
    page_id_t ids[5];
    for (int i = 0; i < 5; i++) {
        assert(storage_allocate_page(&sm, &ids[i]) == 0);

        Page p;
        page_init(&p, ids[i], PAGE_TYPE_DATA);

        char msg[32];
        snprintf(msg, sizeof(msg), "page-%d-data", i);
        uint16_t slot;
        assert(page_insert_record(&p, msg, (uint16_t)strlen(msg), &slot) == 0);
        assert(storage_write_page(&sm, ids[i], &p) == 0);
    }

    /* Bulk read all 5 pages in one call */
    Page bulk[5];
    assert(storage_read_pages(&sm, ids[0], 5, bulk) == 0);

    /* Verify each page has the correct data */
    for (int i = 0; i < 5; i++) {
        assert(bulk[i].id == ids[i]);
        assert(bulk[i].pin_count == 0);
        assert(bulk[i].is_dirty == false);

        uint8_t *out;
        uint16_t len;
        assert(page_get_record(&bulk[i], 0, &out, &len) == 0);

        char expected[32];
        snprintf(expected, sizeof(expected), "page-%d-data", i);
        assert(len == strlen(expected));
        assert(memcmp(out, expected, len) == 0);
    }

    assert(storage_close(&sm) == 0);
    cleanup();
    printf("  PASS: test_bulk_read\n");
}

static void test_bulk_read_subset(void) {
    cleanup();

    StorageManager sm;
    assert(storage_open(&sm, TEST_DB) == 0);

    page_id_t ids[4];
    for (int i = 0; i < 4; i++)
        assert(storage_allocate_page(&sm, &ids[i]) == 0);

    /* Write data to pages 2 and 3 (ids[1] and ids[2]) */
    for (int i = 1; i <= 2; i++) {
        Page p;
        page_init(&p, ids[i], PAGE_TYPE_DATA);
        char msg[16];
        snprintf(msg, sizeof(msg), "rec-%d", i);
        uint16_t slot;
        assert(page_insert_record(&p, msg, (uint16_t)strlen(msg), &slot) == 0);
        assert(storage_write_page(&sm, ids[i], &p) == 0);
    }

    /* Bulk read just the middle 2 pages */
    Page bulk[2];
    assert(storage_read_pages(&sm, ids[1], 2, bulk) == 0);
    assert(bulk[0].id == ids[1]);
    assert(bulk[1].id == ids[2]);

    uint8_t *out;
    uint16_t len;
    assert(page_get_record(&bulk[0], 0, &out, &len) == 0);
    assert(memcmp(out, "rec-1", 5) == 0);

    assert(storage_close(&sm) == 0);
    cleanup();
    printf("  PASS: test_bulk_read_subset\n");
}

static void test_bulk_read_out_of_bounds(void) {
    cleanup();

    StorageManager sm;
    assert(storage_open(&sm, TEST_DB) == 0);

    page_id_t p1;
    assert(storage_allocate_page(&sm, &p1) == 0);

    /* Try to read more pages than exist */
    Page bulk[10];
    assert(storage_read_pages(&sm, 0, 10, bulk) == -1);

    /* Zero count should fail */
    assert(storage_read_pages(&sm, 0, 0, bulk) == -1);

    assert(storage_close(&sm) == 0);
    cleanup();
    printf("  PASS: test_bulk_read_out_of_bounds\n");
}

static void test_vector_sized_record(void) {
    cleanup();

    StorageManager sm;
    assert(storage_open(&sm, TEST_DB) == 0);

    page_id_t pid;
    assert(storage_allocate_page(&sm, &pid) == 0);

    Page p;
    page_init(&p, pid, PAGE_TYPE_VECTOR);

    /*
     * Simulate a 1536-dim float32 vector (OpenAI ada-002 size).
     * 1536 * 4 bytes = 6144 bytes.
     * With 8 KB pages this fits; with 4 KB pages it would not.
     */
    const uint32_t dim = 1536;
    size_t vec_bytes = dim * sizeof(float);
    float *vec = malloc(vec_bytes);
    assert(vec != NULL);

    for (uint32_t i = 0; i < dim; i++)
        vec[i] = (float)i * 0.001f;

    uint16_t slot;
    int rc = page_insert_record(&p, vec, (uint16_t)vec_bytes, &slot);
    assert(rc == 0);

    /* Write to disk and read back */
    assert(storage_write_page(&sm, pid, &p) == 0);

    Page loaded;
    assert(storage_read_page(&sm, pid, &loaded) == 0);

    uint8_t *out;
    uint16_t len;
    assert(page_get_record(&loaded, slot, &out, &len) == 0);
    assert(len == vec_bytes);

    /* Verify the floats survived the round-trip */
    float *loaded_vec = (float *)out;
    for (uint32_t i = 0; i < dim; i++)
        assert(loaded_vec[i] == vec[i]);

    /* Verify page type persisted */
    PageHeader *hdr = page_get_header(&loaded);
    assert(hdr->page_type == PAGE_TYPE_VECTOR);

    free(vec);
    assert(storage_close(&sm) == 0);
    cleanup();
    printf("  PASS: test_vector_sized_record (1536-dim, %zu bytes)\n", vec_bytes);
}

static void test_multiple_vectors_per_page(void) {
    cleanup();

    StorageManager sm;
    assert(storage_open(&sm, TEST_DB) == 0);

    page_id_t pid;
    assert(storage_allocate_page(&sm, &pid) == 0);

    Page p;
    page_init(&p, pid, PAGE_TYPE_VECTOR);

    /*
     * 128-dim vectors = 512 bytes each.
     * With 8 KB pages (~8176 usable), we should fit multiple.
     */
    const uint32_t dim = 128;
    size_t vec_bytes = dim * sizeof(float);
    float vec[128];

    int count = 0;
    uint16_t slot;
    while (1) {
        for (uint32_t i = 0; i < dim; i++)
            vec[i] = (float)(count * 1000 + i);

        if (page_insert_record(&p, vec, (uint16_t)vec_bytes, &slot) != 0)
            break;
        count++;
    }

    /* 8 KB page should fit many 512-byte vectors */
    assert(count >= 10);

    /* Verify we can read back the first and last */
    uint8_t *out;
    uint16_t len;
    assert(page_get_record(&p, 0, &out, &len) == 0);
    float *first = (float *)out;
    assert(first[0] == 0.0f);
    assert(first[127] == 127.0f);

    assert(page_get_record(&p, (uint16_t)(count - 1), &out, &len) == 0);
    float *last = (float *)out;
    assert(last[0] == (float)((count - 1) * 1000));

    assert(storage_close(&sm) == 0);
    cleanup();
    printf("  PASS: test_multiple_vectors_per_page (%d x %u-dim vectors)\n", count, dim);
}

static void test_8kb_page_alignment(void) {
    cleanup();

    StorageManager sm;
    assert(storage_open(&sm, TEST_DB) == 0);

    /* Verify PAGE_SIZE is 8192 */
    assert(PAGE_SIZE == 8192);

    page_id_t p1, p2;
    assert(storage_allocate_page(&sm, &p1) == 0);
    assert(storage_allocate_page(&sm, &p2) == 0);

    /* After meta + 2 data pages, file should be exactly 3 * 8192 = 24576 bytes */
    assert(storage_num_pages(&sm) == 3);

    assert(storage_close(&sm) == 0);

    /* Verify file size on disk */
    FILE *f = fopen(TEST_DB, "rb");
    assert(f != NULL);
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fclose(f);
    assert(size == 3 * PAGE_SIZE);

    cleanup();
    printf("  PASS: test_8kb_page_alignment (file = %ld bytes)\n", size);
}

int main(void) {
    printf("=== Storage Bulk & Vector Tests ===\n");
    test_bulk_read();
    test_bulk_read_subset();
    test_bulk_read_out_of_bounds();
    test_vector_sized_record();
    test_multiple_vectors_per_page();
    test_8kb_page_alignment();
    printf("All storage bulk & vector tests passed.\n\n");
    return 0;
}
