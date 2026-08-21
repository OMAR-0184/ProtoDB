#include "../src/pages/storage_mgr.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>

static const char *TEST_DB = "build/protodb_test.db";

static void cleanup(void) {
    unlink(TEST_DB);
}

static void test_create_and_reopen(void) {
    cleanup();

    StorageManager sm;
    assert(storage_open(&sm, TEST_DB) == 0);
    assert(sm.meta.magic == PROTO_MAGIC);
    assert(sm.meta.page_count == 1);
    assert(sm.meta.free_list_head == INVALID_PAGE_ID);
    assert(storage_close(&sm) == 0);

    assert(storage_open(&sm, TEST_DB) == 0);
    assert(sm.meta.magic == PROTO_MAGIC);
    assert(sm.meta.page_count == 1);
    assert(storage_close(&sm) == 0);

    cleanup();
    printf("  PASS: test_create_and_reopen\n");
}

static void test_allocate_pages(void) {
    cleanup();

    StorageManager sm;
    assert(storage_open(&sm, TEST_DB) == 0);

    page_id_t p1, p2, p3;
    assert(storage_allocate_page(&sm, &p1) == 0);
    assert(p1 == 1);
    assert(storage_allocate_page(&sm, &p2) == 0);
    assert(p2 == 2);
    assert(storage_allocate_page(&sm, &p3) == 0);
    assert(p3 == 3);
    assert(storage_num_pages(&sm) == 4);

    assert(storage_close(&sm) == 0);
    cleanup();
    printf("  PASS: test_allocate_pages\n");
}

static void test_read_write(void) {
    cleanup();

    StorageManager sm;
    assert(storage_open(&sm, TEST_DB) == 0);

    page_id_t pid;
    assert(storage_allocate_page(&sm, &pid) == 0);

    Page p;
    page_init(&p, pid, PAGE_TYPE_DATA);

    uint16_t slot;
    const char *msg = "persistent record";
    assert(page_insert_record(&p, msg, (uint16_t)strlen(msg), &slot) == 0);
    assert(storage_write_page(&sm, pid, &p) == 0);

    Page p2;
    memset(&p2, 0, sizeof(Page));
    assert(storage_read_page(&sm, pid, &p2) == 0);

    uint8_t *out;
    uint16_t len;
    assert(page_get_record(&p2, slot, &out, &len) == 0);
    assert(len == strlen(msg));
    assert(memcmp(out, msg, len) == 0);

    assert(storage_close(&sm) == 0);
    cleanup();
    printf("  PASS: test_read_write\n");
}

static void test_deallocate_and_reuse(void) {
    cleanup();

    StorageManager sm;
    assert(storage_open(&sm, TEST_DB) == 0);

    page_id_t p1, p2, p3;
    assert(storage_allocate_page(&sm, &p1) == 0);
    assert(storage_allocate_page(&sm, &p2) == 0);
    assert(storage_allocate_page(&sm, &p3) == 0);

    assert(storage_deallocate_page(&sm, p2) == 0);
    assert(sm.meta.free_list_head == p2);

    page_id_t reused;
    assert(storage_allocate_page(&sm, &reused) == 0);
    assert(reused == p2);
    assert(sm.meta.free_list_head == INVALID_PAGE_ID);

    assert(storage_close(&sm) == 0);
    cleanup();
    printf("  PASS: test_deallocate_and_reuse\n");
}

static void test_free_list_chain(void) {
    cleanup();

    StorageManager sm;
    assert(storage_open(&sm, TEST_DB) == 0);

    page_id_t pages[5];
    for (int i = 0; i < 5; i++)
        assert(storage_allocate_page(&sm, &pages[i]) == 0);

    assert(storage_deallocate_page(&sm, pages[1]) == 0);
    assert(storage_deallocate_page(&sm, pages[3]) == 0);

    page_id_t r1, r2;
    assert(storage_allocate_page(&sm, &r1) == 0);
    assert(r1 == pages[3]);
    assert(storage_allocate_page(&sm, &r2) == 0);
    assert(r2 == pages[1]);

    page_id_t fresh;
    assert(storage_allocate_page(&sm, &fresh) == 0);
    assert(fresh == 6);

    assert(storage_close(&sm) == 0);
    cleanup();
    printf("  PASS: test_free_list_chain\n");
}

static void test_persistence_across_close(void) {
    cleanup();

    StorageManager sm;
    assert(storage_open(&sm, TEST_DB) == 0);

    page_id_t p1, p2;
    assert(storage_allocate_page(&sm, &p1) == 0);
    assert(storage_allocate_page(&sm, &p2) == 0);

    Page page;
    page_init(&page, p1, PAGE_TYPE_DATA);
    const char *data = "survives reopen";
    uint16_t slot;
    assert(page_insert_record(&page, data, (uint16_t)strlen(data), &slot) == 0);
    assert(storage_write_page(&sm, p1, &page) == 0);

    assert(storage_deallocate_page(&sm, p2) == 0);
    assert(storage_close(&sm) == 0);

    assert(storage_open(&sm, TEST_DB) == 0);
    assert(sm.meta.page_count == 3);
    assert(sm.meta.free_list_head == p2);

    Page loaded;
    assert(storage_read_page(&sm, p1, &loaded) == 0);
    uint8_t *out;
    uint16_t len;
    assert(page_get_record(&loaded, slot, &out, &len) == 0);
    assert(memcmp(out, data, len) == 0);

    assert(storage_close(&sm) == 0);
    cleanup();
    printf("  PASS: test_persistence_across_close\n");
}

static void test_cannot_deallocate_meta(void) {
    cleanup();

    StorageManager sm;
    assert(storage_open(&sm, TEST_DB) == 0);
    assert(storage_deallocate_page(&sm, META_PAGE_ID) == -1);
    assert(storage_close(&sm) == 0);

    cleanup();
    printf("  PASS: test_cannot_deallocate_meta\n");
}

static void test_read_out_of_bounds(void) {
    cleanup();

    StorageManager sm;
    assert(storage_open(&sm, TEST_DB) == 0);

    Page p;
    assert(storage_read_page(&sm, 999, &p) == -1);

    assert(storage_close(&sm) == 0);
    cleanup();
    printf("  PASS: test_read_out_of_bounds\n");
}

int main(void) {
    printf("=== Storage Manager Tests ===\n");
    test_create_and_reopen();
    test_allocate_pages();
    test_read_write();
    test_deallocate_and_reuse();
    test_free_list_chain();
    test_persistence_across_close();
    test_cannot_deallocate_meta();
    test_read_out_of_bounds();
    printf("All storage manager tests passed.\n\n");
    return 0;
}
