#include "../src/buffer/buffer_pool.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>
#include <pthread.h>

static const char *TEST_DB = "build/protodb_bp_test.db";

static void cleanup(void) {
    unlink(TEST_DB);
}

/* Helper: open storage + init buffer pool */
static void setup(StorageManager *sm, BufferPool *bp, uint32_t nframes) {
    cleanup();
    assert(storage_open(sm, TEST_DB) == 0);
    assert(bp_init(bp, sm, nframes) == 0);
}

static void teardown(StorageManager *sm, BufferPool *bp) {
    bp_destroy(bp);
    storage_close(sm);
    cleanup();
}

static void test_fetch_and_unpin(void) {
    StorageManager sm;
    BufferPool bp;
    setup(&sm, &bp, 4);

    /* Allocate a page on disk, write data, then fetch through buffer pool */
    page_id_t pid;
    assert(storage_allocate_page(&sm, &pid) == 0);

    Page tmp;
    page_init(&tmp, pid, PAGE_TYPE_DATA);
    uint16_t slot;
    assert(page_insert_record(&tmp, "hello", 5, &slot) == 0);
    assert(storage_write_page(&sm, pid, &tmp) == 0);

    /* Fetch through buffer pool */
    Page *p = bp_fetch_page(&bp, pid);
    assert(p != NULL);
    assert(p->id == pid);
    assert(p->pin_count == 1);

    uint8_t *out;
    uint16_t len;
    assert(page_get_record(p, slot, &out, &len) == 0);
    assert(len == 5);
    assert(memcmp(out, "hello", 5) == 0);

    /* Unpin */
    assert(bp_unpin_page(&bp, pid, false) == 0);
    assert(p->pin_count == 0);

    teardown(&sm, &bp);
    printf("  PASS: test_fetch_and_unpin\n");
}

static void test_fetch_same_page_twice(void) {
    StorageManager sm;
    BufferPool bp;
    setup(&sm, &bp, 4);

    page_id_t pid;
    Page *p1 = bp_new_page(&bp, &pid);
    assert(p1 != NULL);
    assert(p1->pin_count == 1);

    /* Fetch same page again — should return same pointer, bump pin count */
    Page *p2 = bp_fetch_page(&bp, pid);
    assert(p2 == p1);
    assert(p2->pin_count == 2);

    assert(bp_unpin_page(&bp, pid, false) == 0);
    assert(bp_unpin_page(&bp, pid, false) == 0);
    assert(p1->pin_count == 0);

    teardown(&sm, &bp);
    printf("  PASS: test_fetch_same_page_twice\n");
}

static void test_dirty_flush_on_eviction(void) {
    StorageManager sm;
    BufferPool bp;
    setup(&sm, &bp, 2);  /* only 2 frames */

    /* Fill both frames */
    page_id_t id1, id2;
    Page *p1 = bp_new_page(&bp, &id1);
    assert(p1 != NULL);
    uint16_t slot;
    assert(page_insert_record(p1, "persist-me", 10, &slot) == 0);
    assert(bp_unpin_page(&bp, id1, true) == 0);  /* dirty */

    Page *p2 = bp_new_page(&bp, &id2);
    assert(p2 != NULL);
    assert(bp_unpin_page(&bp, id2, false) == 0);

    /* Fetch a 3rd page — forces eviction of one of the above */
    page_id_t id3;
    assert(storage_allocate_page(&sm, &id3) == 0);
    /* Write something to page 3 on disk so fetch works */
    Page tmp;
    page_init(&tmp, id3, PAGE_TYPE_DATA);
    assert(storage_write_page(&sm, id3, &tmp) == 0);

    Page *p3 = bp_fetch_page(&bp, id3);
    assert(p3 != NULL);
    assert(bp_unpin_page(&bp, id3, false) == 0);

    /*
     * Now fetch page 1 back. If it was evicted (dirty), it should
     * have been flushed, so our data should survive.
     */
    Page *p1_again = bp_fetch_page(&bp, id1);
    assert(p1_again != NULL);

    uint8_t *out;
    uint16_t len;
    assert(page_get_record(p1_again, slot, &out, &len) == 0);
    assert(len == 10);
    assert(memcmp(out, "persist-me", 10) == 0);

    assert(bp_unpin_page(&bp, id1, false) == 0);

    teardown(&sm, &bp);
    printf("  PASS: test_dirty_flush_on_eviction\n");
}

static void test_pin_prevents_eviction(void) {
    StorageManager sm;
    BufferPool bp;
    setup(&sm, &bp, 2);  /* only 2 frames */

    /* Pin both frames */
    page_id_t id1, id2;
    Page *p1 = bp_new_page(&bp, &id1);
    assert(p1 != NULL);
    /* pin_count = 1, don't unpin */

    Page *p2 = bp_new_page(&bp, &id2);
    assert(p2 != NULL);
    /* pin_count = 1, don't unpin */

    /* Try to fetch a 3rd page — should fail, both frames pinned */
    page_id_t id3;
    assert(storage_allocate_page(&sm, &id3) == 0);
    Page tmp;
    page_init(&tmp, id3, PAGE_TYPE_DATA);
    assert(storage_write_page(&sm, id3, &tmp) == 0);

    Page *p3 = bp_fetch_page(&bp, id3);
    assert(p3 == NULL);  /* no evictable frame */

    /* Unpin one and try again — should succeed now */
    assert(bp_unpin_page(&bp, id1, false) == 0);
    p3 = bp_fetch_page(&bp, id3);
    assert(p3 != NULL);
    assert(p3->id == id3);

    assert(bp_unpin_page(&bp, id2, false) == 0);
    assert(bp_unpin_page(&bp, id3, false) == 0);

    teardown(&sm, &bp);
    printf("  PASS: test_pin_prevents_eviction\n");
}

static void test_clock_eviction_order(void) {
    StorageManager sm;
    BufferPool bp;
    setup(&sm, &bp, 3);

    /* Create 3 pages, filling all frames */
    page_id_t ids[3];
    for (int i = 0; i < 3; i++) {
        Page *p = bp_new_page(&bp, &ids[i]);
        assert(p != NULL);
        assert(bp_unpin_page(&bp, ids[i], false) == 0);
    }

    /*
     * Re-access page 0 and page 2 to set their ref bits.
     * Page 1 is NOT re-accessed, so it should be the eviction victim.
     */
    Page *p0 = bp_fetch_page(&bp, ids[0]);
    assert(p0 != NULL);
    assert(bp_unpin_page(&bp, ids[0], false) == 0);

    Page *p2 = bp_fetch_page(&bp, ids[2]);
    assert(p2 != NULL);
    assert(bp_unpin_page(&bp, ids[2], false) == 0);

    /* Write a distinguishing record into page 1 before it gets evicted */
    Page *p1_pre = bp_fetch_page(&bp, ids[1]);
    assert(p1_pre != NULL);
    uint16_t slot;
    assert(page_insert_record(p1_pre, "victim", 6, &slot) == 0);
    assert(bp_unpin_page(&bp, ids[1], true) == 0);
    /* Note: this fetch also set page 1's ref bit, but it will be
       cleared on the first clock sweep, and page 1 will be evicted
       on the second sweep pass. */

    /* Fetch a 4th page — forces eviction */
    page_id_t id4;
    assert(storage_allocate_page(&sm, &id4) == 0);
    Page tmp;
    page_init(&tmp, id4, PAGE_TYPE_DATA);
    assert(storage_write_page(&sm, id4, &tmp) == 0);

    Page *p4 = bp_fetch_page(&bp, id4);
    assert(p4 != NULL);
    assert(bp_unpin_page(&bp, id4, false) == 0);

    /*
     * Verify page 1's data survived eviction (it was dirty, so it
     * should have been flushed to disk before the frame was reused).
     */
    Page *p1_after = bp_fetch_page(&bp, ids[1]);
    assert(p1_after != NULL);

    uint8_t *out;
    uint16_t len;
    assert(page_get_record(p1_after, slot, &out, &len) == 0);
    assert(len == 6);
    assert(memcmp(out, "victim", 6) == 0);

    assert(bp_unpin_page(&bp, ids[1], false) == 0);

    teardown(&sm, &bp);
    printf("  PASS: test_clock_eviction_order\n");
}

static void test_flush_page(void) {
    StorageManager sm;
    BufferPool bp;
    setup(&sm, &bp, 4);

    page_id_t pid;
    Page *p = bp_new_page(&bp, &pid);
    assert(p != NULL);

    uint16_t slot;
    assert(page_insert_record(p, "flush-test", 10, &slot) == 0);
    assert(bp_unpin_page(&bp, pid, true) == 0);

    /* Explicit flush */
    assert(bp_flush_page(&bp, pid) == 0);
    assert(p->is_dirty == false);

    /* Verify data persisted by reading directly from disk */
    Page disk_page;
    assert(storage_read_page(&sm, pid, &disk_page) == 0);

    uint8_t *out;
    uint16_t len;
    assert(page_get_record(&disk_page, slot, &out, &len) == 0);
    assert(len == 10);
    assert(memcmp(out, "flush-test", 10) == 0);

    teardown(&sm, &bp);
    printf("  PASS: test_flush_page\n");
}

static void test_flush_all(void) {
    StorageManager sm;
    BufferPool bp;
    setup(&sm, &bp, 4);

    page_id_t ids[3];
    for (int i = 0; i < 3; i++) {
        Page *p = bp_new_page(&bp, &ids[i]);
        assert(p != NULL);

        char msg[16];
        snprintf(msg, sizeof(msg), "data-%d", i);
        uint16_t slot;
        assert(page_insert_record(p, msg, (uint16_t)strlen(msg), &slot) == 0);
        assert(bp_unpin_page(&bp, ids[i], true) == 0);
    }

    assert(bp_flush_all(&bp) == 0);

    /* Verify all dirty flags cleared */
    for (uint32_t i = 0; i < bp.num_frames; i++) {
        if (bp.frames[i].id != INVALID_PAGE_ID)
            assert(bp.frames[i].is_dirty == false);
    }

    teardown(&sm, &bp);
    printf("  PASS: test_flush_all\n");
}

static void test_new_page_and_delete(void) {
    StorageManager sm;
    BufferPool bp;
    setup(&sm, &bp, 4);

    page_id_t pid;
    Page *p = bp_new_page(&bp, &pid);
    assert(p != NULL);
    assert(pid >= 1);
    assert(p->pin_count == 1);
    assert(p->is_dirty == true);

    assert(bp_unpin_page(&bp, pid, true) == 0);

    /* Delete the page */
    assert(bp_delete_page(&bp, pid) == 0);

    /* Fetching the deleted page should load a free page from disk */
    /* (it was deallocated, so storage layer may have it in free list) */

    teardown(&sm, &bp);
    printf("  PASS: test_new_page_and_delete\n");
}

static void test_cannot_delete_pinned(void) {
    StorageManager sm;
    BufferPool bp;
    setup(&sm, &bp, 4);

    page_id_t pid;
    Page *p = bp_new_page(&bp, &pid);
    assert(p != NULL);

    /* Page is pinned — delete should fail */
    assert(bp_delete_page(&bp, pid) == -1);

    assert(bp_unpin_page(&bp, pid, false) == 0);

    /* Now it should succeed */
    assert(bp_delete_page(&bp, pid) == 0);

    teardown(&sm, &bp);
    printf("  PASS: test_cannot_delete_pinned\n");
}

static void test_unpin_errors(void) {
    StorageManager sm;
    BufferPool bp;
    setup(&sm, &bp, 4);

    /* Unpin a page that's not in the pool */
    assert(bp_unpin_page(&bp, 999, false) == -1);

    /* Unpin below zero */
    page_id_t pid;
    Page *p = bp_new_page(&bp, &pid);
    assert(p != NULL);
    assert(bp_unpin_page(&bp, pid, false) == 0);
    assert(bp_unpin_page(&bp, pid, false) == -1);  /* already at 0 */

    teardown(&sm, &bp);
    printf("  PASS: test_unpin_errors\n");
}

static void test_many_pages_stress(void) {
    StorageManager sm;
    BufferPool bp;
    setup(&sm, &bp, 8);

    /* Allocate 32 pages through the buffer pool, cycling through 8 frames */
    page_id_t ids[32];
    for (int i = 0; i < 32; i++) {
        Page *p = bp_new_page(&bp, &ids[i]);
        assert(p != NULL);

        char msg[32];
        snprintf(msg, sizeof(msg), "vec-%d", i);
        uint16_t slot;
        assert(page_insert_record(p, msg, (uint16_t)strlen(msg), &slot) == 0);
        assert(bp_unpin_page(&bp, ids[i], true) == 0);
    }

    /* Fetch all 32 back and verify data */
    for (int i = 0; i < 32; i++) {
        Page *p = bp_fetch_page(&bp, ids[i]);
        assert(p != NULL);

        uint8_t *out;
        uint16_t len;
        assert(page_get_record(p, 0, &out, &len) == 0);

        char expected[32];
        snprintf(expected, sizeof(expected), "vec-%d", i);
        assert(len == strlen(expected));
        assert(memcmp(out, expected, len) == 0);

        assert(bp_unpin_page(&bp, ids[i], false) == 0);
    }

    teardown(&sm, &bp);
    printf("  PASS: test_many_pages_stress\n");
}

struct ThreadArgs {
    BufferPool *bp;
    page_id_t *page_ids;
    int num_pages;
    int iters;
};

static void *thread_worker(void *arg) {
    struct ThreadArgs *args = (struct ThreadArgs *)arg;
    for (int i = 0; i < args->iters; i++) {
        for (int j = 0; j < args->num_pages; j++) {
            page_id_t pid = args->page_ids[j];
            Page *p = bp_fetch_page(args->bp, pid);
            assert(p != NULL);
            
            page_rlock(p);
            // Simulate read
            page_runlock(p);

            page_wlock(p);
            // Simulate write
            // Note: Do not set p->is_dirty directly; use bp_unpin_page(..., true)
            page_wunlock(p);
            
            assert(bp_unpin_page(args->bp, pid, true) == 0);
        }
    }
    return NULL;
}

static void test_multithreaded_stress(void) {
    StorageManager sm;
    BufferPool bp;
    setup(&sm, &bp, 8);

    page_id_t ids[16];
    for (int i = 0; i < 16; i++) {
        Page *p = bp_new_page(&bp, &ids[i]);
        assert(p != NULL);
        assert(bp_unpin_page(&bp, ids[i], true) == 0);
    }

    #define NUM_THREADS 4
    pthread_t threads[NUM_THREADS];
    struct ThreadArgs args[NUM_THREADS];
    
    for (int i = 0; i < NUM_THREADS; i++) {
        args[i].bp = &bp;
        args[i].page_ids = ids;
        args[i].num_pages = 16;
        args[i].iters = 100;
        pthread_create(&threads[i], NULL, thread_worker, &args[i]);
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    teardown(&sm, &bp);
    printf("  PASS: test_multithreaded_stress\n");
}

int main(void) {
    printf("=== Buffer Pool Tests ===\n");
    test_fetch_and_unpin();
    test_fetch_same_page_twice();
    test_dirty_flush_on_eviction();
    test_pin_prevents_eviction();
    test_clock_eviction_order();
    test_flush_page();
    test_flush_all();
    test_new_page_and_delete();
    test_cannot_delete_pinned();
    test_unpin_errors();
    test_many_pages_stress();
    test_multithreaded_stress();
    printf("All buffer pool tests passed.\n\n");
    return 0;
}
