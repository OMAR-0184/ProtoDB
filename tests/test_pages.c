#include "../src/pages/page.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

static void test_init(void) {
    Page p;
    page_init(&p, 1, PAGE_TYPE_DATA);

    PageHeader *h = page_get_header(&p);
    assert(h->page_id == 1);
    assert(h->page_type == PAGE_TYPE_DATA);
    assert(h->num_slots == 0);
    assert(h->free_space_end == PAGE_SIZE);
    printf("  PASS: test_init\n");
}

static void test_insert_and_get(void) {
    Page p;
    page_init(&p, 2, PAGE_TYPE_DATA);

    const char *rec1 = "hello";
    const char *rec2 = "world!";
    uint16_t s1, s2;

    assert(page_insert_record(&p, rec1, 5, &s1) == 0);
    assert(s1 == 0);

    assert(page_insert_record(&p, rec2, 6, &s2) == 0);
    assert(s2 == 1);

    uint8_t *out;
    uint16_t len;

    assert(page_get_record(&p, s1, &out, &len) == 0);
    assert(len == 5);
    assert(memcmp(out, "hello", 5) == 0);

    assert(page_get_record(&p, s2, &out, &len) == 0);
    assert(len == 6);
    assert(memcmp(out, "world!", 6) == 0);

    printf("  PASS: test_insert_and_get\n");
}

static void test_delete(void) {
    Page p;
    page_init(&p, 3, PAGE_TYPE_DATA);

    const char *rec = "deleteme";
    uint16_t slot;
    assert(page_insert_record(&p, rec, 8, &slot) == 0);

    assert(page_delete_record(&p, slot) == 0);

    uint8_t *out;
    uint16_t len;
    assert(page_get_record(&p, slot, &out, &len) == -1);

    assert(page_delete_record(&p, slot) == -1);

    printf("  PASS: test_delete\n");
}

static void test_slot_reuse(void) {
    Page p;
    page_init(&p, 4, PAGE_TYPE_DATA);

    uint16_t s0, s1, s2;
    assert(page_insert_record(&p, "AAA", 3, &s0) == 0);
    assert(page_insert_record(&p, "BBB", 3, &s1) == 0);
    assert(s0 == 0 && s1 == 1);

    page_delete_record(&p, s0);

    assert(page_insert_record(&p, "CCC", 3, &s2) == 0);
    assert(s2 == 0);

    uint8_t *out;
    uint16_t len;
    assert(page_get_record(&p, s2, &out, &len) == 0);
    assert(memcmp(out, "CCC", 3) == 0);

    printf("  PASS: test_slot_reuse\n");
}

static void test_update_same_size(void) {
    Page p;
    page_init(&p, 5, PAGE_TYPE_DATA);

    uint16_t slot;
    assert(page_insert_record(&p, "AAAA", 4, &slot) == 0);
    assert(page_update_record(&p, slot, "BBBB", 4) == 0);

    uint8_t *out;
    uint16_t len;
    assert(page_get_record(&p, slot, &out, &len) == 0);
    assert(memcmp(out, "BBBB", 4) == 0);

    printf("  PASS: test_update_same_size\n");
}

static void test_update_different_size(void) {
    Page p;
    page_init(&p, 6, PAGE_TYPE_DATA);

    uint16_t slot;
    assert(page_insert_record(&p, "short", 5, &slot) == 0);
    assert(page_update_record(&p, slot, "a longer record", 15) == 0);

    uint8_t *out;
    uint16_t len;
    assert(page_get_record(&p, slot, &out, &len) == 0);
    assert(len == 15);
    assert(memcmp(out, "a longer record", 15) == 0);

    printf("  PASS: test_update_different_size\n");
}

static void test_compact(void) {
    Page p;
    page_init(&p, 7, PAGE_TYPE_DATA);

    uint16_t s0, s1, s2;
    assert(page_insert_record(&p, "AAAA", 4, &s0) == 0);
    assert(page_insert_record(&p, "BBBB", 4, &s1) == 0);
    assert(page_insert_record(&p, "CCCC", 4, &s2) == 0);

    uint16_t before = page_free_space(&p);
    page_delete_record(&p, s1);
    page_compact(&p);
    uint16_t after = page_free_space(&p);

    assert(after == before + 4);

    uint8_t *out;
    uint16_t len;
    assert(page_get_record(&p, s0, &out, &len) == 0);
    assert(memcmp(out, "AAAA", 4) == 0);
    assert(page_get_record(&p, s2, &out, &len) == 0);
    assert(memcmp(out, "CCCC", 4) == 0);

    printf("  PASS: test_compact\n");
}

static void test_free_space_exhaustion(void) {
    Page p;
    page_init(&p, 8, PAGE_TYPE_DATA);

    uint16_t slot;
    int count = 0;
    char buf[64];
    memset(buf, 'X', sizeof(buf));

    while (page_insert_record(&p, buf, sizeof(buf), &slot) == 0)
        count++;

    assert(count > 0);
    assert(page_free_space(&p) < sizeof(buf) + SLOT_SIZE);

    printf("  PASS: test_free_space_exhaustion (inserted %d records)\n", count);
}

static void test_boundary_checks(void) {
    Page p;
    page_init(&p, 9, PAGE_TYPE_DATA);

    uint8_t *out;
    uint16_t len;
    assert(page_get_record(&p, 0, &out, &len) == -1);
    assert(page_delete_record(&p, 0) == -1);
    assert(page_update_record(&p, 0, "x", 1) == -1);

    uint16_t slot;
    assert(page_insert_record(&p, "x", 0, &slot) == -1);

    printf("  PASS: test_boundary_checks\n");
}

int main(void) {
    printf("=== Page Tests ===\n");
    test_init();
    test_insert_and_get();
    test_delete();
    test_slot_reuse();
    test_update_same_size();
    test_update_different_size();
    test_compact();
    test_free_space_exhaustion();
    test_boundary_checks();
    printf("All page tests passed.\n\n");
    return 0;
}
