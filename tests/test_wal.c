#include "../src/wal/wal.h"
#include "../src/pages/storage_mgr.h"
#include "../src/buffer/buffer_pool.h"
#include "../src/index/flat/flat_index.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>
#include <fcntl.h>
#include <stddef.h>

static const char *TEST_DB  = "build/protodb_wal_test.db";
static const char *TEST_WAL = "build/protodb_wal_test.db.wal";

static void cleanup(void) {
    unlink(TEST_DB);
    unlink(TEST_WAL);
}

/* ---- Test 1: Basic WAL write and checkpoint ---- */
static void test_wal_basic_checkpoint(void) {
    cleanup();

    StorageManager sm;
    assert(storage_open(&sm, TEST_DB) == 0);

    Wal wal;
    assert(wal_open(&wal, TEST_DB, &sm) == 0);
    assert(wal.num_records == 0);

    /* Allocate a page and write data to it */
    page_id_t pid;
    assert(storage_allocate_page(&sm, &pid) == 0);

    Page p;
    assert(storage_read_page(&sm, pid, &p) == 0);
    memset(p.data + PAGE_DATA_OFFSET, 0xAB, 64);

    /* Log to WAL */
    assert(wal_log_page(&wal, pid, p.data) == 0);
    assert(wal.num_records == 1);

    /* Checkpoint — should replay into db and clear WAL */
    assert(wal_checkpoint(&wal, &sm) == 0);
    assert(wal.num_records == 0);

    /* Verify the data is in the database */
    Page verify;
    assert(storage_read_page(&sm, pid, &verify) == 0);
    assert(memcmp(verify.data + PAGE_DATA_OFFSET, p.data + PAGE_DATA_OFFSET, 64) == 0);

    wal_close(&wal);
    storage_close(&sm);
    cleanup();

    printf("  PASS: test_wal_basic_checkpoint\n");
}

/* ---- Test 2: WAL recovery after simulated crash ---- */
static void test_wal_crash_recovery(void) {
    cleanup();

    StorageManager sm;
    assert(storage_open(&sm, TEST_DB) == 0);

    Wal wal;
    assert(wal_open(&wal, TEST_DB, &sm) == 0);

    /* Allocate two pages and write different data to each */
    page_id_t pid1, pid2;
    assert(storage_allocate_page(&sm, &pid1) == 0);
    assert(storage_allocate_page(&sm, &pid2) == 0);

    Page p1, p2;
    assert(storage_read_page(&sm, pid1, &p1) == 0);
    assert(storage_read_page(&sm, pid2, &p2) == 0);

    memset(p1.data + PAGE_DATA_OFFSET, 0xAA, 128);
    memset(p2.data + PAGE_DATA_OFFSET, 0xBB, 128);

    /* Log both to WAL but do NOT checkpoint (simulating a crash) */
    assert(wal_log_page(&wal, pid1, p1.data) == 0);
    assert(wal_log_page(&wal, pid2, p2.data) == 0);
    assert(wal.num_records == 2);

    /* Close without checkpointing — simulates a crash */
    wal_close(&wal);
    storage_close(&sm);

    /* Re-open — WAL recovery should automatically replay */
    StorageManager sm2;
    assert(storage_open(&sm2, TEST_DB) == 0);

    Wal wal2;
    assert(wal_open(&wal2, TEST_DB, &sm2) == 0);

    /* After recovery, WAL should be empty */
    assert(wal2.num_records == 0);

    /* Verify recovered data */
    Page v1, v2;
    assert(storage_read_page(&sm2, pid1, &v1) == 0);
    assert(storage_read_page(&sm2, pid2, &v2) == 0);

    uint8_t expected1[128], expected2[128];
    memset(expected1, 0xAA, 128);
    memset(expected2, 0xBB, 128);

    assert(memcmp(v1.data + PAGE_DATA_OFFSET, expected1, 128) == 0);
    assert(memcmp(v2.data + PAGE_DATA_OFFSET, expected2, 128) == 0);

    wal_close(&wal2);
    storage_close(&sm2);
    cleanup();

    printf("  PASS: test_wal_crash_recovery\n");
}

/* ---- Test 3: CRC32 corruption detection ---- */
static void test_wal_crc32_corruption(void) {
    cleanup();

    StorageManager sm;
    assert(storage_open(&sm, TEST_DB) == 0);

    Wal wal;
    assert(wal_open(&wal, TEST_DB, &sm) == 0);

    page_id_t pid;
    assert(storage_allocate_page(&sm, &pid) == 0);

    Page p;
    assert(storage_read_page(&sm, pid, &p) == 0);
    memset(p.data + PAGE_DATA_OFFSET, 0xCC, 64);

    assert(wal_log_page(&wal, pid, p.data) == 0);
    wal_close(&wal);

    /* Corrupt the WAL record's data by flipping a byte */
    int wal_fd = open(TEST_WAL, O_RDWR);
    assert(wal_fd >= 0);
    
    /* Corrupt a byte in the page data area of the first record */
    off_t corrupt_offset = (off_t)sizeof(WalHeader) + offsetof(WalRecord, data) + 100;
    uint8_t bad_byte = 0xFF;
    assert(pwrite(wal_fd, &bad_byte, 1, corrupt_offset) == 1);
    close(wal_fd);

    /* Re-open — recovery should detect the corruption and skip the record */
    StorageManager sm2;
    assert(storage_open(&sm2, TEST_DB) == 0);

    Wal wal2;
    /* Recovery should succeed (skipping the corrupted record) */
    assert(wal_open(&wal2, TEST_DB, &sm2) == 0);
    assert(wal2.num_records == 0); /* WAL cleared after recovery */

    wal_close(&wal2);
    storage_close(&sm2);
    cleanup();

    printf("  PASS: test_wal_crc32_corruption\n");
}

/* ---- Test 4: WAL integration with buffer pool ---- */
static void test_wal_buffer_pool_integration(void) {
    cleanup();

    StorageManager sm;
    assert(storage_open(&sm, TEST_DB) == 0);

    Wal wal;
    assert(wal_open(&wal, TEST_DB, &sm) == 0);

    BufferPool bp;
    assert(bp_init(&bp, &sm, 4) == 0);
    bp.wal = &wal;  /* Enable WAL on the buffer pool */

    /* Create a flat index (which allocates pages through the buffer pool) */
    FlatIndex idx;
    assert(flat_index_create(&idx, &bp, 4, VEC_DIST_L2) == 0);

    /* Insert some vectors */
    float v1[4] = {1.0, 2.0, 3.0, 4.0};
    float v2[4] = {5.0, 6.0, 7.0, 8.0};
    page_id_t p; uint16_t s;
    assert(flat_index_insert(&idx, v1, &p, &s) == 0);
    assert(flat_index_insert(&idx, v2, &p, &s) == 0);

    /* Flush — this should log to WAL first */
    assert(bp_flush_all(&bp) == 0);

    /* WAL should have records from the flush */
    assert(wal.num_records > 0);

    /* Checkpoint */
    assert(wal_checkpoint(&wal, &sm) == 0);
    assert(wal.num_records == 0);

    flat_index_destroy(&idx);
    bp_destroy(&bp);
    wal_close(&wal);
    storage_close(&sm);
    cleanup();

    printf("  PASS: test_wal_buffer_pool_integration\n");
}

/* ---- Test 5: Multiple checkpoints ---- */
static void test_wal_multiple_checkpoints(void) {
    cleanup();

    StorageManager sm;
    assert(storage_open(&sm, TEST_DB) == 0);

    Wal wal;
    assert(wal_open(&wal, TEST_DB, &sm) == 0);

    for (int round = 0; round < 5; round++) {
        page_id_t pid;
        assert(storage_allocate_page(&sm, &pid) == 0);

        Page p;
        assert(storage_read_page(&sm, pid, &p) == 0);
        memset(p.data + PAGE_DATA_OFFSET, (uint8_t)(round + 1), 64);

        assert(wal_log_page(&wal, pid, p.data) == 0);
        assert(wal_checkpoint(&wal, &sm) == 0);
        assert(wal.num_records == 0);

        /* Verify */
        Page v;
        assert(storage_read_page(&sm, pid, &v) == 0);
        for (int j = 0; j < 64; j++) {
            assert(v.data[PAGE_DATA_OFFSET + j] == (uint8_t)(round + 1));
        }
    }

    wal_close(&wal);
    storage_close(&sm);
    cleanup();

    printf("  PASS: test_wal_multiple_checkpoints\n");
}

/* ---- Test 6: Empty WAL checkpoint is a no-op ---- */
static void test_wal_empty_checkpoint(void) {
    cleanup();

    StorageManager sm;
    assert(storage_open(&sm, TEST_DB) == 0);

    Wal wal;
    assert(wal_open(&wal, TEST_DB, &sm) == 0);

    /* Checkpoint with no records should succeed and be a no-op */
    assert(wal_checkpoint(&wal, &sm) == 0);
    assert(wal.num_records == 0);

    wal_close(&wal);
    storage_close(&sm);
    cleanup();

    printf("  PASS: test_wal_empty_checkpoint\n");
}

/* ---- Test 7: CRC32 known-value test ---- */
static void test_wal_crc32_known_values(void) {
    /* CRC32 of empty data */
    uint32_t crc_empty = wal_crc32((const uint8_t *)"", 0);
    assert(crc_empty == 0x00000000);  /* CRC32 of empty = 0 */

    /* CRC32 of "123456789" — standard test vector */
    uint32_t crc_test = wal_crc32((const uint8_t *)"123456789", 9);
    assert(crc_test == 0xE20BB246);

    printf("  PASS: test_wal_crc32_known_values\n");
}

int main(void) {
    printf("=== WAL Tests ===\n");
    test_wal_crc32_known_values();
    test_wal_basic_checkpoint();
    test_wal_crash_recovery();
    test_wal_crc32_corruption();
    test_wal_buffer_pool_integration();
    test_wal_multiple_checkpoints();
    test_wal_empty_checkpoint();
    printf("All WAL tests passed.\n\n");
    return 0;
}
