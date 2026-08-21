#include "../src/pages/page.h"
#include "../src/pages/storage_mgr.h"
#include "../src/buffer/buffer_pool.h"
#include "../src/index/vec_utils.h"
#include "../src/index/vec_topk.h"
#include "../src/index/vec_page.h"
#include "../src/index/flat/flat_index.h"
#include "../src/index/ivf/ivf_index.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/stat.h>
#include <math.h>
#include <stdarg.h>

/* ---- timing helpers ---- */

#define BENCH_DB       "build/bench_protodb.db"
#define RESULTS_DIR    "bench/results"

static double elapsed_ms(struct timespec start, struct timespec end) {
    double sec  = (double)(end.tv_sec  - start.tv_sec);
    double nsec = (double)(end.tv_nsec - start.tv_nsec);
    return sec * 1000.0 + nsec / 1e6;
}

#define BENCH_START() \
    clock_gettime(CLOCK_MONOTONIC, &_ts_start)

#define BENCH_END() \
    clock_gettime(CLOCK_MONOTONIC, &_ts_end)

#define BENCH_MS() elapsed_ms(_ts_start, _ts_end)

/* ---- output helpers ---- */

static FILE *g_report = NULL;

static void emit(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vprintf(fmt, ap);
    va_end(ap);

    if (g_report) {
        va_start(ap, fmt);
        vfprintf(g_report, fmt, ap);
        va_end(ap);
    }
}

static void separator(void) {
    emit("─────────────────────────────────────────────────────────\n");
}

/* ---- random data generation ---- */

static float *make_random_vectors(uint32_t n, uint16_t dim) {
    float *vecs = malloc((size_t)n * dim * sizeof(float));
    if (!vecs) return NULL;
    for (uint32_t i = 0; i < n; i++)
        for (uint16_t d = 0; d < dim; d++)
            vecs[(size_t)i * dim + d] = (float)rand() / (float)RAND_MAX;
    return vecs;
}

/* ============================================================
 *  Benchmark 1: In-memory page operations
 * ============================================================ */

static void bench_page_ops(void) {
    struct timespec _ts_start, _ts_end;
    emit("\n[1] Page Insert / Get / Delete\n");
    separator();

    const int N = 10000;
    Page p;
    uint16_t slots[200]; /* max slots per page */

    /* --- Insert --- */
    int total_inserted = 0;
    BENCH_START();
    for (int round = 0; round < N; round++) {
        page_init(&p, 1, PAGE_TYPE_DATA);
        int count = 0;
        uint16_t slot;
        char rec[32];
        memset(rec, 'A', sizeof(rec));
        while (page_insert_record(&p, rec, sizeof(rec), &slot) == 0) {
            if (count < 200) slots[count] = slot;
            count++;
        }
        total_inserted += count;
    }
    BENCH_END();
    double insert_ms = BENCH_MS();
    emit("  Insert: %d total records in %.2f ms  (%.0f ops/sec)\n",
         total_inserted, insert_ms, total_inserted / (insert_ms / 1000.0));

    /* --- Get --- */
    page_init(&p, 1, PAGE_TYPE_DATA);
    int filled = 0;
    uint16_t slot;
    char rec[32];
    memset(rec, 'B', sizeof(rec));
    while (page_insert_record(&p, rec, sizeof(rec), &slot) == 0) {
        if (filled < 200) slots[filled] = slot;
        filled++;
    }

    BENCH_START();
    for (int round = 0; round < N; round++) {
        for (int i = 0; i < filled; i++) {
            uint8_t *out;
            uint16_t len;
            page_get_record(&p, slots[i], &out, &len);
        }
    }
    BENCH_END();
    double get_ms = BENCH_MS();
    long total_gets = (long)N * filled;
    emit("  Get:    %ld total lookups in %.2f ms  (%.0f ops/sec)\n",
         total_gets, get_ms, total_gets / (get_ms / 1000.0));

    /* --- Delete + Compact --- */
    BENCH_START();
    for (int round = 0; round < N; round++) {
        page_init(&p, 1, PAGE_TYPE_DATA);
        int c = 0;
        while (page_insert_record(&p, rec, sizeof(rec), &slot) == 0) {
            if (c < 200) slots[c] = slot;
            c++;
        }
        for (int i = 0; i < c; i += 2)
            page_delete_record(&p, slots[i]);
        page_compact(&p);
    }
    BENCH_END();
    double del_ms = BENCH_MS();
    emit("  Delete+Compact: %d cycles in %.2f ms  (%.0f cycles/sec)\n",
         N, del_ms, N / (del_ms / 1000.0));
}

/* ============================================================
 *  Benchmark 2: Storage I/O
 * ============================================================ */

static void bench_storage_io(void) {
    struct timespec _ts_start, _ts_end;
    emit("\n[2] Storage I/O (read / write pages)\n");
    separator();

    unlink(BENCH_DB);
    StorageManager sm;
    storage_open(&sm, BENCH_DB);

    const int N = 1000;
    page_id_t ids[1000];

    /* Allocate pages */
    for (int i = 0; i < N; i++)
        storage_allocate_page(&sm, &ids[i]);

    /* --- Write --- */
    Page p;
    BENCH_START();
    for (int i = 0; i < N; i++) {
        page_init(&p, ids[i], PAGE_TYPE_DATA);
        storage_write_page(&sm, ids[i], &p);
    }
    BENCH_END();
    double write_ms = BENCH_MS();
    double write_mb = (double)N * PAGE_SIZE / (1024.0 * 1024.0);
    emit("  Write: %d pages (%.1f MB) in %.2f ms  (%.1f MB/s)\n",
         N, write_mb, write_ms, write_mb / (write_ms / 1000.0));

    /* --- Sequential read --- */
    BENCH_START();
    for (int i = 0; i < N; i++)
        storage_read_page(&sm, ids[i], &p);
    BENCH_END();
    double read_ms = BENCH_MS();
    emit("  Read:  %d pages (%.1f MB) in %.2f ms  (%.1f MB/s)\n",
         N, write_mb, read_ms, write_mb / (read_ms / 1000.0));

    /* --- Bulk read --- */
    Page *bulk = malloc((size_t)N * sizeof(Page));
    if (bulk) {
        BENCH_START();
        storage_read_pages(&sm, ids[0], (uint32_t)N, bulk);
        BENCH_END();
        double bulk_ms = BENCH_MS();
        emit("  Bulk:  %d pages (%.1f MB) in %.2f ms  (%.1f MB/s)\n",
             N, write_mb, bulk_ms, write_mb / (bulk_ms / 1000.0));
        free(bulk);
    }

    storage_close(&sm);
    unlink(BENCH_DB);
}

/* ============================================================
 *  Benchmark 3: Buffer pool throughput
 * ============================================================ */

static void bench_buffer_pool(void) {
    struct timespec _ts_start, _ts_end;
    emit("\n[3] Buffer Pool (fetch / unpin cycles)\n");
    separator();

    unlink(BENCH_DB);
    StorageManager sm;
    storage_open(&sm, BENCH_DB);

    const uint32_t POOL_SIZE = 64;
    const int N_PAGES = 256;
    const int ITERS = 10;

    BufferPool bp;
    bp_init(&bp, &sm, POOL_SIZE);

    /* Create pages */
    page_id_t ids[256];
    for (int i = 0; i < N_PAGES; i++) {
        Page *p = bp_new_page(&bp, &ids[i]);
        bp_unpin_page(&bp, ids[i], true);
        (void)p;
    }
    bp_flush_all(&bp);

    /* Benchmark fetch/unpin with cache pressure */
    long total_ops = 0;
    BENCH_START();
    for (int iter = 0; iter < ITERS; iter++) {
        for (int i = 0; i < N_PAGES; i++) {
            Page *p = bp_fetch_page(&bp, ids[i]);
            if (p) {
                bp_unpin_page(&bp, ids[i], false);
                total_ops++;
            }
        }
    }
    BENCH_END();
    double ms = BENCH_MS();
    emit("  Pool=%u frames, %d pages, %d iterations\n", POOL_SIZE, N_PAGES, ITERS);
    emit("  Fetch+Unpin: %ld ops in %.2f ms  (%.0f ops/sec)\n",
         total_ops, ms, total_ops / (ms / 1000.0));

    long hits = 0, misses = 0;
    for (int i = 0; i < N_PAGES; i++) {
        Page *p = bp_fetch_page(&bp, ids[i]);
        if (p) {
            /* If page was already in pool, it's a hit */
            bp_unpin_page(&bp, ids[i], false);
        }
    }
    /* Approximate: last POOL_SIZE pages should be cached */
    emit("  Cache ratio: %u/%d (%.0f%% capacity)\n",
         POOL_SIZE, N_PAGES, 100.0 * POOL_SIZE / N_PAGES);
    (void)hits; (void)misses;

    bp_destroy(&bp);
    storage_close(&sm);
    unlink(BENCH_DB);
}

/* ============================================================
 *  Benchmark 4: Flat index
 * ============================================================ */

static void bench_flat_index(void) {
    struct timespec _ts_start, _ts_end;
    emit("\n[4] Flat Index (brute-force vector search)\n");
    separator();

    unlink(BENCH_DB);
    StorageManager sm;
    storage_open(&sm, BENCH_DB);
    BufferPool bp;
    bp_init(&bp, &sm, 256);

    uint16_t dim = 128;
    uint32_t n_vectors = 5000;
    uint32_t k = 10;
    uint32_t n_queries = 100;

    FlatIndex idx;
    flat_index_create(&idx, &bp, dim, VEC_DIST_L2);

    srand(42);
    float *vecs = make_random_vectors(n_vectors, dim);

    /* --- Insert --- */
    BENCH_START();
    for (uint32_t i = 0; i < n_vectors; i++)
        flat_index_insert(&idx, vecs + (size_t)i * dim, NULL, NULL);
    BENCH_END();
    double insert_ms = BENCH_MS();
    emit("  dim=%u, n=%u\n", dim, n_vectors);
    emit("  Insert: %.2f ms  (%.0f vectors/sec)\n",
         insert_ms, n_vectors / (insert_ms / 1000.0));

    /* --- Search --- */
    float *queries = make_random_vectors(n_queries, dim);
    VecResult results[10];
    uint32_t num;

    BENCH_START();
    for (uint32_t q = 0; q < n_queries; q++)
        flat_index_search(&idx, queries + (size_t)q * dim, k, results, &num);
    BENCH_END();
    double search_ms = BENCH_MS();
    emit("  Search (k=%u): %u queries in %.2f ms  (%.1f ms/query, %.0f QPS)\n",
         k, n_queries, search_ms, search_ms / n_queries,
         n_queries / (search_ms / 1000.0));

    free(vecs);
    free(queries);
    flat_index_destroy(&idx);
    bp_flush_all(&bp);
    bp_destroy(&bp);
    storage_close(&sm);
    unlink(BENCH_DB);
}

/* ============================================================
 *  Benchmark 5: IVF index
 * ============================================================ */

static void bench_ivf_index(void) {
    struct timespec _ts_start, _ts_end;
    emit("\n[5] IVF Index (approximate vector search)\n");
    separator();

    unlink(BENCH_DB);
    StorageManager sm;
    storage_open(&sm, BENCH_DB);
    BufferPool bp;
    bp_init(&bp, &sm, 512);

    uint16_t dim = 128;
    uint32_t n_vectors = 5000;
    uint32_t nlist = 32;
    uint32_t nprobe = 4;
    uint32_t k = 10;
    uint32_t n_queries = 100;

    IvfIndex idx;
    ivf_index_create(&idx, &bp, dim, VEC_DIST_L2, nlist, nprobe);

    srand(42);
    float *vecs = make_random_vectors(n_vectors, dim);

    /* --- Train --- */
    BENCH_START();
    ivf_index_train(&idx, vecs, n_vectors);
    BENCH_END();
    double train_ms = BENCH_MS();
    emit("  dim=%u, n=%u, nlist=%u, nprobe=%u\n", dim, n_vectors, nlist, nprobe);
    emit("  Train: %.2f ms\n", train_ms);

    /* --- Insert --- */
    BENCH_START();
    for (uint32_t i = 0; i < n_vectors; i++)
        ivf_index_insert(&idx, vecs + (size_t)i * dim, NULL, NULL);
    BENCH_END();
    double insert_ms = BENCH_MS();
    emit("  Insert: %.2f ms  (%.0f vectors/sec)\n",
         insert_ms, n_vectors / (insert_ms / 1000.0));

    /* --- Search --- */
    float *queries = make_random_vectors(n_queries, dim);
    VecResult results[10];
    uint32_t num;

    BENCH_START();
    for (uint32_t q = 0; q < n_queries; q++)
        ivf_index_search(&idx, queries + (size_t)q * dim, k, results, &num);
    BENCH_END();
    double search_ms = BENCH_MS();
    emit("  Search (k=%u): %u queries in %.2f ms  (%.1f ms/query, %.0f QPS)\n",
         k, n_queries, search_ms, search_ms / n_queries,
         n_queries / (search_ms / 1000.0));

    /* --- Speedup vs flat --- */
    emit("  Speedup context: nprobe=%u / nlist=%u → scans ~%.0f%% of data\n",
         nprobe, nlist, 100.0 * nprobe / nlist);

    free(vecs);
    free(queries);
    ivf_index_destroy(&idx);
    bp_flush_all(&bp);
    bp_destroy(&bp);
    storage_close(&sm);
    unlink(BENCH_DB);
}

/* ============================================================
 *  Benchmark 6: Distance function micro-benchmark
 * ============================================================ */

static void bench_distance_funcs(void) {
    struct timespec _ts_start, _ts_end;
    emit("\n[6] Distance Functions (SIMD micro-benchmark)\n");
    separator();

    uint16_t dim = 128;
    uint32_t n_pairs = 100000;

    srand(99);
    float *a = make_random_vectors(1, dim);
    float *b = make_random_vectors(1, dim);

    /* --- L2² --- */
    volatile float sink = 0;
    BENCH_START();
    for (uint32_t i = 0; i < n_pairs; i++)
        sink = vec_l2_distance_sq(a, b, dim);
    BENCH_END();
    double l2_ms = BENCH_MS();
    emit("  dim=%u, %u iterations\n", dim, n_pairs);
    emit("  L2²:     %.2f ms  (%.0f Mops/sec)\n",
         l2_ms, n_pairs / (l2_ms / 1000.0) / 1e6);

    /* --- Cosine --- */
    BENCH_START();
    for (uint32_t i = 0; i < n_pairs; i++)
        sink = vec_cosine_distance(a, b, dim);
    BENCH_END();
    double cos_ms = BENCH_MS();
    emit("  Cosine:  %.2f ms  (%.0f Mops/sec)\n",
         cos_ms, n_pairs / (cos_ms / 1000.0) / 1e6);

    /* --- Inner product --- */
    BENCH_START();
    for (uint32_t i = 0; i < n_pairs; i++)
        sink = vec_inner_product_distance(a, b, dim);
    BENCH_END();
    double ip_ms = BENCH_MS();
    emit("  IP:      %.2f ms  (%.0f Mops/sec)\n",
         ip_ms, n_pairs / (ip_ms / 1000.0) / 1e6);

    (void)sink;
    free(a);
    free(b);
}

/* ============================================================
 *  Main
 * ============================================================ */

int main(void) {
    /* Generate timestamp filename */
    time_t now = time(NULL);
    struct tm *t = localtime(&now);
    char timestamp[64];
    strftime(timestamp, sizeof(timestamp), "%Y-%m-%d_%H-%M-%S", t);

    char filepath[256];
    snprintf(filepath, sizeof(filepath), "%s/%s.txt", RESULTS_DIR, timestamp);

    /* Ensure results directory exists */
    mkdir(RESULTS_DIR, 0755);

    g_report = fopen(filepath, "w");

    emit("╔═══════════════════════════════════════════════════════╗\n");
    emit("║             ProtoDB Benchmark Suite                   ║\n");
    emit("╚═══════════════════════════════════════════════════════╝\n");

    char time_str[64];
    strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", t);
    emit("  Timestamp:  %s\n", time_str);
    emit("  Page size:  %d bytes\n", PAGE_SIZE);
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    emit("  SIMD:       ARM NEON\n");
#elif defined(__SSE__)
    emit("  SIMD:       SSE\n");
#else
    emit("  SIMD:       None (scalar)\n");
#endif

    bench_page_ops();
    bench_storage_io();
    bench_buffer_pool();
    bench_flat_index();
    bench_ivf_index();
    bench_distance_funcs();

    emit("\n");
    separator();
    if (g_report) {
        emit("Results saved to: %s\n", filepath);
        fclose(g_report);
        g_report = NULL;
        printf("Results saved to: %s\n", filepath);
    }

    return 0;
}
