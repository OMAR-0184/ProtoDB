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
#include <pthread.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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

static float random_normal(void) {
    float u1 = (float)rand() / (float)RAND_MAX;
    float u2 = (float)rand() / (float)RAND_MAX;
    if (u1 <= 1e-7f) u1 = 1e-7f;
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
}

static float *make_random_vectors(uint32_t n, uint16_t dim) {
    float *vecs = malloc((size_t)n * dim * sizeof(float));
    if (!vecs) return NULL;
    for (uint32_t i = 0; i < n; i++)
        for (uint16_t d = 0; d < dim; d++)
            vecs[(size_t)i * dim + d] = (float)rand() / (float)RAND_MAX;
    return vecs;
}

static float *make_clustered_vectors(uint32_t n, uint16_t dim, uint32_t num_clusters) {
    float *vecs = malloc((size_t)n * dim * sizeof(float));
    float *centers = malloc((size_t)num_clusters * dim * sizeof(float));
    if (!vecs || !centers) {
        free(vecs); free(centers);
        return NULL;
    }

    for (uint32_t c = 0; c < num_clusters; c++) {
        for (uint16_t d = 0; d < dim; d++) {
            centers[c * dim + d] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
        }
    }

    for (uint32_t i = 0; i < n; i++) {
        uint32_t c = rand() % num_clusters;
        for (uint16_t d = 0; d < dim; d++) {
            float noise = random_normal() * 0.1f;
            vecs[i * dim + d] = centers[c * dim + d] + noise;
        }
    }

    free(centers);
    return vecs;
}

static int compare_doubles(const void *a, const void *b) {
    double da = *(const double *)a;
    double db = *(const double *)b;
    if (da < db) return -1;
    if (da > db) return 1;
    return 0;
}

/* ============================================================
 *  Benchmark 1: In-memory page operations
 * ============================================================ */

static void bench_page_ops(void) {
    struct timespec _ts_start, _ts_end;
    emit("\n[1] In-Memory Page Operations (No I/O)\n");
    separator();

    const int N = 10000;
    Page p;
    uint16_t slots[200];

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

    for (int i = 0; i < N; i++)
        storage_allocate_page(&sm, &ids[i]);

    Page p;
    BENCH_START();
    for (int i = 0; i < N; i++) {
        page_init(&p, ids[i], PAGE_TYPE_DATA);
        storage_write_page(&sm, ids[i], &p);
    }
    BENCH_END();
    double write_ms = BENCH_MS();
    double write_mb = (double)N * PAGE_SIZE / (1024.0 * 1024.0);
    emit("  Write (cache): %d pages (%.1f MB) in %.2f ms  (%.1f MB/s)\n",
         N, write_mb, write_ms, write_mb / (write_ms / 1000.0));

    BENCH_START();
    for (int i = 0; i < N; i++) {
        page_init(&p, ids[i], PAGE_TYPE_DATA);
        storage_write_page(&sm, ids[i], &p);
    }
    fsync(sm.fd);
    BENCH_END();
    double sync_ms = BENCH_MS();
    emit("  Write (fsync): %d pages (%.1f MB) in %.2f ms  (%.1f MB/s)\n",
         N, write_mb, sync_ms, write_mb / (sync_ms / 1000.0));

    BENCH_START();
    for (int i = 0; i < N; i++)
        storage_read_page(&sm, ids[i], &p);
    BENCH_END();
    double read_ms = BENCH_MS();
    emit("  Read:  %d pages (%.1f MB) in %.2f ms  (%.1f MB/s)\n",
         N, write_mb, read_ms, write_mb / (read_ms / 1000.0));

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
    const int N_PAGES = 10000;
    const int ITERS = 2;

    BufferPool bp;
    bp_init(&bp, &sm, POOL_SIZE);

    page_id_t *ids = malloc((size_t)N_PAGES * sizeof(page_id_t));
    for (int i = 0; i < N_PAGES; i++) {
        Page *p = bp_new_page(&bp, &ids[i]);
        bp_unpin_page(&bp, ids[i], true);
        (void)p;
    }
    bp_flush_all(&bp);

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

    emit("  Cache ratio: %u/%d (%.1f%% capacity)\n",
         POOL_SIZE, N_PAGES, 100.0 * POOL_SIZE / N_PAGES);
    double hit_rate = 100.0 * (double)bp.hits / (double)(bp.hits + bp.misses);
    emit("  Stats: %llu hits, %llu misses (%.2f%% hit rate)\n", 
         (unsigned long long)bp.hits, (unsigned long long)bp.misses, hit_rate);

    free(ids);
    bp_destroy(&bp);
    storage_close(&sm);
    unlink(BENCH_DB);
}

/* ============================================================
 *  Benchmark Vector Setup
 * ============================================================ */

#define VEC_DIM 128
#define VEC_N_VECTORS 1000000
#define VEC_K 10
#define VEC_N_QUERIES 100
#define NUM_THREADS 4

typedef struct {
    FlatIndex *f_idx;
    IvfIndex *i_idx;
    const float *queries;
    uint32_t dim;
    uint32_t k;
    uint32_t start_q;
    uint32_t end_q;
    double *latencies;
    uint32_t *total_matches;
    bool is_flat;
} WorkerArgs;

static void *query_worker(void *arg) {
    WorkerArgs *w = (WorkerArgs *)arg;
    for (uint32_t q = w->start_q; q < w->end_q; q++) {
        struct timespec t1, t2;
        clock_gettime(CLOCK_MONOTONIC, &t1);
        
        VecResult results[VEC_K];
        uint32_t num;
        if (w->is_flat) {
            flat_index_search(w->f_idx, w->queries + (size_t)q * w->dim, w->k, NULL, NULL, results, &num);
        } else {
            ivf_index_search(w->i_idx, w->queries + (size_t)q * w->dim, w->k, NULL, NULL, results, &num);
        }
        
        clock_gettime(CLOCK_MONOTONIC, &t2);
        w->latencies[q] = elapsed_ms(t1, t2);
    }
    return NULL;
}

/* ============================================================
 *  Benchmark 4 & 5: Unified Flat & IVF
 * ============================================================ */

static void bench_vector_indexes(void) {
    struct timespec _ts_start, _ts_end;
    
    emit("\n[4 & 5] Vector DB (1M+ Vectors, Multithreaded)\n");
    separator();
    emit("  dim=%u, n=%u, queries=%u, threads=%u\n", VEC_DIM, VEC_N_VECTORS, VEC_N_QUERIES, NUM_THREADS);

    unlink(BENCH_DB);
    StorageManager sm;
    storage_open(&sm, BENCH_DB);
    BufferPool bp;
    bp_init(&bp, &sm, 100000); /* 800MB cache to hold the entire 1M vector flat index in RAM and avoid sequential flooding thrashing */

    srand(42);
    emit("  Generating clustered data...\n");
    float *vecs = make_clustered_vectors(VEC_N_VECTORS, VEC_DIM, 1024);
    float *queries = make_clustered_vectors(VEC_N_QUERIES, VEC_DIM, 1024);

    /* --- Flat Index Setup --- */
    FlatIndex f_idx;
    flat_index_create(&f_idx, &bp, VEC_DIM, VEC_DIST_L2);
    
    emit("  Inserting 1,000,000 vectors into Flat Index...\n");
    BENCH_START();
    for (uint32_t i = 0; i < VEC_N_VECTORS; i++)
        flat_index_insert(&f_idx, vecs + (size_t)i * VEC_DIM, NULL, NULL);
    BENCH_END();
    emit("  Flat Insert: %.2f ms\n", BENCH_MS());

    double latencies[VEC_N_QUERIES];
    pthread_t threads[NUM_THREADS];
    WorkerArgs args[NUM_THREADS];

    emit("\n  --- Flat Index Search ---\n");
    BENCH_START();
    uint32_t q_per_thread = VEC_N_QUERIES / NUM_THREADS;
    for (int i = 0; i < NUM_THREADS; i++) {
        args[i].f_idx = &f_idx;
        args[i].i_idx = NULL;
        args[i].queries = queries;
        args[i].dim = VEC_DIM;
        args[i].k = VEC_K;
        args[i].start_q = i * q_per_thread;
        args[i].end_q = (i == NUM_THREADS - 1) ? VEC_N_QUERIES : (i + 1) * q_per_thread;
        args[i].latencies = latencies;
        args[i].is_flat = true;
        pthread_create(&threads[i], NULL, query_worker, &args[i]);
    }
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
    BENCH_END();
    double flat_search_ms = BENCH_MS();
    
    qsort(latencies, VEC_N_QUERIES, sizeof(double), compare_doubles);
    emit("  QPS:  %.0f\n", VEC_N_QUERIES / (flat_search_ms / 1000.0));
    emit("  p50:  %.2f ms/query\n", latencies[VEC_N_QUERIES / 2]);
    emit("  p95:  %.2f ms/query\n", latencies[(int)(VEC_N_QUERIES * 0.95)]);
    emit("  p99:  %.2f ms/query\n", latencies[(int)(VEC_N_QUERIES * 0.99)]);

    /* --- Generate Ground Truth --- */
    VecResult flat_gt[VEC_N_QUERIES * VEC_K];
    for (uint32_t q = 0; q < VEC_N_QUERIES; q++) {
        uint32_t num;
        flat_index_search(&f_idx, queries + (size_t)q * VEC_DIM, VEC_K, NULL, NULL, &flat_gt[q * VEC_K], &num);
    }

    /* --- IVF Index Setup --- */
    IvfIndex i_idx;
    uint32_t nlist = 1024;
    ivf_index_create(&i_idx, &bp, VEC_DIM, VEC_DIST_L2, nlist, 1);
    
    uint32_t train_size = VEC_N_VECTORS > 65536 ? 65536 : VEC_N_VECTORS;
    emit("\n  --- IVF Index Setup ---\n");
    emit("  Training IVF on %u vectors (nlist=%u)...\n", train_size, nlist);
    BENCH_START();
    ivf_index_train(&i_idx, vecs, train_size);
    BENCH_END();
    emit("  IVF Train: %.2f ms\n", BENCH_MS());

    emit("  Inserting 1,000,000 vectors into IVF Index...\n");
    BENCH_START();
    for (uint32_t i = 0; i < VEC_N_VECTORS; i++)
        ivf_index_insert(&i_idx, vecs + (size_t)i * VEC_DIM, NULL, NULL);
    BENCH_END();
    emit("  IVF Insert: %.2f ms\n", BENCH_MS());

    /* --- IVF Search nprobe Trade-off --- */
    uint32_t probes[] = {1, 2, 4, 8, 16, 32, 64};
    int num_probes = sizeof(probes) / sizeof(probes[0]);

    emit("\n  --- IVF Trade-off Curve ---\n");
    emit("  %8s | %8s | %6s | %6s | %6s | %8s\n", "nprobe", "QPS", "Recall", "p50 ms", "p95 ms", "Scanned");
    emit("  -----------------------------------------------------------------\n");

    for (int p_idx = 0; p_idx < num_probes; p_idx++) {
        i_idx.nprobe = probes[p_idx];
        
        BENCH_START();
        for (int i = 0; i < NUM_THREADS; i++) {
            args[i].i_idx = &i_idx;
            args[i].is_flat = false;
            pthread_create(&threads[i], NULL, query_worker, &args[i]);
        }
        for (int i = 0; i < NUM_THREADS; i++) {
            pthread_join(threads[i], NULL);
        }
        BENCH_END();
        double ivf_search_ms = BENCH_MS();
        
        qsort(latencies, VEC_N_QUERIES, sizeof(double), compare_doubles);
        
        /* Calculate Recall */
        uint32_t total_matches = 0;
        for (uint32_t q = 0; q < VEC_N_QUERIES; q++) {
            VecResult ivf_res[10];
            uint32_t i_num;
            ivf_index_search(&i_idx, queries + (size_t)q * VEC_DIM, VEC_K, NULL, NULL, ivf_res, &i_num);
            
            for (uint32_t i = 0; i < i_num; i++) {
                for (uint32_t j = 0; j < VEC_K; j++) {
                    if (fabs(ivf_res[i].distance - flat_gt[q * VEC_K + j].distance) < 1e-5f) {
                        total_matches++;
                        break;
                    }
                }
            }
        }
        double recall = 100.0 * (double)total_matches / (double)(VEC_N_QUERIES * VEC_K);
        
        emit("  %8u | %8.0f | %5.1f%% | %6.2f | %6.2f | %7.1f%%\n",
             i_idx.nprobe,
             VEC_N_QUERIES / (ivf_search_ms / 1000.0),
             recall,
             latencies[VEC_N_QUERIES / 2],
             latencies[(int)(VEC_N_QUERIES * 0.95)],
             100.0 * i_idx.nprobe / nlist);
    }

    free(vecs);
    free(queries);
    flat_index_destroy(&f_idx);
    ivf_index_destroy(&i_idx);
    bp_flush_all(&bp);
    bp_destroy(&bp);
    storage_close(&sm);
    unlink(BENCH_DB);
}

/* ============================================================
 *  Benchmark 6: Distance function micro-benchmark
 * ============================================================ */

static float scalar_l2_sq(const float *a, const float *b, uint32_t dim) {
    float sum = 0.0f;
    for (uint32_t i = 0; i < dim; i++) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

static void bench_distance_funcs(void) {
    struct timespec _ts_start, _ts_end;
    emit("\n[6] Distance Functions (SIMD micro-benchmark)\n");
    separator();

    uint16_t dim = 128;
    uint32_t n_pairs = 100000;

    srand(99);
    float *a = make_random_vectors(1, dim);
    float *b = make_random_vectors(1, dim);

    volatile float sink = 0;
    BENCH_START();
    for (uint32_t i = 0; i < n_pairs; i++)
        sink = scalar_l2_sq(a, b, dim);
    BENCH_END();
    double scalar_ms = BENCH_MS();
    emit("  dim=%u, %u iterations\n", dim, n_pairs);
    emit("  Scalar L2:  %.2f ms  (%.0f Mops/sec)\n",
         scalar_ms, n_pairs / (scalar_ms / 1000.0) / 1e6);

    BENCH_START();
    for (uint32_t i = 0; i < n_pairs; i++)
        sink = vec_l2_distance_sq(a, b, dim);
    BENCH_END();
    double l2_ms = BENCH_MS();
    emit("  SIMD L2:    %.2f ms  (%.0f Mops/sec)\n",
         l2_ms, n_pairs / (l2_ms / 1000.0) / 1e6);
    emit("  SIMD Speedup = %.2f×\n", scalar_ms / l2_ms);

    BENCH_START();
    for (uint32_t i = 0; i < n_pairs; i++)
        sink = vec_cosine_distance(a, b, dim);
    BENCH_END();
    double cos_ms = BENCH_MS();
    emit("  SIMD Cosine: %.2f ms  (%.0f Mops/sec)\n",
         cos_ms, n_pairs / (cos_ms / 1000.0) / 1e6);

    BENCH_START();
    for (uint32_t i = 0; i < n_pairs; i++)
        sink = vec_inner_product_distance(a, b, dim);
    BENCH_END();
    double ip_ms = BENCH_MS();
    emit("  SIMD IP:     %.2f ms  (%.0f Mops/sec)\n",
         ip_ms, n_pairs / (ip_ms / 1000.0) / 1e6);

    (void)sink;
    free(a);
    free(b);
}

/* ============================================================
 *  Main
 * ============================================================ */

int main(void) {
    time_t now = time(NULL);
    struct tm *t = localtime(&now);
    char timestamp[64];
    strftime(timestamp, sizeof(timestamp), "%Y-%m-%d_%H-%M-%S", t);

    char filepath[256];
    snprintf(filepath, sizeof(filepath), "%s/%s.txt", RESULTS_DIR, timestamp);

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
    bench_vector_indexes();
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
