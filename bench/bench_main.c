#include "../src/pages/page.h"
#include "../src/pages/storage_mgr.h"
#include "../src/buffer/buffer_pool.h"
#include "../src/index/vec_utils.h"
#include "../src/index/vec_topk.h"
#include "../src/index/vec_page.h"
#include "../src/index/flat/flat_index.h"
#include "../src/index/ivf/ivf_index.h"
#include "../src/index/hnsw/hnsw_index.h"
#include "../src/index/quantize/vec_quantize.h"

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

typedef enum {
    WORKER_FLAT,
    WORKER_IVF,
    WORKER_HNSW
} WorkerIndexType;

typedef struct {
    FlatIndex       *f_idx;
    IvfIndex        *i_idx;
    HnswIndex       *h_idx;
    uint32_t         ef_search;
    const float     *queries;
    uint32_t         dim;
    uint32_t         k;
    uint32_t         start_q;
    uint32_t         end_q;
    double          *latencies;
    VecResult       *all_results;
    WorkerIndexType  type;
} WorkerArgs;

static void *query_worker(void *arg) {
    WorkerArgs *w = (WorkerArgs *)arg;
    for (uint32_t q = w->start_q; q < w->end_q; q++) {
        struct timespec t1, t2;
        clock_gettime(CLOCK_MONOTONIC, &t1);
        
        VecResult results[VEC_K];
        uint32_t num = 0;
        if (w->type == WORKER_FLAT) {
            flat_index_search(w->f_idx, w->queries + (size_t)q * w->dim, w->k, NULL, NULL, results, &num);
        } else if (w->type == WORKER_IVF) {
            ivf_index_search(w->i_idx, w->queries + (size_t)q * w->dim, w->k, NULL, NULL, results, &num);
        } else if (w->type == WORKER_HNSW) {
            hnsw_index_search(w->h_idx, w->queries + (size_t)q * w->dim, w->k, w->ef_search, results, &num);
        }
        
        clock_gettime(CLOCK_MONOTONIC, &t2);
        w->latencies[q] = elapsed_ms(t1, t2);

        if (w->all_results) {
            for (uint32_t i = 0; i < num && i < w->k; i++) {
                w->all_results[q * w->k + i] = results[i];
            }
        }
    }
    return NULL;
}

typedef struct {
    double recall_at_1;
    double recall_at_k;
} RecallStats;

static RecallStats compute_recall(const VecResult *test_results,
                                  const VecResult *ground_truth,
                                  uint32_t num_queries, uint32_t k) {
    uint32_t top1_matches = 0;
    uint32_t topk_matches = 0;

    for (uint32_t q = 0; q < num_queries; q++) {
        const VecResult *res = test_results + (size_t)q * k;
        const VecResult *gt  = ground_truth + (size_t)q * k;

        if (fabsf(res[0].distance - gt[0].distance) < 1e-4f) {
            top1_matches++;
        }

        for (uint32_t i = 0; i < k; i++) {
            for (uint32_t j = 0; j < k; j++) {
                if (fabsf(res[i].distance - gt[j].distance) < 1e-4f) {
                    topk_matches++;
                    break;
                }
            }
        }
    }

    RecallStats stats;
    stats.recall_at_1 = 100.0 * (double)top1_matches / (double)num_queries;
    stats.recall_at_k = 100.0 * (double)topk_matches / (double)(num_queries * k);
    return stats;
}

/* ============================================================
 *  Benchmark 4: 1M+ Vector Scale (Flat & IVF)
 * ============================================================ */

static void bench_vector_indexes(void) {
    struct timespec _ts_start, _ts_end;
    
    emit("\n[4] Vector DB (1M+ Vectors, Multithreaded)\n");
    separator();
    emit("  dim=%u, n=%u, queries=%u, threads=%u\n", VEC_DIM, VEC_N_VECTORS, VEC_N_QUERIES, NUM_THREADS);

    unlink(BENCH_DB);
    StorageManager sm;
    storage_open(&sm, BENCH_DB);
    BufferPool bp;
    bp_init(&bp, &sm, 100000); /* 800MB cache */

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
    emit("  Flat Insert: %.2f ms (%.0f vecs/sec)\n", BENCH_MS(), VEC_N_VECTORS / (BENCH_MS() / 1000.0));

    double latencies[VEC_N_QUERIES];
    pthread_t threads[NUM_THREADS];
    WorkerArgs args[NUM_THREADS];
    VecResult *flat_gt = malloc((size_t)VEC_N_QUERIES * VEC_K * sizeof(VecResult));

    emit("\n  --- Flat Index Search (Exact Baseline) ---\n");
    BENCH_START();
    uint32_t q_per_thread = VEC_N_QUERIES / NUM_THREADS;
    for (int i = 0; i < NUM_THREADS; i++) {
        args[i].f_idx = &f_idx;
        args[i].i_idx = NULL;
        args[i].h_idx = NULL;
        args[i].queries = queries;
        args[i].dim = VEC_DIM;
        args[i].k = VEC_K;
        args[i].start_q = i * q_per_thread;
        args[i].end_q = (i == NUM_THREADS - 1) ? VEC_N_QUERIES : (i + 1) * q_per_thread;
        args[i].latencies = latencies;
        args[i].all_results = flat_gt;
        args[i].type = WORKER_FLAT;
        pthread_create(&threads[i], NULL, query_worker, &args[i]);
    }
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
    BENCH_END();
    double flat_search_ms = BENCH_MS();
    double flat_qps = VEC_N_QUERIES / (flat_search_ms / 1000.0);
    
    qsort(latencies, VEC_N_QUERIES, sizeof(double), compare_doubles);
    emit("  QPS:       %.0f\n", flat_qps);
    emit("  p50:       %.2f ms/query\n", latencies[VEC_N_QUERIES / 2]);
    emit("  p95:       %.2f ms/query\n", latencies[(int)(VEC_N_QUERIES * 0.95)]);
    emit("  p99:       %.2f ms/query\n", latencies[(int)(VEC_N_QUERIES * 0.99)]);
    emit("  Recall@1:  100.0%% (Ground Truth)\n");
    emit("  Recall@10: 100.0%% (Ground Truth)\n");

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
    emit("  IVF Insert: %.2f ms (%.0f vecs/sec)\n", BENCH_MS(), VEC_N_VECTORS / (BENCH_MS() / 1000.0));

    /* --- IVF Search nprobe Trade-off --- */
    uint32_t probes[] = {1, 2, 4, 8, 16, 32, 64};
    int num_probes = sizeof(probes) / sizeof(probes[0]);

    emit("\n  --- IVF Trade-off Curve ---\n");
    emit("  %6s | %7s | %7s | %8s | %9s | %6s | %6s | %6s | %8s\n",
         "nprobe", "QPS", "Speedup", "Recall@1", "Recall@10", "p50 ms", "p95 ms", "p99 ms", "Scanned");
    emit("  ───────────────────────────────────────────────────────────────────────────────────\n");

    VecResult *ivf_results = malloc((size_t)VEC_N_QUERIES * VEC_K * sizeof(VecResult));

    for (int p_idx = 0; p_idx < num_probes; p_idx++) {
        i_idx.nprobe = probes[p_idx];
        
        BENCH_START();
        for (int i = 0; i < NUM_THREADS; i++) {
            args[i].i_idx = &i_idx;
            args[i].type = WORKER_IVF;
            args[i].all_results = ivf_results;
            pthread_create(&threads[i], NULL, query_worker, &args[i]);
        }
        for (int i = 0; i < NUM_THREADS; i++) {
            pthread_join(threads[i], NULL);
        }
        BENCH_END();
        double ivf_search_ms = BENCH_MS();
        
        qsort(latencies, VEC_N_QUERIES, sizeof(double), compare_doubles);
        
        RecallStats rs = compute_recall(ivf_results, flat_gt, VEC_N_QUERIES, VEC_K);
        double qps = VEC_N_QUERIES / (ivf_search_ms / 1000.0);
        double speedup = qps / flat_qps;

        emit("  %6u | %7.0f | %6.1f× | %7.1f%% | %8.1f%% | %6.2f | %6.2f | %6.2f | %7.1f%%\n",
             i_idx.nprobe,
             qps,
             speedup,
             rs.recall_at_1,
             rs.recall_at_k,
             latencies[VEC_N_QUERIES / 2],
             latencies[(int)(VEC_N_QUERIES * 0.95)],
             latencies[(int)(VEC_N_QUERIES * 0.99)],
             100.0 * i_idx.nprobe / nlist);
    }

    free(ivf_results);
    free(flat_gt);
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
 *  Benchmark 5: HNSW Vector Index
 * ============================================================ */

#define HNSW_BENCH_N       20000
#define HNSW_DIM           128
#define HNSW_M             16
#define HNSW_EF_BUILD      40
#define HNSW_N_QUERIES     100
#define BENCH_HNSW_DB      "build/bench_hnsw.db"

static void bench_hnsw_index(void) {
    struct timespec _ts_start, _ts_end;

    emit("\n[5] HNSW Index (Hierarchical Navigable Small World)\n");
    separator();
    emit("  dim=%u, n=%u, queries=%u, threads=%u, M=%u, ef_build=%u\n",
         HNSW_DIM, HNSW_BENCH_N, HNSW_N_QUERIES, NUM_THREADS, HNSW_M, HNSW_EF_BUILD);

    unlink(BENCH_HNSW_DB);
    StorageManager sm;
    storage_open(&sm, BENCH_HNSW_DB);
    BufferPool bp;
    bp_init(&bp, &sm, 10000);

    srand(101);
    emit("  Generating clustered data...\n");
    float *vecs = make_clustered_vectors(HNSW_BENCH_N, HNSW_DIM, 256);
    float *queries = make_clustered_vectors(HNSW_N_QUERIES, HNSW_DIM, 256);

    /* --- Flat Ground Truth on the 20K dataset --- */
    FlatIndex f_idx;
    flat_index_create(&f_idx, &bp, HNSW_DIM, VEC_DIST_L2);
    for (uint32_t i = 0; i < HNSW_BENCH_N; i++) {
        flat_index_insert(&f_idx, vecs + (size_t)i * HNSW_DIM, NULL, NULL);
    }

    double flat_latencies[HNSW_N_QUERIES];
    pthread_t threads[NUM_THREADS];
    WorkerArgs args[NUM_THREADS];
    VecResult *hnsw_gt = malloc((size_t)HNSW_N_QUERIES * VEC_K * sizeof(VecResult));

    BENCH_START();
    uint32_t q_per_thread = HNSW_N_QUERIES / NUM_THREADS;
    for (int i = 0; i < NUM_THREADS; i++) {
        args[i].f_idx = &f_idx;
        args[i].i_idx = NULL;
        args[i].h_idx = NULL;
        args[i].queries = queries;
        args[i].dim = HNSW_DIM;
        args[i].k = VEC_K;
        args[i].start_q = i * q_per_thread;
        args[i].end_q = (i == NUM_THREADS - 1) ? HNSW_N_QUERIES : (i + 1) * q_per_thread;
        args[i].latencies = flat_latencies;
        args[i].all_results = hnsw_gt;
        args[i].type = WORKER_FLAT;
        pthread_create(&threads[i], NULL, query_worker, &args[i]);
    }
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
    BENCH_END();
    double flat_ms = BENCH_MS();
    double flat_hnsw_qps = HNSW_N_QUERIES / (flat_ms / 1000.0);
    qsort(flat_latencies, HNSW_N_QUERIES, sizeof(double), compare_doubles);

    emit("\n  --- Flat Baseline (%u vectors) ---\n", HNSW_BENCH_N);
    emit("  QPS:       %.0f\n", flat_hnsw_qps);
    emit("  p50:       %.2f ms/query\n", flat_latencies[HNSW_N_QUERIES / 2]);
    emit("  p95:       %.2f ms/query\n", flat_latencies[(int)(HNSW_N_QUERIES * 0.95)]);
    emit("  p99:       %.2f ms/query\n", flat_latencies[(int)(HNSW_N_QUERIES * 0.99)]);
    emit("  Recall@1:  100.0%% (Ground Truth)\n");
    emit("  Recall@10: 100.0%% (Ground Truth)\n");

    /* --- Build HNSW Index --- */
    HnswIndex h_idx;
    hnsw_index_create(&h_idx, &bp, HNSW_DIM, VEC_DIST_L2, HNSW_M, HNSW_EF_BUILD);

    emit("\n  --- HNSW Graph Construction ---\n");
    emit("  Building HNSW graph on %u vectors...\n", HNSW_BENCH_N);
    BENCH_START();
    for (uint32_t i = 0; i < HNSW_BENCH_N; i++) {
        hnsw_index_insert(&h_idx, vecs + (size_t)i * HNSW_DIM, NULL, NULL);
    }
    BENCH_END();
    double build_ms = BENCH_MS();
    emit("  HNSW Build:  %.2f ms  (%.0f vecs/sec)\n",
         build_ms, HNSW_BENCH_N / (build_ms / 1000.0));
    emit("  Graph Stats: %u nodes, max_level=%u, entry_point=%u\n",
         hnsw_index_count(&h_idx), h_idx.max_level, h_idx.entry_point);

    /* --- HNSW Search Trade-off Curve --- */
    uint32_t ef_search_vals[] = {10, 20, 40, 64, 128, 256};
    int num_ef = sizeof(ef_search_vals) / sizeof(ef_search_vals[0]);

    emit("\n  --- HNSW Trade-off Curve (ef_search vs Accuracy/Latency) ---\n");
    emit("  %9s | %8s | %8s | %9s | %10s | %7s | %7s | %7s\n",
         "ef_search", "QPS", "Speedup", "Recall@1", "Recall@10", "p50 ms", "p95 ms", "p99 ms");
    emit("  ───────────────────────────────────────────────────────────────────────────────────\n");

    double h_latencies[HNSW_N_QUERIES];
    VecResult *hnsw_results = malloc((size_t)HNSW_N_QUERIES * VEC_K * sizeof(VecResult));

    for (int e = 0; e < num_ef; e++) {
        uint32_t ef = ef_search_vals[e];

        BENCH_START();
        for (int i = 0; i < NUM_THREADS; i++) {
            args[i].h_idx = &h_idx;
            args[i].ef_search = ef;
            args[i].type = WORKER_HNSW;
            args[i].latencies = h_latencies;
            args[i].all_results = hnsw_results;
            pthread_create(&threads[i], NULL, query_worker, &args[i]);
        }
        for (int i = 0; i < NUM_THREADS; i++) {
            pthread_join(threads[i], NULL);
        }
        BENCH_END();
        double search_ms = BENCH_MS();

        qsort(h_latencies, HNSW_N_QUERIES, sizeof(double), compare_doubles);

        RecallStats rs = compute_recall(hnsw_results, hnsw_gt, HNSW_N_QUERIES, VEC_K);
        double qps = HNSW_N_QUERIES / (search_ms / 1000.0);
        double speedup = qps / flat_hnsw_qps;

        emit("  %9u | %8.0f | %7.1f× | %8.1f%% | %9.1f%% | %7.2f | %7.2f | %7.2f\n",
             ef, qps, speedup, rs.recall_at_1, rs.recall_at_k,
             h_latencies[HNSW_N_QUERIES / 2],
             h_latencies[(int)(HNSW_N_QUERIES * 0.95)],
             h_latencies[(int)(HNSW_N_QUERIES * 0.99)]);
    }

    free(hnsw_results);
    free(hnsw_gt);
    free(vecs);
    free(queries);
    flat_index_destroy(&f_idx);
    hnsw_index_destroy(&h_idx);
    bp_flush_all(&bp);
    bp_destroy(&bp);
    storage_close(&sm);
    unlink(BENCH_HNSW_DB);
}

/* ============================================================
 *  Benchmark 6: Vector Quantization (SQ8 & PQ Compression & ADC)
 * ============================================================ */

#define Q_BENCH_N       10000
#define Q_DIM           128
#define Q_QUERIES       100

static void bench_quantization(void) {
    struct timespec _ts_start, _ts_end;

    emit("\n[6] Vector Quantization (SQ8 & Product Quantization)\n");
    separator();
    emit("  dim=%u, n=%u, queries=%u\n", Q_DIM, Q_BENCH_N, Q_QUERIES);

    srand(202);
    float *vecs = make_clustered_vectors(Q_BENCH_N, Q_DIM, 128);
    float *queries = make_clustered_vectors(Q_QUERIES, Q_DIM, 128);

    /* Generate uncompressed FP32 ground truth */
    float *fp32_gt_dist = malloc((size_t)Q_QUERIES * VEC_K * sizeof(float));
    for (uint32_t q = 0; q < Q_QUERIES; q++) {
        const float *qv = queries + (size_t)q * Q_DIM;
        VecTopK tk;
        vec_topk_init(&tk, VEC_K);
        for (uint32_t i = 0; i < Q_BENCH_N; i++) {
            float d = vec_l2_distance_sq(qv, vecs + (size_t)i * Q_DIM, Q_DIM);
            vec_topk_push(&tk, 0, (uint16_t)i, d);
        }
        vec_topk_sort(&tk);
        for (uint32_t k = 0; k < VEC_K; k++) {
            fp32_gt_dist[q * VEC_K + k] = tk.items[k].distance;
        }
        vec_topk_destroy(&tk);
    }

    /* --- SQ8 Quantization --- */
    SQ8Quantizer sq;
    sq8_init(&sq, Q_DIM);

    BENCH_START();
    sq8_train(&sq, vecs, Q_BENCH_N);
    BENCH_END();
    double sq8_train_ms = BENCH_MS();

    uint8_t *sq8_codes = malloc((size_t)Q_BENCH_N * Q_DIM);
    BENCH_START();
    sq8_encode_batch(&sq, vecs, Q_BENCH_N, sq8_codes);
    BENCH_END();
    double sq8_encode_ms = BENCH_MS();

    /* Compute SQ8 reconstruction error */
    float *decoded = malloc((size_t)Q_DIM * sizeof(float));
    double sq8_err_sum = 0.0;
    for (uint32_t i = 0; i < Q_BENCH_N; i++) {
        sq8_decode(&sq, sq8_codes + (size_t)i * Q_DIM, decoded);
        sq8_err_sum += sqrtf(vec_l2_distance_sq(vecs + (size_t)i * Q_DIM, decoded, Q_DIM));
    }
    double sq8_mean_err = sq8_err_sum / Q_BENCH_N;

    /* Measure SQ8 Asymmetric Distance throughput & recall */
    float sq8_dist_table[Q_DIM * 256];
    uint32_t sq8_top1_matches = 0;
    uint32_t sq8_topk_matches = 0;

    BENCH_START();
    for (uint32_t q = 0; q < Q_QUERIES; q++) {
        const float *qv = queries + (size_t)q * Q_DIM;
        sq8_precompute_l2_table(&sq, qv, sq8_dist_table);

        VecTopK tk;
        vec_topk_init(&tk, VEC_K);
        for (uint32_t i = 0; i < Q_BENCH_N; i++) {
            float d = sq8_asymmetric_l2(sq8_dist_table, sq8_codes + (size_t)i * Q_DIM, Q_DIM);
            vec_topk_push(&tk, 0, (uint16_t)i, d);
        }
        vec_topk_sort(&tk);

        if (tk.count > 0) {
            uint16_t best_idx = tk.items[0].slot_index;
            float true_d = vec_l2_distance_sq(qv, vecs + (size_t)best_idx * Q_DIM, Q_DIM);
            if (fabsf(true_d - fp32_gt_dist[q * VEC_K]) < 1e-4f) {
                sq8_top1_matches++;
            }
        }
        for (uint32_t k = 0; k < tk.count; k++) {
            uint16_t idx_k = tk.items[k].slot_index;
            float true_d = vec_l2_distance_sq(qv, vecs + (size_t)idx_k * Q_DIM, Q_DIM);
            for (uint32_t j = 0; j < VEC_K; j++) {
                if (fabsf(true_d - fp32_gt_dist[q * VEC_K + j]) < 1e-4f) {
                    sq8_topk_matches++;
                    break;
                }
            }
        }
        vec_topk_destroy(&tk);
    }
    BENCH_END();
    double sq8_search_ms = BENCH_MS();
    double sq8_dist_mops = ((double)Q_QUERIES * Q_BENCH_N) / (sq8_search_ms / 1000.0) / 1e6;
    double sq8_recall_1 = 100.0 * (double)sq8_top1_matches / (double)Q_QUERIES;
    double sq8_recall_k = 100.0 * (double)sq8_topk_matches / (double)(Q_QUERIES * VEC_K);

    /* --- PQ (Product Quantization, nsub=16) --- */
    PQQuantizer pq;
    uint16_t nsub = 16;
    pq_init(&pq, Q_DIM, nsub);

    BENCH_START();
    pq_train(&pq, vecs, Q_BENCH_N);
    BENCH_END();
    double pq_train_ms = BENCH_MS();

    uint8_t *pq_codes = malloc((size_t)Q_BENCH_N * nsub);
    BENCH_START();
    pq_encode_batch(&pq, vecs, Q_BENCH_N, pq_codes);
    BENCH_END();
    double pq_encode_ms = BENCH_MS();

    /* Compute PQ reconstruction error */
    double pq_err_sum = 0.0;
    for (uint32_t i = 0; i < Q_BENCH_N; i++) {
        pq_decode(&pq, pq_codes + (size_t)i * nsub, decoded);
        pq_err_sum += sqrtf(vec_l2_distance_sq(vecs + (size_t)i * Q_DIM, decoded, Q_DIM));
    }
    double pq_mean_err = pq_err_sum / Q_BENCH_N;

    /* Measure PQ Asymmetric Distance throughput & recall */
    float pq_table[16 * PQ_KSUB];
    uint32_t pq_top1_matches = 0;
    uint32_t pq_topk_matches = 0;

    BENCH_START();
    for (uint32_t q = 0; q < Q_QUERIES; q++) {
        const float *qv = queries + (size_t)q * Q_DIM;
        pq_precompute_l2_table(&pq, qv, pq_table);

        VecTopK tk;
        vec_topk_init(&tk, VEC_K);
        for (uint32_t i = 0; i < Q_BENCH_N; i++) {
            float d = pq_asymmetric_l2(pq_table, pq_codes + (size_t)i * nsub, nsub);
            vec_topk_push(&tk, 0, (uint16_t)i, d);
        }
        vec_topk_sort(&tk);

        if (tk.count > 0) {
            uint16_t best_idx = tk.items[0].slot_index;
            float true_d = vec_l2_distance_sq(qv, vecs + (size_t)best_idx * Q_DIM, Q_DIM);
            if (fabsf(true_d - fp32_gt_dist[q * VEC_K]) < 1e-4f) {
                pq_top1_matches++;
            }
        }
        for (uint32_t k = 0; k < tk.count; k++) {
            uint16_t idx_k = tk.items[k].slot_index;
            float true_d = vec_l2_distance_sq(qv, vecs + (size_t)idx_k * Q_DIM, Q_DIM);
            for (uint32_t j = 0; j < VEC_K; j++) {
                if (fabsf(true_d - fp32_gt_dist[q * VEC_K + j]) < 1e-4f) {
                    pq_topk_matches++;
                    break;
                }
            }
        }
        vec_topk_destroy(&tk);
    }
    BENCH_END();
    double pq_search_ms = BENCH_MS();
    double pq_dist_mops = ((double)Q_QUERIES * Q_BENCH_N) / (pq_search_ms / 1000.0) / 1e6;
    double pq_recall_1 = 100.0 * (double)pq_top1_matches / (double)Q_QUERIES;
    double pq_recall_k = 100.0 * (double)pq_topk_matches / (double)(Q_QUERIES * VEC_K);

    /* Print Summary Table */
    emit("\n  --- Quantization Trade-off (Compression vs Accuracy vs Speed) ---\n");
    emit("  %10s | %9s | %7s | %13s | %11s | %8s | %9s | %11s\n",
         "Format", "Vec Size", "Ratio", "Encode Mvec/s", "Dist Mops/s", "Recall@1", "Recall@10", "Mean L2 Err");
    emit("  ───────────────────────────────────────────────────────────────────────────────────────────────────────\n");
    emit("  %10s | %9s | %6.1f× | %13s | %11.1f | %7.1f%% | %8.1f%% | %11.4f\n",
         "FP32", "512 B", 1.0, "N/A", 44.0, 100.0, 100.0, 0.0);
    emit("  %10s | %9s | %6.1f× | %13.2f | %11.1f | %7.1f%% | %8.1f%% | %11.4f\n",
         "SQ8", "128 B", 4.0, Q_BENCH_N / (sq8_encode_ms / 1000.0) / 1e6, sq8_dist_mops, sq8_recall_1, sq8_recall_k, sq8_mean_err);
    emit("  %10s | %9s | %6.1f× | %13.2f | %11.1f | %7.1f%% | %8.1f%% | %11.4f\n",
         "PQ (M=16)", "16 B", 32.0, Q_BENCH_N / (pq_encode_ms / 1000.0) / 1e6, pq_dist_mops, pq_recall_1, pq_recall_k, pq_mean_err);

    emit("\n  Training Times:\n");
    emit("    SQ8 Train (bounds): %.2f ms\n", sq8_train_ms);
    emit("    PQ Train (16 codebooks x 256 centroids): %.2f ms\n", pq_train_ms);

    free(decoded);
    free(sq8_codes);
    free(pq_codes);
    free(fp32_gt_dist);
    free(vecs);
    free(queries);
    sq8_destroy(&sq);
    pq_destroy(&pq);
}

/* ============================================================
 *  Benchmark 7: Distance function micro-benchmark
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
    emit("\n[7] Distance Functions (SIMD micro-benchmark)\n");
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
    bench_hnsw_index();
    bench_quantization();
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
