#include "../src/index/vec_utils.h"
#include "../src/index/vec_topk.h"
#include "../src/index/vec_page.h"
#include "../src/index/flat/flat_index.h"
#include "../src/pages/storage_mgr.h"
#include "../src/buffer/buffer_pool.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <unistd.h>

#define TEST_DB_PATH "/tmp/test_flat_index.db"

/* ============================================================
 *  Distance function tests
 * ============================================================ */

static void test_vec_distances(void) {
    float a[] = {1.0f, 0.0f, 0.0f};
    float b[] = {0.0f, 1.0f, 0.0f};

    /* L2² between (1,0,0) and (0,1,0) = 1+1 = 2 */
    float d = vec_l2_distance_sq(a, b, 3);
    assert(fabsf(d - 2.0f) < 1e-6f);

    /* L2 distance = sqrt(2) */
    d = vec_l2_distance(a, b, 3);
    assert(fabsf(d - sqrtf(2.0f)) < 1e-6f);

    /* Cosine distance: cos(90°) = 0, so distance = 1.0 */
    d = vec_cosine_distance(a, b, 3);
    assert(fabsf(d - 1.0f) < 1e-6f);

    /* Cosine distance of a vector with itself = 0 */
    d = vec_cosine_distance(a, a, 3);
    assert(fabsf(d) < 1e-6f);

    /* Inner product distance: -(1*0 + 0*1 + 0*0) = 0 */
    d = vec_inner_product_distance(a, b, 3);
    assert(fabsf(d) < 1e-6f);

    /* Inner product distance of (1,0,0) with itself = -(1) = -1 */
    d = vec_inner_product_distance(a, a, 3);
    assert(fabsf(d - (-1.0f)) < 1e-6f);

    printf("  PASS: test_vec_distances\n");
}

static void test_vec_normalize(void) {
    float v[] = {3.0f, 4.0f, 0.0f};
    vec_normalize(v, 3);

    float norm = sqrtf(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    assert(fabsf(norm - 1.0f) < 1e-6f);
    assert(fabsf(v[0] - 0.6f) < 1e-6f);
    assert(fabsf(v[1] - 0.8f) < 1e-6f);

    printf("  PASS: test_vec_normalize\n");
}

/* ============================================================
 *  Vector page tests
 * ============================================================ */

static void test_vec_page_pack(void) {
    Page p;
    uint16_t dim = 4;
    vec_page_init(&p, 1, dim);

    assert(vec_page_count(&p) == 0);
    assert(vec_page_dim(&p) == dim);

    float v1[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float v2[] = {5.0f, 6.0f, 7.0f, 8.0f};

    assert(vec_page_append(&p, v1) == 0);
    assert(vec_page_append(&p, v2) == 0);
    assert(vec_page_count(&p) == 2);

    const float *r1 = vec_page_get(&p, 0);
    const float *r2 = vec_page_get(&p, 1);
    assert(r1 != NULL && r2 != NULL);

    for (int i = 0; i < 4; i++) {
        assert(fabsf(r1[i] - v1[i]) < 1e-6f);
        assert(fabsf(r2[i] - v2[i]) < 1e-6f);
    }

    /* Out of bounds */
    assert(vec_page_get(&p, 2) == NULL);

    printf("  PASS: test_vec_page_pack\n");
}

static void test_vec_page_capacity(void) {
    Page p;
    uint16_t dim = 4;
    vec_page_init(&p, 2, dim);

    uint16_t cap = vec_page_capacity(dim);
    assert(cap > 0);

    /* Fill the page to capacity */
    float vec[4] = {0};
    for (uint16_t i = 0; i < cap; i++) {
        vec[0] = (float)i;
        assert(vec_page_append(&p, vec) == 0);
    }

    /* One more should fail */
    assert(vec_page_append(&p, vec) == -1);
    assert(vec_page_count(&p) == cap);

    printf("  PASS: test_vec_page_capacity (cap=%u for dim=%u)\n", cap, dim);
}

/* ============================================================
 *  Top-K heap tests
 * ============================================================ */

static void test_topk_heap(void) {
    VecTopK tk;
    assert(vec_topk_init(&tk, 3) == 0);

    /* Push 5 items, heap should keep the 3 smallest */
    vec_topk_push(&tk, 0, 0, 5.0f);
    vec_topk_push(&tk, 0, 1, 1.0f);
    vec_topk_push(&tk, 0, 2, 3.0f);
    vec_topk_push(&tk, 0, 3, 0.5f);
    vec_topk_push(&tk, 0, 4, 2.0f);

    assert(tk.count == 3);

    vec_topk_sort(&tk);

    /* Should be: 0.5, 1.0, 2.0 */
    assert(fabsf(tk.items[0].distance - 0.5f) < 1e-6f);
    assert(fabsf(tk.items[1].distance - 1.0f) < 1e-6f);
    assert(fabsf(tk.items[2].distance - 2.0f) < 1e-6f);

    vec_topk_destroy(&tk);
    printf("  PASS: test_topk_heap\n");
}

static void test_topk_underfilled(void) {
    VecTopK tk;
    assert(vec_topk_init(&tk, 10) == 0);

    /* Push only 2 items into a k=10 heap */
    vec_topk_push(&tk, 0, 0, 3.0f);
    vec_topk_push(&tk, 0, 1, 1.0f);

    assert(tk.count == 2);

    vec_topk_sort(&tk);
    assert(fabsf(tk.items[0].distance - 1.0f) < 1e-6f);
    assert(fabsf(tk.items[1].distance - 3.0f) < 1e-6f);

    vec_topk_destroy(&tk);
    printf("  PASS: test_topk_underfilled\n");
}

/* ============================================================
 *  Flat index integration tests
 * ============================================================ */

/* Helper: set up storage + buffer pool + flat index */
typedef struct {
    StorageManager sm;
    BufferPool     bp;
    FlatIndex      idx;
} TestEnv;

static void env_setup(TestEnv *env, uint16_t dim, VecDistanceMetric metric) {
    unlink(TEST_DB_PATH);
    assert(storage_open(&env->sm, TEST_DB_PATH) == 0);
    assert(bp_init(&env->bp, &env->sm, 64) == 0);
    assert(flat_index_create(&env->idx, &env->bp, dim, metric) == 0);
}

static void env_teardown(TestEnv *env) {
    flat_index_destroy(&env->idx);
    bp_flush_all(&env->bp);
    bp_destroy(&env->bp);
    storage_close(&env->sm);
    unlink(TEST_DB_PATH);
}

static void test_flat_insert_and_search(void) {
    TestEnv env;
    env_setup(&env, 3, VEC_DIST_L2);

    /* Insert 5 vectors */
    float vecs[5][3] = {
        {0.0f, 0.0f, 0.0f},
        {1.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f},
        {10.0f, 10.0f, 10.0f},
        {0.1f, 0.1f, 0.0f},
    };

    for (int i = 0; i < 5; i++) {
        page_id_t pid;
        uint16_t slot;
        assert(flat_index_insert(&env.idx, vecs[i], &pid, &slot) == 0);
    }
    assert(flat_index_count(&env.idx) == 5);

    /* Query for nearest to origin — should be vec[0] = (0,0,0) */
    float query[] = {0.0f, 0.0f, 0.0f};
    VecResult results[3];
    uint32_t num;
    assert(flat_index_search(&env.idx, query, 3, NULL, NULL, results, &num) == 0);
    assert(num == 3);

    /* Closest should be vec[0] with distance 0 (L2² = 0) */
    assert(fabsf(results[0].distance) < 1e-6f);

    /* Second closest should be vec[4] = (0.1, 0.1, 0) with L2² = 0.02 */
    assert(fabsf(results[1].distance - 0.02f) < 1e-4f);

    env_teardown(&env);
    printf("  PASS: test_flat_insert_and_search\n");
}

static void test_flat_multipage(void) {
    TestEnv env;
    uint16_t dim = 128;
    env_setup(&env, dim, VEC_DIST_L2);

    uint16_t cap = vec_page_capacity(dim);
    /* Insert enough vectors to fill at least 3 pages */
    uint32_t total = (uint32_t)cap * 3 + 1;

    float *vec = calloc(dim, sizeof(float));
    assert(vec != NULL);

    for (uint32_t i = 0; i < total; i++) {
        /* Each vector has a unique first element */
        vec[0] = (float)i;
        assert(flat_index_insert(&env.idx, vec, NULL, NULL) == 0);
    }

    assert(flat_index_count(&env.idx) == total);
    assert(env.idx.num_pages >= 4);  /* at least 4 pages (first + 3 full) */

    /* Search for vec closest to (0, 0, ..., 0) — should be vec[0] */
    memset(vec, 0, dim * sizeof(float));
    VecResult results[1];
    uint32_t num;
    assert(flat_index_search(&env.idx, vec, 1, NULL, NULL, results, &num) == 0);
    assert(num == 1);
    assert(fabsf(results[0].distance) < 1e-6f);

    /* Search for vec closest to (5.0, 0, ..., 0) — should be vec[5] */
    vec[0] = 5.0f;
    assert(flat_index_search(&env.idx, vec, 1, NULL, NULL, results, &num) == 0);
    assert(num == 1);
    assert(fabsf(results[0].distance) < 1e-6f);

    free(vec);
    uint32_t np = env.idx.num_pages;
    env_teardown(&env);
    printf("  PASS: test_flat_multipage (inserted %u vectors across %u pages)\n",
           total, np);
}

static void test_flat_cosine_metric(void) {
    TestEnv env;
    env_setup(&env, 3, VEC_DIST_COSINE);

    /* Insert vectors in different directions */
    float v1[] = {1.0f, 0.0f, 0.0f};   /* pointing right */
    float v2[] = {0.0f, 1.0f, 0.0f};   /* pointing up */
    float v3[] = {0.9f, 0.1f, 0.0f};   /* mostly right */

    assert(flat_index_insert(&env.idx, v1, NULL, NULL) == 0);
    assert(flat_index_insert(&env.idx, v2, NULL, NULL) == 0);
    assert(flat_index_insert(&env.idx, v3, NULL, NULL) == 0);

    /* Query with (1,0,0) — closest by cosine should be v1 (distance 0), then v3 */
    float query[] = {1.0f, 0.0f, 0.0f};
    VecResult results[3];
    uint32_t num;
    assert(flat_index_search(&env.idx, query, 3, NULL, NULL, results, &num) == 0);
    assert(num == 3);

    /* First result should be exact match (cosine distance ≈ 0) */
    assert(fabsf(results[0].distance) < 1e-6f);

    /* Second result should be v3 (close to right), not v2 (perpendicular) */
    assert(results[1].distance < results[2].distance);

    env_teardown(&env);
    printf("  PASS: test_flat_cosine_metric\n");
}

static void test_flat_inner_product(void) {
    TestEnv env;
    env_setup(&env, 3, VEC_DIST_INNER_PRODUCT);

    float v1[] = {1.0f, 0.0f, 0.0f};
    float v2[] = {5.0f, 0.0f, 0.0f};   /* high dot product with query */
    float v3[] = {0.1f, 0.0f, 0.0f};

    assert(flat_index_insert(&env.idx, v1, NULL, NULL) == 0);
    assert(flat_index_insert(&env.idx, v2, NULL, NULL) == 0);
    assert(flat_index_insert(&env.idx, v3, NULL, NULL) == 0);

    /* Query (1,0,0): IP distances are -1, -5, -0.1.
     * Most similar (highest dot product) = v2, IP distance = -5 (smallest). */
    float query[] = {1.0f, 0.0f, 0.0f};
    VecResult results[3];
    uint32_t num;
    assert(flat_index_search(&env.idx, query, 3, NULL, NULL, results, &num) == 0);
    assert(num == 3);

    assert(fabsf(results[0].distance - (-5.0f)) < 1e-6f);
    assert(fabsf(results[1].distance - (-1.0f)) < 1e-6f);
    assert(fabsf(results[2].distance - (-0.1f)) < 1e-5f);

    env_teardown(&env);
    printf("  PASS: test_flat_inner_product\n");
}

static void test_flat_delete(void) {
    TestEnv env;
    env_setup(&env, 4, VEC_DIST_L2);

    float v1[4] = {1.0, 1.0, 1.0, 1.0};
    float v2[4] = {2.0, 2.0, 2.0, 2.0};
    float v3[4] = {3.0, 3.0, 3.0, 3.0};

    page_id_t p1, p2, p3;
    uint16_t s1, s2, s3;
    assert(flat_index_insert(&env.idx, v1, &p1, &s1) == 0);
    assert(flat_index_insert(&env.idx, v2, &p2, &s2) == 0);
    assert(flat_index_insert(&env.idx, v3, &p3, &s3) == 0);

    assert(flat_index_count(&env.idx) == 3);

    /* Search before deletion */
    float q[4] = {2.1, 2.1, 2.1, 2.1};
    VecResult res[3];
    uint32_t n;
    assert(flat_index_search(&env.idx, q, 3, NULL, NULL, res, &n) == 0);
    assert(n == 3);
    assert(res[0].page_id == p2 && res[0].slot_index == s2); /* closest */

    /* Delete v2 */
    assert(flat_index_delete(&env.idx, p2, s2) == 0);
    assert(flat_index_count(&env.idx) == 2);

    /* Double delete fails */
    assert(flat_index_delete(&env.idx, p2, s2) < 0);

    /* Search after deletion */
    assert(flat_index_search(&env.idx, q, 3, NULL, NULL, res, &n) == 0);
    assert(n == 2);
    /* Now v3 or v1 should be returned, but v2 is gone */
    assert((res[0].page_id != p2) || (res[0].slot_index != s2));
    assert((res[1].page_id != p2) || (res[1].slot_index != s2));

    env_teardown(&env);
    printf("  PASS: test_flat_delete\n");
}

/* ============================================================
 *  Main
 * ============================================================ */

int main(void) {
    printf("=== Flat Vector Index Tests ===\n");

    /* Unit tests */
    test_vec_distances();
    test_vec_normalize();
    test_vec_page_pack();
    test_vec_page_capacity();
    test_topk_heap();
    test_topk_underfilled();

    /* Integration tests */
    test_flat_insert_and_search();
    test_flat_multipage();
    test_flat_cosine_metric();
    test_flat_inner_product();
    test_flat_delete();

    printf("All flat vector index tests passed.\n\n");
    return 0;
}
