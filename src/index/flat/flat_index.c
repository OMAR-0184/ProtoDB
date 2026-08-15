#include "flat_index.h"
#include "../vec_page.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#define INITIAL_PAGE_CAP 16

/* ---- distance function dispatch ---- */

typedef float (*dist_fn_t)(const float *, const float *, uint32_t);

static dist_fn_t get_distance_fn(VecDistanceMetric metric) {
    switch (metric) {
        case VEC_DIST_L2:             return vec_l2_distance_sq;
        case VEC_DIST_COSINE:         return vec_cosine_distance;
        case VEC_DIST_INNER_PRODUCT:  return vec_inner_product_distance;
        default:                      return vec_l2_distance_sq;
    }
}

/* ---- internal helpers ---- */

static int grow_page_array(FlatIndex *idx) {
    uint32_t new_cap = idx->page_cap * 2;
    page_id_t *new_arr = realloc(idx->page_ids, new_cap * sizeof(page_id_t));
    if (!new_arr)
        return -1;
    idx->page_ids = new_arr;
    idx->page_cap = new_cap;
    return 0;
}

static int alloc_new_vector_page(FlatIndex *idx) {
    if (idx->num_pages >= idx->page_cap) {
        if (grow_page_array(idx) < 0)
            return -1;
    }

    page_id_t new_pid;
    Page *p = bp_new_page(idx->bp, &new_pid);
    if (!p)
        return -1;

    /* Initialize as a vector page */
    vec_page_init(p, new_pid, idx->dim);

    /* Mark dirty and unpin */
    bp_unpin_page(idx->bp, new_pid, true);

    idx->page_ids[idx->num_pages] = new_pid;
    idx->num_pages++;
    return 0;
}

/* ---- public API ---- */

int flat_index_create(FlatIndex *idx, BufferPool *bp,
                      uint16_t dim, VecDistanceMetric metric) {
    if (dim == 0 || dim > VECTOR_MAX_DIM)
        return -1;

    memset(idx, 0, sizeof(FlatIndex));
    idx->bp     = bp;
    idx->dim    = dim;
    idx->metric = metric;

    idx->page_ids = malloc(INITIAL_PAGE_CAP * sizeof(page_id_t));
    if (!idx->page_ids)
        return -1;
    idx->page_cap  = INITIAL_PAGE_CAP;
    idx->num_pages = 0;

    /* Allocate the first vector page */
    if (alloc_new_vector_page(idx) < 0) {
        free(idx->page_ids);
        return -1;
    }

    idx->total_vectors = 0;
    return 0;
}

void flat_index_destroy(FlatIndex *idx) {
    free(idx->page_ids);
    idx->page_ids     = NULL;
    idx->num_pages    = 0;
    idx->page_cap     = 0;
    idx->total_vectors = 0;
}

int flat_index_insert(FlatIndex *idx, const float *vec,
                      page_id_t *out_pid, uint16_t *out_slot) {
    if (idx->num_pages == 0)
        return -1;

    /* Try to append to the last page */
    page_id_t last_pid = idx->page_ids[idx->num_pages - 1];
    Page *p = bp_fetch_page(idx->bp, last_pid);
    if (!p)
        return -1;

    int rc = vec_page_append(p, vec);
    if (rc == 0) {
        /* Success — record the slot index (it's count-1 after append) */
        uint16_t slot = vec_page_count(p) - 1;
        bp_unpin_page(idx->bp, last_pid, true);

        if (out_pid)  *out_pid  = last_pid;
        if (out_slot) *out_slot = slot;
        idx->total_vectors++;
        return 0;
    }

    /* Page was full — unpin it and allocate a new one */
    bp_unpin_page(idx->bp, last_pid, false);

    if (alloc_new_vector_page(idx) < 0)
        return -1;

    /* Insert into the new page */
    page_id_t new_pid = idx->page_ids[idx->num_pages - 1];
    p = bp_fetch_page(idx->bp, new_pid);
    if (!p)
        return -1;

    rc = vec_page_append(p, vec);
    if (rc < 0) {
        bp_unpin_page(idx->bp, new_pid, false);
        return -1;
    }

    uint16_t slot = vec_page_count(p) - 1;
    bp_unpin_page(idx->bp, new_pid, true);

    if (out_pid)  *out_pid  = new_pid;
    if (out_slot) *out_slot = slot;
    idx->total_vectors++;
    return 0;
}

int flat_index_search(FlatIndex *idx, const float *query, uint32_t k,
                      VecResult *results, uint32_t *num_results) {
    if (k == 0) {
        *num_results = 0;
        return 0;
    }

    dist_fn_t dist_fn = get_distance_fn(idx->metric);

    VecTopK topk;
    if (vec_topk_init(&topk, k) < 0)
        return -1;

    /* Sequential scan over all vector pages */
    for (uint32_t pi = 0; pi < idx->num_pages; pi++) {
        page_id_t pid = idx->page_ids[pi];
        Page *p = bp_fetch_page(idx->bp, pid);
        if (!p) {
            vec_topk_destroy(&topk);
            return -1;
        }

        uint16_t count = vec_page_count(p);
        for (uint16_t vi = 0; vi < count; vi++) {
            const float *vec = vec_page_get(p, vi);
            float dist = dist_fn(query, vec, idx->dim);
            vec_topk_push(&topk, pid, vi, dist);
        }

        bp_unpin_page(idx->bp, pid, false);
    }

    /* Sort results and copy out */
    vec_topk_sort(&topk);

    uint32_t n = topk.count;
    memcpy(results, topk.items, n * sizeof(VecResult));
    *num_results = n;

    vec_topk_destroy(&topk);
    return 0;
}

uint32_t flat_index_count(const FlatIndex *idx) {
    return idx->total_vectors;
}
