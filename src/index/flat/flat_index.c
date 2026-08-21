#include "flat_index.h"
#include "../vec_page.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "../../pages/stream_page.h"

#define INITIAL_PAGE_CAP 16


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

int flat_index_delete(FlatIndex *idx, page_id_t pid, uint16_t slot) {
    /* Verify pid is part of this index */
    bool found = false;
    for (uint32_t i = 0; i < idx->num_pages; i++) {
        if (idx->page_ids[i] == pid) {
            found = true;
            break;
        }
    }
    if (!found) return -1;

    Page *p = bp_fetch_page(idx->bp, pid);
    if (!p) return -1;

    int rc = vec_page_delete(p, slot);
    if (rc == 0) {
        idx->total_vectors--;
    }
    /* Unpin with dirty flag if successful, clean otherwise */
    bp_unpin_page(idx->bp, pid, (rc == 0));
    return rc;
}

int flat_index_search(FlatIndex *idx, const float *query, uint32_t k,
                      VecResult *results, uint32_t *num_results) {
    if (k == 0) {
        *num_results = 0;
        return 0;
    }

    vec_dist_fn_t dist_fn = vec_get_distance_fn(idx->metric);

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
            if (vec_page_is_deleted(p, vi)) {
                continue;
            }
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

typedef struct {
    uint16_t dim;
    uint32_t metric;
    uint32_t num_pages;
    uint32_t total_vectors;
} FlatIndexMeta;

int flat_index_save(FlatIndex *idx, page_id_t *out_pid) {
    size_t meta_size = sizeof(FlatIndexMeta);
    size_t array_size = idx->num_pages * sizeof(page_id_t);
    size_t total_size = meta_size + array_size;

    uint8_t *buf = malloc(total_size);
    if (!buf) return -1;

    FlatIndexMeta *meta = (FlatIndexMeta *)buf;
    meta->dim = idx->dim;
    meta->metric = (uint32_t)idx->metric;
    meta->num_pages = idx->num_pages;
    meta->total_vectors = idx->total_vectors;

    memcpy(buf + meta_size, idx->page_ids, array_size);

    int rc = stream_write(idx->bp, buf, total_size, out_pid);
    free(buf);
    return rc;
}

int flat_index_load(FlatIndex *idx, BufferPool *bp, page_id_t pid) {
    /* Read first page to get total size? Wait, stream_read needs expected size.
       We don't know the size until we read FlatIndexMeta.
       Let's read just the meta first. */
    FlatIndexMeta meta;
    if (stream_read(bp, pid, &meta, sizeof(FlatIndexMeta)) < 0) {
        printf("flat_index_load: failed to read meta from pid %u\n", pid);
        return -1;
    }
    printf("flat_index_load: read meta: dim=%u metric=%u num_pages=%u vectors=%u\n", meta.dim, meta.metric, meta.num_pages, meta.total_vectors);

    size_t total_size = sizeof(FlatIndexMeta) + meta.num_pages * sizeof(page_id_t);
    uint8_t *buf = malloc(total_size);
    if (!buf) {
        printf("flat_index_load: failed to malloc %zu bytes\n", total_size);
        return -1;
    }

    if (stream_read(bp, pid, buf, total_size) < 0) {
        printf("flat_index_load: failed to read %zu bytes\n", total_size);
        free(buf);
        return -1;
    }

    memset(idx, 0, sizeof(FlatIndex));
    idx->bp = bp;
    idx->dim = meta.dim;
    idx->metric = (VecDistanceMetric)meta.metric;
    idx->num_pages = meta.num_pages;
    idx->page_cap = meta.num_pages > INITIAL_PAGE_CAP ? meta.num_pages : INITIAL_PAGE_CAP;
    idx->total_vectors = meta.total_vectors;

    idx->page_ids = malloc(idx->page_cap * sizeof(page_id_t));
    if (!idx->page_ids) {
        free(buf);
        return -1;
    }

    memcpy(idx->page_ids, buf + sizeof(FlatIndexMeta), meta.num_pages * sizeof(page_id_t));
    free(buf);
    return 0;
}
