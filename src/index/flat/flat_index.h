#ifndef FLAT_INDEX_H
#define FLAT_INDEX_H

#include "../../buffer/buffer_pool.h"
#include "../vec_topk.h"
#include "../vec_utils.h"

/*
 * Flat (brute-force) vector index.
 *
 * Manages a growable list of vector pages.  Insert appends to the last page,
 * allocating a new page via the buffer pool when full.  Search performs a
 * sequential scan over all pages, computing distance to every stored vector.
 */
typedef struct {
    BufferPool        *bp;
    uint16_t           dim;
    VecDistanceMetric  metric;

    page_id_t         *page_ids;      /* dynamic array of page IDs */
    uint32_t           num_pages;
    uint32_t           page_cap;      /* allocated capacity of page_ids */

    uint32_t           total_vectors;
} FlatIndex;

/*
 * Create a new empty flat index.
 * Allocates the first vector page through the buffer pool.
 * Returns 0 on success, -1 on failure.
 */
int flat_index_create(FlatIndex *idx, BufferPool *bp,
                      uint16_t dim, VecDistanceMetric metric);

/* Free the in-memory page_ids array.  Does NOT deallocate pages on disk. */
void flat_index_destroy(FlatIndex *idx);

/*
 * Insert a single vector into the index.
 * If the current last page is full, a new page is allocated.
 * On success, *out_pid and *out_slot identify the stored location.
 * Returns 0 on success, -1 on failure.
 */
int flat_index_insert(FlatIndex *idx, const float *vec,
                      page_id_t *out_pid, uint16_t *out_slot);

/*
 * Delete a vector by marking its tombstone bit.
 * Returns 0 on success, -1 if not found or already deleted.
 */
int flat_index_delete(FlatIndex *idx, page_id_t pid, uint16_t slot);

/*
 * Brute-force k-nearest-neighbor search.
 * Results are written to the caller-provided array (must have room for k entries).
 * *num_results is set to the actual number returned (may be < k if index has fewer vectors).
 * Results are sorted ascending by distance.
 * Returns 0 on success, -1 on failure.
 */
int flat_index_search(FlatIndex *idx, const float *query, uint32_t k,
                      VecResult *results, uint32_t *num_results);

/* Number of vectors in the index. */
uint32_t flat_index_count(const FlatIndex *idx);

/*
 * Save the FlatIndex metadata and page list to disk via stream pages.
 * Returns 0 on success and sets *out_pid to the starting stream page ID.
 */
int flat_index_save(FlatIndex *idx, page_id_t *out_pid);

/*
 * Load the FlatIndex from a saved stream page.
 * Returns 0 on success, -1 on failure.
 */
int flat_index_load(FlatIndex *idx, BufferPool *bp, page_id_t pid);

#endif
