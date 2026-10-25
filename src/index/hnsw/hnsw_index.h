#ifndef HNSW_INDEX_H
#define HNSW_INDEX_H

#include "../../buffer/buffer_pool.h"
#include "../vec_topk.h"
#include "../vec_utils.h"

#include <stdbool.h>

/*
 * HNSW (Hierarchical Navigable Small World) vector index.
 *
 * Multi-layer proximity graph for approximate nearest neighbor search.
 * Each layer is a navigable small world graph; higher layers are sparser.
 * Search starts at the entry point on the topmost layer and greedily
 * descends, then performs a beam search on layer 0.
 *
 * Key parameters:
 *   M              – max connections per node per layer (default 16)
 *   M_max0         – max connections on layer 0 (default 2*M = 32)
 *   ef_construction – beam width during insert (default 200)
 *   ef_search       – beam width during query  (default 64)
 *   ml              – level multiplier: 1/ln(M) per the paper
 *
 * Reference: Malkov & Yashunin, "Efficient and robust approximate
 *            nearest neighbor search using HNSW graphs", 2018.
 */

#define HNSW_DEFAULT_M               16
#define HNSW_DEFAULT_EF_CONSTRUCTION 200
#define HNSW_DEFAULT_EF_SEARCH       64
#define HNSW_MAX_LEVELS              16

/* ---- Node representation ---- */

/*
 * Each inserted vector becomes an HnswNode.
 * Neighbors are stored per-layer in a flat array.
 */
typedef struct {
    page_id_t  vec_pid;          /* page ID where the vector data lives   */
    uint16_t   vec_slot;         /* slot index within vec_pid             */
    uint8_t    level;            /* max level this node was assigned to   */
    uint8_t    _pad;

    /*
     * Neighbor lists: neighbors[l] points to an array of neighbor IDs
     * (indices into the HnswIndex node array), with counts[l] valid entries.
     * Capacity is M_max0 for layer 0, M for layers >= 1.
     */
    uint32_t  *neighbors[HNSW_MAX_LEVELS];
    uint16_t   counts[HNSW_MAX_LEVELS];
} HnswNode;

/* ---- Index structure ---- */

typedef struct {
    BufferPool        *bp;
    uint16_t           dim;
    VecDistanceMetric  metric;

    uint32_t           M;                /* max neighbors per layer >= 1     */
    uint32_t           M_max0;           /* max neighbors on layer 0         */
    uint32_t           ef_construction;  /* beam width during construction   */
    uint32_t           ef_search;        /* beam width during query          */
    float              ml;               /* level multiplier = 1/ln(M)       */

    HnswNode          *nodes;            /* dynamic array of all nodes       */
    uint32_t           num_nodes;
    uint32_t           node_cap;

    uint32_t           entry_point;      /* node ID of the entry point       */
    uint8_t            max_level;        /* current highest occupied level    */

    /*
     * Vector data pages — the vectors themselves are stored on vec pages
     * (same format as FlatIndex) so we can retrieve them for distance calc.
     */
    page_id_t         *vec_page_ids;
    uint32_t           num_vec_pages;
    uint32_t           vec_page_cap;
} HnswIndex;

/*
 * Create an empty HNSW index.
 * M=0 uses default (16), ef_construction=0 uses default (200).
 * Returns 0 on success, -1 on failure.
 */
int hnsw_index_create(HnswIndex *idx, BufferPool *bp,
                      uint16_t dim, VecDistanceMetric metric,
                      uint32_t M, uint32_t ef_construction);

/* Free all in-memory structures. Does NOT deallocate pages on disk. */
void hnsw_index_destroy(HnswIndex *idx);

/*
 * Insert a single vector into the index.
 * On success, *out_pid and *out_slot identify the stored location.
 * Returns 0 on success, -1 on failure.
 */
int hnsw_index_insert(HnswIndex *idx, const float *vec,
                      page_id_t *out_pid, uint16_t *out_slot);

/*
 * Search the index for the top-k approximate nearest neighbors.
 * ef_search controls the beam width (set to 0 to use the default).
 * Returns 0 on success, -1 on failure.
 */
int hnsw_index_search(HnswIndex *idx, const float *query, uint32_t k,
                      uint32_t ef_search,
                      VecResult *results, uint32_t *num_results);

/* Number of vectors in the index. */
uint32_t hnsw_index_count(const HnswIndex *idx);

/*
 * Set the ef_search parameter for subsequent queries.
 */
void hnsw_index_set_ef_search(HnswIndex *idx, uint32_t ef_search);

/*
 * Save the HnswIndex to disk via stream pages.
 * Returns 0 on success and sets *out_pid to the starting stream page ID.
 */
int hnsw_index_save(HnswIndex *idx, page_id_t *out_pid);

/*
 * Load the HnswIndex from a saved stream page.
 * Returns 0 on success, -1 on failure.
 */
int hnsw_index_load(HnswIndex *idx, BufferPool *bp, page_id_t pid);

#endif
