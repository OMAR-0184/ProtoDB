#ifndef IVF_INDEX_H
#define IVF_INDEX_H

#include "../../buffer/buffer_pool.h"
#include "../flat/flat_index.h"
#include "../vec_topk.h"
#include "../vec_utils.h"

#include <stdbool.h>

#define IVF_DEFAULT_NLIST    128
#define IVF_DEFAULT_NPROBE     8
#define IVF_KMEANS_MAX_ITERS  25
#define IVF_KMEANS_TOL        1e-6f

/*
 * IVF (Inverted File) vector index.
 *
 * Partitions vectors into nlist Voronoi cells via k-means.
 * Each cell is a FlatIndex.  Search probes only the nprobe
 * nearest partitions — sub-linear scan cost.
 */
typedef struct {
    BufferPool        *bp;
    uint16_t           dim;
    VecDistanceMetric  metric;

    uint32_t           nlist;        /* number of partitions             */
    uint32_t           nprobe;       /* partitions searched per query    */

    float             *centroids;    /* nlist × dim, row-major           */
    FlatIndex         *partitions;   /* one FlatIndex per centroid       */

    bool               is_trained;
    uint32_t           total_vectors;
} IvfIndex;

/* Allocate an untrained index.  nlist/nprobe of 0 use defaults.
 * Returns 0 on success, -1 on failure. */
int ivf_index_create(IvfIndex *idx, BufferPool *bp,
                     uint16_t dim, VecDistanceMetric metric,
                     uint32_t nlist, uint32_t nprobe);

/* Free centroids, partitions, and all associated memory. */
void ivf_index_destroy(IvfIndex *idx);

/* Learn nlist centroids via k-means on train_vecs (n >= nlist).
 * Returns 0 on success, -1 on failure. */
int ivf_index_train(IvfIndex *idx, const float *train_vecs, uint32_t n);

/* Insert a vector into the nearest centroid's partition.
 * Returns 0 on success, -1 on failure or if untrained. */
int ivf_index_insert(IvfIndex *idx, const float *vec,
                     page_id_t *out_pid, uint16_t *out_slot);

/*
 * Delete a vector by marking its tombstone bit.
 * Returns 0 on success, -1 if not found or already deleted.
 */
int ivf_index_delete(IvfIndex *idx, page_id_t pid, uint16_t slot);

/* Approximate kNN — probes nprobe partitions, merges results.
 * Results sorted ascending by distance.
 * Returns 0 on success, -1 on failure or if untrained. */
int ivf_index_search(IvfIndex *idx, const float *query, uint32_t k,
                     vec_filter_fn_t filter_fn, void *filter_arg,
                     VecResult *results, uint32_t *num_results);

uint32_t ivf_index_count(const IvfIndex *idx);
bool ivf_index_is_trained(const IvfIndex *idx);

/*
 * Save the IvfIndex metadata, centroids, and partitions to disk.
 * Returns 0 on success and sets *out_pid to the starting stream page ID.
 */
int ivf_index_save(IvfIndex *idx, page_id_t *out_pid);

/*
 * Load the IvfIndex from a saved stream page.
 * Returns 0 on success, -1 on failure.
 */
int ivf_index_load(IvfIndex *idx, BufferPool *bp, page_id_t pid);

#endif
