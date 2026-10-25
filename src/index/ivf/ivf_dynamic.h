#ifndef IVF_DYNAMIC_H
#define IVF_DYNAMIC_H

#include "ivf_index.h"

/*
 * Online IVF Dynamic Re-Clustering
 *
 * Extends the IVF index with runtime cluster health monitoring and
 * dynamic centroid adaptation. Handles distribution drift by detecting
 * imbalanced or oversized clusters and splitting/merging them.
 *
 * Key operations:
 *   - Cluster health check (imbalance detection)
 *   - Cluster splitting (oversized → two smaller clusters)
 *   - Cluster merging (undersized → absorbed into nearest neighbor)
 *   - Incremental centroid update (online moving average)
 *   - Full background re-index from scratch
 */

/* ---- Configuration ---- */

typedef struct {
    float    split_threshold;    /* split if count > mean * split_threshold     */
    float    merge_threshold;    /* merge if count < mean * merge_threshold     */
    uint32_t min_cluster_size;   /* never merge clusters with >= this count     */
    float    centroid_alpha;     /* EMA smoothing: new = α·sample + (1-α)·old  */
    uint32_t update_interval;    /* re-check after this many inserts            */
} IvfDynamicConfig;

/* Default configuration */
#define IVF_DYN_DEFAULT_SPLIT_THRESHOLD  4.0f
#define IVF_DYN_DEFAULT_MERGE_THRESHOLD  0.1f
#define IVF_DYN_DEFAULT_MIN_CLUSTER_SIZE 16
#define IVF_DYN_DEFAULT_CENTROID_ALPHA   0.01f
#define IVF_DYN_DEFAULT_UPDATE_INTERVAL  10000

/* ---- Cluster health statistics ---- */

typedef struct {
    uint32_t  cluster_id;
    uint32_t  count;            /* number of vectors in this cluster           */
    float     centroid_drift;   /* L2 distance from centroid to cluster mean   */
} ClusterStats;

typedef struct {
    ClusterStats *stats;
    uint32_t      num_clusters;
    uint32_t      total_vectors;
    float         mean_count;
    float         std_count;
    uint32_t      num_oversized;   /* clusters above split threshold          */
    uint32_t      num_undersized;  /* clusters below merge threshold          */
} ClusterHealthReport;

/*
 * Initialize default dynamic config.
 */
void ivf_dynamic_config_default(IvfDynamicConfig *cfg);

/*
 * Compute cluster health statistics for the IVF index.
 * Caller must free report->stats after use.
 * Returns 0 on success, -1 on failure.
 */
int ivf_dynamic_health_check(IvfIndex *idx, const IvfDynamicConfig *cfg,
                              ClusterHealthReport *report);

/* Free the stats array inside a health report. */
void ivf_dynamic_health_free(ClusterHealthReport *report);

/*
 * Update a single centroid incrementally using exponential moving average.
 * Called after each insert to keep centroids tracking data drift.
 *   new_centroid = alpha * vec + (1 - alpha) * old_centroid
 *
 * cluster_id: the cluster the vector was assigned to.
 * vec: the newly inserted vector.
 * Returns 0 on success, -1 on failure.
 */
int ivf_dynamic_update_centroid(IvfIndex *idx, uint32_t cluster_id,
                                 const float *vec, float alpha);

/*
 * Split an oversized cluster into two.
 * Uses 2-means on the cluster's vectors to find two new centroids.
 * The original cluster keeps half the vectors; a new cluster is appended.
 *
 * Warning: This increases nlist by 1. The caller should update nprobe
 * proportionally if desired.
 *
 * Returns 0 on success, -1 on failure.
 */
int ivf_dynamic_split_cluster(IvfIndex *idx, uint32_t cluster_id);

/*
 * Merge an undersized cluster into its nearest neighbor cluster.
 * All vectors are moved to the target cluster; the source is emptied.
 *
 * Note: This does NOT reduce nlist — the empty cluster remains as a
 * placeholder with 0 vectors. Call ivf_dynamic_compact() to remove empties.
 *
 * Returns 0 on success, -1 on failure.
 */
int ivf_dynamic_merge_cluster(IvfIndex *idx, uint32_t cluster_id);

/*
 * Compact the IVF index by removing empty clusters (from merges).
 * Reduces nlist and reindexes centroids/partitions arrays.
 * Returns 0 on success, -1 on failure.
 */
int ivf_dynamic_compact(IvfIndex *idx);

/*
 * Run a full rebalance pass: check health, split oversized, merge undersized.
 * Returns the number of operations performed, or -1 on error.
 */
int ivf_dynamic_rebalance(IvfIndex *idx, const IvfDynamicConfig *cfg);

/*
 * Full background re-train: extract all vectors, re-run k-means, rebuild.
 * This is the nuclear option for heavily drifted distributions.
 * Uses the same nlist as the current index.
 * Returns 0 on success, -1 on failure.
 */
int ivf_dynamic_retrain(IvfIndex *idx);

#endif
