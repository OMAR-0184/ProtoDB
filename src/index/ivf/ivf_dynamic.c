#include "ivf_dynamic.h"
#include "../vec_page.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ============================================================
 *  Default configuration
 * ============================================================ */

void ivf_dynamic_config_default(IvfDynamicConfig *cfg) {
    cfg->split_threshold  = IVF_DYN_DEFAULT_SPLIT_THRESHOLD;
    cfg->merge_threshold  = IVF_DYN_DEFAULT_MERGE_THRESHOLD;
    cfg->min_cluster_size = IVF_DYN_DEFAULT_MIN_CLUSTER_SIZE;
    cfg->centroid_alpha   = IVF_DYN_DEFAULT_CENTROID_ALPHA;
    cfg->update_interval  = IVF_DYN_DEFAULT_UPDATE_INTERVAL;
}

/* ============================================================
 *  Health check
 * ============================================================ */

int ivf_dynamic_health_check(IvfIndex *idx, const IvfDynamicConfig *cfg,
                              ClusterHealthReport *report) {
    if (!idx->is_trained || !report)
        return -1;

    report->num_clusters = idx->nlist;
    report->total_vectors = idx->total_vectors;
    report->stats = malloc(idx->nlist * sizeof(ClusterStats));
    if (!report->stats)
        return -1;

    /* Collect per-cluster counts */
    uint64_t sum = 0;
    for (uint32_t i = 0; i < idx->nlist; i++) {
        report->stats[i].cluster_id = i;
        report->stats[i].count = flat_index_count(&idx->partitions[i]);
        report->stats[i].centroid_drift = 0.0f; /* computed below if needed */
        sum += report->stats[i].count;
    }

    report->mean_count = (float)sum / (float)idx->nlist;

    /* Compute std deviation */
    float var_sum = 0.0f;
    for (uint32_t i = 0; i < idx->nlist; i++) {
        float diff = (float)report->stats[i].count - report->mean_count;
        var_sum += diff * diff;
    }
    report->std_count = sqrtf(var_sum / (float)idx->nlist);

    /* Classify clusters */
    report->num_oversized = 0;
    report->num_undersized = 0;

    float split_limit = report->mean_count * cfg->split_threshold;
    float merge_limit = report->mean_count * cfg->merge_threshold;

    for (uint32_t i = 0; i < idx->nlist; i++) {
        if ((float)report->stats[i].count > split_limit)
            report->num_oversized++;
        if ((float)report->stats[i].count < merge_limit &&
            report->stats[i].count < cfg->min_cluster_size)
            report->num_undersized++;
    }

    return 0;
}

void ivf_dynamic_health_free(ClusterHealthReport *report) {
    free(report->stats);
    report->stats = NULL;
}

/* ============================================================
 *  Incremental centroid update (EMA)
 * ============================================================ */

int ivf_dynamic_update_centroid(IvfIndex *idx, uint32_t cluster_id,
                                 const float *vec, float alpha) {
    if (!idx->is_trained || cluster_id >= idx->nlist || !vec)
        return -1;
    if (alpha < 0.0f || alpha > 1.0f)
        return -1;

    float *centroid = idx->centroids + (size_t)cluster_id * idx->dim;

    /* Exponential moving average: c = α·v + (1-α)·c */
    float one_minus_alpha = 1.0f - alpha;
    for (uint16_t d = 0; d < idx->dim; d++) {
        centroid[d] = alpha * vec[d] + one_minus_alpha * centroid[d];
    }

    return 0;
}

/* ============================================================
 *  Internal helpers for extracting vectors from a FlatIndex
 * ============================================================ */

/*
 * Extract all live vectors from a FlatIndex partition into a contiguous
 * float array. Caller must free the returned array.
 * Sets *out_count to the number of vectors extracted.
 * Returns NULL on failure.
 */
static float *extract_partition_vectors(IvfIndex *idx, uint32_t cluster_id,
                                         uint32_t *out_count) {
    FlatIndex *part = &idx->partitions[cluster_id];
    uint32_t count = flat_index_count(part);
    if (count == 0) {
        *out_count = 0;
        return NULL;
    }

    float *vecs = malloc((size_t)count * idx->dim * sizeof(float));
    if (!vecs) {
        *out_count = 0;
        return NULL;
    }

    uint32_t filled = 0;
    for (uint32_t pi = 0; pi < part->num_pages && filled < count; pi++) {
        page_id_t pid = part->page_ids[pi];
        Page *p = bp_fetch_page(idx->bp, pid);
        if (!p) {
            free(vecs);
            *out_count = 0;
            return NULL;
        }

        uint16_t nv = vec_page_count(p);
        for (uint16_t vi = 0; vi < nv && filled < count; vi++) {
            if (vec_page_is_deleted(p, vi))
                continue;
            const float *v = vec_page_get(p, vi);
            if (v) {
                memcpy(vecs + (size_t)filled * idx->dim, v,
                       idx->dim * sizeof(float));
                filled++;
            }
        }

        bp_unpin_page(idx->bp, pid, false);
    }

    *out_count = filled;
    return vecs;
}

/* ============================================================
 *  Cluster splitting (2-means)
 * ============================================================ */

int ivf_dynamic_split_cluster(IvfIndex *idx, uint32_t cluster_id) {
    if (!idx->is_trained || cluster_id >= idx->nlist)
        return -1;

    /* Extract all vectors from this cluster */
    uint32_t count;
    float *vecs = extract_partition_vectors(idx, cluster_id, &count);
    if (count < 2) {
        free(vecs);
        return -1; /* can't split a cluster with fewer than 2 vectors */
    }

    vec_dist_fn_t dist_fn = vec_get_distance_fn(idx->metric);
    uint16_t dim = idx->dim;

    /* Run 2-means on the extracted vectors */
    float *c1 = malloc(dim * sizeof(float));
    float *c2 = malloc(dim * sizeof(float));
    if (!c1 || !c2) {
        free(c1); free(c2); free(vecs);
        return -1;
    }

    /* Initialize from first and last vector */
    memcpy(c1, vecs, dim * sizeof(float));
    memcpy(c2, vecs + (size_t)(count - 1) * dim, dim * sizeof(float));

    uint32_t *assignments = malloc(count * sizeof(uint32_t));
    if (!assignments) {
        free(c1); free(c2); free(vecs);
        return -1;
    }

    /* 2-means iterations */
    for (int iter = 0; iter < IVF_KMEANS_MAX_ITERS; iter++) {
        /* Assignment */
        for (uint32_t i = 0; i < count; i++) {
            const float *v = vecs + (size_t)i * dim;
            float d1 = dist_fn(v, c1, dim);
            float d2 = dist_fn(v, c2, dim);
            assignments[i] = (d1 <= d2) ? 0 : 1;
        }

        /* Update centroids */
        float *new_c1 = calloc(dim, sizeof(float));
        float *new_c2 = calloc(dim, sizeof(float));
        uint32_t cnt1 = 0, cnt2 = 0;

        if (!new_c1 || !new_c2) {
            free(new_c1); free(new_c2);
            free(assignments); free(c1); free(c2); free(vecs);
            return -1;
        }

        for (uint32_t i = 0; i < count; i++) {
            const float *v = vecs + (size_t)i * dim;
            if (assignments[i] == 0) {
                for (uint16_t d = 0; d < dim; d++) new_c1[d] += v[d];
                cnt1++;
            } else {
                for (uint16_t d = 0; d < dim; d++) new_c2[d] += v[d];
                cnt2++;
            }
        }

        float max_shift = 0.0f;
        if (cnt1 > 0) {
            for (uint16_t d = 0; d < dim; d++) new_c1[d] /= (float)cnt1;
            float s = dist_fn(c1, new_c1, dim);
            if (s > max_shift) max_shift = s;
            memcpy(c1, new_c1, dim * sizeof(float));
        }
        if (cnt2 > 0) {
            for (uint16_t d = 0; d < dim; d++) new_c2[d] /= (float)cnt2;
            float s = dist_fn(c2, new_c2, dim);
            if (s > max_shift) max_shift = s;
            memcpy(c2, new_c2, dim * sizeof(float));
        }

        free(new_c1);
        free(new_c2);

        if (max_shift < IVF_KMEANS_TOL) break;
    }

    /* Grow centroids and partitions arrays by 1 */
    uint32_t new_nlist = idx->nlist + 1;
    float *new_centroids = realloc(idx->centroids,
                                    (size_t)new_nlist * dim * sizeof(float));
    if (!new_centroids) {
        free(assignments); free(c1); free(c2); free(vecs);
        return -1;
    }
    idx->centroids = new_centroids;

    FlatIndex *new_parts = realloc(idx->partitions,
                                    new_nlist * sizeof(FlatIndex));
    if (!new_parts) {
        free(assignments); free(c1); free(c2); free(vecs);
        return -1;
    }
    idx->partitions = new_parts;

    /* Create new partition */
    uint32_t new_idx = idx->nlist;
    if (flat_index_create(&idx->partitions[new_idx], idx->bp,
                          dim, idx->metric) < 0) {
        free(assignments); free(c1); free(c2); free(vecs);
        return -1;
    }

    /* Update centroid for original cluster */
    memcpy(idx->centroids + (size_t)cluster_id * dim, c1, dim * sizeof(float));
    /* Set centroid for new cluster */
    memcpy(idx->centroids + (size_t)new_idx * dim, c2, dim * sizeof(float));

    /* Clear original partition and re-insert assigned vectors */
    /* First, destroy old partition and recreate */
    flat_index_destroy(&idx->partitions[cluster_id]);
    if (flat_index_create(&idx->partitions[cluster_id], idx->bp,
                          dim, idx->metric) < 0) {
        free(assignments); free(c1); free(c2); free(vecs);
        return -1;
    }

    /* Re-insert vectors into their assigned clusters */
    for (uint32_t i = 0; i < count; i++) {
        const float *v = vecs + (size_t)i * dim;
        uint32_t target = (assignments[i] == 0) ? cluster_id : new_idx;

        page_id_t pid;
        uint16_t slot;
        if (flat_index_insert(&idx->partitions[target], v, &pid, &slot) < 0) {
            /* Non-fatal: vector lost during split */
            idx->total_vectors--;
            continue;
        }
    }

    idx->nlist = new_nlist;

    free(assignments);
    free(c1);
    free(c2);
    free(vecs);
    return 0;
}

/* ============================================================
 *  Cluster merging
 * ============================================================ */

int ivf_dynamic_merge_cluster(IvfIndex *idx, uint32_t cluster_id) {
    if (!idx->is_trained || cluster_id >= idx->nlist)
        return -1;

    vec_dist_fn_t dist_fn = vec_get_distance_fn(idx->metric);
    uint16_t dim = idx->dim;

    /* Find nearest neighboring cluster */
    uint32_t nearest = UINT32_MAX;
    float nearest_dist = FLT_MAX;
    const float *src_centroid = idx->centroids + (size_t)cluster_id * dim;

    for (uint32_t c = 0; c < idx->nlist; c++) {
        if (c == cluster_id) continue;
        if (flat_index_count(&idx->partitions[c]) == 0) continue;

        float d = dist_fn(src_centroid,
                          idx->centroids + (size_t)c * dim, dim);
        if (d < nearest_dist) {
            nearest_dist = d;
            nearest = c;
        }
    }

    if (nearest == UINT32_MAX)
        return -1; /* no valid merge target */

    /* Extract all vectors from source cluster */
    uint32_t count;
    float *vecs = extract_partition_vectors(idx, cluster_id, &count);

    /* Insert all vectors into the target cluster */
    if (vecs && count > 0) {
        for (uint32_t i = 0; i < count; i++) {
            page_id_t pid;
            uint16_t slot;
            if (flat_index_insert(&idx->partitions[nearest],
                                  vecs + (size_t)i * dim,
                                  &pid, &slot) < 0) {
                /* Non-fatal: vector lost during merge */
                idx->total_vectors--;
            }
        }
        free(vecs);
    }

    /* Clear the source partition */
    flat_index_destroy(&idx->partitions[cluster_id]);
    flat_index_create(&idx->partitions[cluster_id], idx->bp, dim, idx->metric);

    /* Update the target centroid as weighted average */
    uint32_t target_count = flat_index_count(&idx->partitions[nearest]);
    if (target_count > 0) {
        float *target_centroid = idx->centroids + (size_t)nearest * dim;
        /* Simple weighted average: (old_count * old + new_count * src) / total */
        float old_weight = (float)(target_count - count);
        float new_weight = (float)count;
        float total_weight = old_weight + new_weight;
        if (total_weight > 0.0f) {
            for (uint16_t d = 0; d < dim; d++) {
                target_centroid[d] = (old_weight * target_centroid[d] +
                                      new_weight * src_centroid[d]) /
                                     total_weight;
            }
        }
    }

    return 0;
}

/* ============================================================
 *  Compact — remove empty clusters
 * ============================================================ */

int ivf_dynamic_compact(IvfIndex *idx) {
    if (!idx->is_trained)
        return -1;

    uint16_t dim = idx->dim;

    /* Count non-empty clusters */
    uint32_t live = 0;
    for (uint32_t i = 0; i < idx->nlist; i++) {
        if (flat_index_count(&idx->partitions[i]) > 0)
            live++;
    }

    if (live == idx->nlist)
        return 0; /* nothing to compact */

    if (live == 0)
        return -1; /* can't compact to zero clusters */

    /* Build compacted arrays */
    float *new_centroids = malloc((size_t)live * dim * sizeof(float));
    FlatIndex *new_parts = malloc(live * sizeof(FlatIndex));
    if (!new_centroids || !new_parts) {
        free(new_centroids);
        free(new_parts);
        return -1;
    }

    uint32_t write = 0;
    for (uint32_t i = 0; i < idx->nlist; i++) {
        if (flat_index_count(&idx->partitions[i]) > 0) {
            memcpy(new_centroids + (size_t)write * dim,
                   idx->centroids + (size_t)i * dim,
                   dim * sizeof(float));
            new_parts[write] = idx->partitions[i];
            write++;
        } else {
            flat_index_destroy(&idx->partitions[i]);
        }
    }

    free(idx->centroids);
    free(idx->partitions);
    idx->centroids = new_centroids;
    idx->partitions = new_parts;
    idx->nlist = live;

    /* Clamp nprobe if it now exceeds nlist */
    if (idx->nprobe > idx->nlist)
        idx->nprobe = idx->nlist;

    return 0;
}

/* ============================================================
 *  Rebalance pass
 * ============================================================ */

int ivf_dynamic_rebalance(IvfIndex *idx, const IvfDynamicConfig *cfg) {
    if (!idx->is_trained)
        return -1;

    ClusterHealthReport report;
    if (ivf_dynamic_health_check(idx, cfg, &report) < 0)
        return -1;

    int ops = 0;

    /* Phase 1: Split oversized clusters */
    float split_limit = report.mean_count * cfg->split_threshold;
    for (uint32_t i = 0; i < report.num_clusters; i++) {
        if ((float)report.stats[i].count > split_limit) {
            if (ivf_dynamic_split_cluster(idx, report.stats[i].cluster_id) == 0)
                ops++;
        }
    }

    /* Phase 2: Merge undersized clusters (re-check after splits) */
    float merge_limit = report.mean_count * cfg->merge_threshold;
    for (uint32_t i = 0; i < idx->nlist; i++) {
        uint32_t count = flat_index_count(&idx->partitions[i]);
        if ((float)count < merge_limit && count < cfg->min_cluster_size) {
            if (ivf_dynamic_merge_cluster(idx, i) == 0)
                ops++;
        }
    }

    /* Phase 3: Compact empty clusters */
    if (ops > 0) {
        ivf_dynamic_compact(idx);
    }

    ivf_dynamic_health_free(&report);
    return ops;
}

/* ============================================================
 *  Full re-train (nuclear option)
 * ============================================================ */

int ivf_dynamic_retrain(IvfIndex *idx) {
    if (!idx->is_trained)
        return -1;

    uint16_t dim = idx->dim;
    uint32_t nlist = idx->nlist;

    /* Extract ALL vectors from all partitions */
    uint32_t total = idx->total_vectors;
    if (total == 0) return 0;

    float *all_vecs = malloc((size_t)total * dim * sizeof(float));
    if (!all_vecs) return -1;

    uint32_t filled = 0;
    for (uint32_t c = 0; c < idx->nlist; c++) {
        uint32_t count;
        float *cvecs = extract_partition_vectors(idx, c, &count);
        if (cvecs && count > 0) {
            if (filled + count > total) count = total - filled;
            memcpy(all_vecs + (size_t)filled * dim, cvecs,
                   (size_t)count * dim * sizeof(float));
            filled += count;
            free(cvecs);
        }
    }

    if (filled == 0) {
        free(all_vecs);
        return -1;
    }

    /* Destroy all existing partitions */
    for (uint32_t i = 0; i < idx->nlist; i++) {
        flat_index_destroy(&idx->partitions[i]);
    }
    free(idx->partitions);
    free(idx->centroids);
    idx->partitions = NULL;
    idx->centroids = NULL;
    idx->is_trained = false;
    idx->total_vectors = 0;
    idx->nlist = nlist;

    /* Re-train from scratch */
    if (ivf_index_train(idx, all_vecs, filled) < 0) {
        free(all_vecs);
        return -1;
    }

    /* Re-insert all vectors */
    if (ivf_index_insert_batch(idx, all_vecs, filled) < 0) {
        free(all_vecs);
        return -1;
    }

    free(all_vecs);
    return 0;
}
