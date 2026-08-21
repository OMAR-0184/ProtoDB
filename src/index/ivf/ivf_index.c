#include "ivf_index.h"
#include "../vec_page.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../../pages/stream_page.h"

/* ---- k-means helpers ---- */

/* Pick k random centroids via partial Fisher-Yates shuffle. */
static void kmeans_init_random(const float *vecs, uint32_t n, float *centroids,
                               uint32_t k, uint16_t dim) {
  uint32_t *indices = malloc(n * sizeof(uint32_t));
  if (!indices) {
    memcpy(centroids, vecs, (size_t)k * dim * sizeof(float));
    return;
  }

  for (uint32_t i = 0; i < n; i++)
    indices[i] = i;

  for (uint32_t i = 0; i < k && i < n; i++) {
    uint32_t j = i + (uint32_t)(rand() % (n - i));
    uint32_t tmp = indices[i];
    indices[i] = indices[j];
    indices[j] = tmp;
  }

  for (uint32_t i = 0; i < k; i++)
    memcpy(centroids + (size_t)i * dim, vecs + (size_t)indices[i] * dim,
           dim * sizeof(float));

  free(indices);
}

static uint32_t find_nearest_centroid(const float *vec, const float *centroids,
                                      uint32_t nlist, uint16_t dim,
                                      vec_dist_fn_t dist_fn) {
  uint32_t best = 0;
  float best_dist = dist_fn(vec, centroids, dim);

  for (uint32_t c = 1; c < nlist; c++) {
    float d = dist_fn(vec, centroids + (size_t)c * dim, dim);
    if (d < best_dist) {
      best_dist = d;
      best = c;
    }
  }
  return best;
}

/* Lloyd's algorithm.  Caller seeds the PRNG. */
static int kmeans_train(const float *train_vecs, uint32_t n, float *centroids,
                        uint32_t k, uint16_t dim, vec_dist_fn_t dist_fn) {
  if (k == 0 || n == 0 || dim == 0)
    return -1;

  if (k > n)
    k = n;

  kmeans_init_random(train_vecs, n, centroids, k, dim);

  float *new_centroids = calloc((size_t)k * dim, sizeof(float));
  uint32_t *counts = calloc(k, sizeof(uint32_t));
  if (!new_centroids || !counts) {
    free(new_centroids);
    free(counts);
    return -1;
  }

  for (int iter = 0; iter < IVF_KMEANS_MAX_ITERS; iter++) {
    memset(new_centroids, 0, (size_t)k * dim * sizeof(float));
    memset(counts, 0, k * sizeof(uint32_t));

    /* Assignment — accumulate per-centroid sums */
    for (uint32_t i = 0; i < n; i++) {
      const float *v = train_vecs + (size_t)i * dim;
      uint32_t c = find_nearest_centroid(v, centroids, k, dim, dist_fn);
      counts[c]++;
      float *acc = new_centroids + (size_t)c * dim;
      for (uint16_t d = 0; d < dim; d++)
        acc[d] += v[d];
    }

    /* Update — recompute centroids as cluster means */
    float max_shift = 0.0f;
    for (uint32_t c = 0; c < k; c++) {
      float *nc = new_centroids + (size_t)c * dim;
      float *oc = centroids + (size_t)c * dim;

      if (counts[c] == 0)
        continue; /* empty cluster — keep old centroid */

      float inv = 1.0f / (float)counts[c];
      for (uint16_t d = 0; d < dim; d++)
        nc[d] *= inv;

      float shift = vec_l2_distance_sq(oc, nc, dim);
      if (shift > max_shift)
        max_shift = shift;

      memcpy(oc, nc, dim * sizeof(float));
    }

    if (max_shift < IVF_KMEANS_TOL)
      break;
  }

  free(new_centroids);
  free(counts);
  return 0;
}

/* ---- public API ---- */

int ivf_index_create(IvfIndex *idx, BufferPool *bp, uint16_t dim,
                     VecDistanceMetric metric, uint32_t nlist,
                     uint32_t nprobe) {
  if (dim == 0 || dim > VECTOR_MAX_DIM)
    return -1;

  memset(idx, 0, sizeof(IvfIndex));
  idx->bp = bp;
  idx->dim = dim;
  idx->metric = metric;
  idx->nlist = (nlist > 0) ? nlist : IVF_DEFAULT_NLIST;
  idx->nprobe = (nprobe > 0) ? nprobe : IVF_DEFAULT_NPROBE;

  if (idx->nprobe > idx->nlist)
    idx->nprobe = idx->nlist;

  idx->centroids = NULL;
  idx->partitions = NULL;
  idx->is_trained = false;
  idx->total_vectors = 0;

  return 0;
}

void ivf_index_destroy(IvfIndex *idx) {
  if (idx->partitions) {
    for (uint32_t i = 0; i < idx->nlist; i++)
      flat_index_destroy(&idx->partitions[i]);
    free(idx->partitions);
    idx->partitions = NULL;
  }
  free(idx->centroids);
  idx->centroids = NULL;
  idx->is_trained = false;
  idx->total_vectors = 0;
}

int ivf_index_train(IvfIndex *idx, const float *train_vecs, uint32_t n) {
  if (idx->is_trained)
    return -1;
  if (n < idx->nlist)
    return -1;

  vec_dist_fn_t dist_fn = vec_get_distance_fn(idx->metric);

  idx->centroids = malloc((size_t)idx->nlist * idx->dim * sizeof(float));
  if (!idx->centroids)
    return -1;

  if (kmeans_train(train_vecs, n, idx->centroids, idx->nlist, idx->dim,
                   dist_fn) < 0) {
    free(idx->centroids);
    idx->centroids = NULL;
    return -1;
  }

  /* One FlatIndex per partition */
  idx->partitions = malloc(idx->nlist * sizeof(FlatIndex));
  if (!idx->partitions) {
    free(idx->centroids);
    idx->centroids = NULL;
    return -1;
  }

  for (uint32_t i = 0; i < idx->nlist; i++) {
    if (flat_index_create(&idx->partitions[i], idx->bp, idx->dim, idx->metric) <
        0) {
      for (uint32_t j = 0; j < i; j++)
        flat_index_destroy(&idx->partitions[j]);
      free(idx->partitions);
      free(idx->centroids);
      idx->partitions = NULL;
      idx->centroids = NULL;
      return -1;
    }
  }

  idx->is_trained = true;
  return 0;
}

int ivf_index_insert(IvfIndex *idx, const float *vec, page_id_t *out_pid,
                     uint16_t *out_slot) {
  if (!idx->is_trained)
    return -1;

  vec_dist_fn_t dist_fn = vec_get_distance_fn(idx->metric);
  uint32_t c =
      find_nearest_centroid(vec, idx->centroids, idx->nlist, idx->dim, dist_fn);

  int rc = flat_index_insert(&idx->partitions[c], vec, out_pid, out_slot);
  if (rc == 0)
    idx->total_vectors++;
  return rc;
}

int ivf_index_delete(IvfIndex *idx, page_id_t pid, uint16_t slot) {
    if (!idx->is_trained)
        return -1;
        
    for (uint32_t i = 0; i < idx->nlist; i++) {
        if (flat_index_delete(&idx->partitions[i], pid, slot) == 0) {
            idx->total_vectors--;
            return 0;
        }
    }
    return -1;
}

int ivf_index_search(IvfIndex *idx, const float *query, uint32_t k,
                     VecResult *results, uint32_t *num_results) {
  if (!idx->is_trained) {
    *num_results = 0;
    return -1;
  }
  if (k == 0) {
    *num_results = 0;
    return 0;
  }

  vec_dist_fn_t dist_fn = vec_get_distance_fn(idx->metric);
  uint32_t nprobe = idx->nprobe;

  /* Find the nprobe nearest centroids */
  VecTopK centroid_tk;
  if (vec_topk_init(&centroid_tk, nprobe) < 0)
    return -1;

  for (uint32_t c = 0; c < idx->nlist; c++) {
    float d = dist_fn(query, idx->centroids + (size_t)c * idx->dim, idx->dim);
    vec_topk_push(&centroid_tk, c, 0, d);
  }

  vec_topk_sort(&centroid_tk);

  /* Search each selected partition, merge into final heap */
  VecTopK final_tk;
  if (vec_topk_init(&final_tk, k) < 0) {
    vec_topk_destroy(&centroid_tk);
    return -1;
  }

  for (uint32_t i = 0; i < centroid_tk.count; i++) {
    uint32_t part_idx = centroid_tk.items[i].page_id;
    FlatIndex *part = &idx->partitions[part_idx];

    if (flat_index_count(part) == 0)
      continue;

    uint32_t part_k = k;
    VecResult *part_results = malloc(part_k * sizeof(VecResult));
    if (!part_results) {
      vec_topk_destroy(&centroid_tk);
      vec_topk_destroy(&final_tk);
      return -1;
    }

    uint32_t part_num = 0;
    int rc = flat_index_search(part, query, part_k, part_results, &part_num);
    if (rc < 0) {
      free(part_results);
      vec_topk_destroy(&centroid_tk);
      vec_topk_destroy(&final_tk);
      return -1;
    }

    for (uint32_t j = 0; j < part_num; j++) {
      vec_topk_push(&final_tk, part_results[j].page_id,
                    part_results[j].slot_index, part_results[j].distance);
    }

    free(part_results);
  }

  /* Sort results and copy out */
  vec_topk_sort(&final_tk);

  uint32_t n = final_tk.count;
  memcpy(results, final_tk.items, n * sizeof(VecResult));
  *num_results = n;

  vec_topk_destroy(&centroid_tk);
  vec_topk_destroy(&final_tk);
  return 0;
}

uint32_t ivf_index_count(const IvfIndex *idx) { return idx->total_vectors; }

bool ivf_index_is_trained(const IvfIndex *idx) { return idx->is_trained; }

typedef struct {
    uint16_t dim;
    uint32_t metric;
    uint32_t nlist;
    uint32_t nprobe;
    uint32_t total_vectors;
    uint8_t  is_trained;
} IvfIndexMeta;

int ivf_index_save(IvfIndex *idx, page_id_t *out_pid) {
    if (!idx->is_trained) return -1;

    size_t meta_size = sizeof(IvfIndexMeta);
    size_t centroids_size = idx->nlist * idx->dim * sizeof(float);
    size_t part_pids_size = idx->nlist * sizeof(page_id_t);
    size_t total_size = meta_size + centroids_size + part_pids_size;

    uint8_t *buf = malloc(total_size);
    if (!buf) return -1;

    /* 1. Save partitions first to get their PIDs */
    page_id_t *part_pids = (page_id_t *)(buf + meta_size + centroids_size);
    for (uint32_t i = 0; i < idx->nlist; i++) {
        if (flat_index_save(&idx->partitions[i], &part_pids[i]) < 0) {
            free(buf);
            return -1;
        }
    }

    /* 2. Populate meta */
    IvfIndexMeta *meta = (IvfIndexMeta *)buf;
    meta->dim = idx->dim;
    meta->metric = (uint32_t)idx->metric;
    meta->nlist = idx->nlist;
    meta->nprobe = idx->nprobe;
    meta->total_vectors = idx->total_vectors;
    meta->is_trained = 1;

    /* 3. Copy centroids */
    memcpy(buf + meta_size, idx->centroids, centroids_size);

    /* 4. Write stream */
    int rc = stream_write(idx->bp, buf, total_size, out_pid);
    free(buf);
    return rc;
}

int ivf_index_load(IvfIndex *idx, BufferPool *bp, page_id_t pid) {
    IvfIndexMeta meta;
    if (stream_read(bp, pid, &meta, sizeof(IvfIndexMeta)) < 0) {
        return -1;
    }

    size_t centroids_size = meta.nlist * meta.dim * sizeof(float);
    size_t part_pids_size = meta.nlist * sizeof(page_id_t);
    size_t total_size = sizeof(IvfIndexMeta) + centroids_size + part_pids_size;

    uint8_t *buf = malloc(total_size);
    if (!buf) return -1;

    if (stream_read(bp, pid, buf, total_size) < 0) {
        free(buf);
        return -1;
    }

    memset(idx, 0, sizeof(IvfIndex));
    idx->bp = bp;
    idx->dim = meta.dim;
    idx->metric = (VecDistanceMetric)meta.metric;
    idx->nlist = meta.nlist;
    idx->nprobe = meta.nprobe;
    idx->total_vectors = meta.total_vectors;
    idx->is_trained = meta.is_trained != 0;

    idx->centroids = malloc(centroids_size);
    if (!idx->centroids) {
        free(buf);
        return -1;
    }
    memcpy(idx->centroids, buf + sizeof(IvfIndexMeta), centroids_size);

    idx->partitions = malloc(meta.nlist * sizeof(FlatIndex));
    if (!idx->partitions) {
        free(idx->centroids);
        free(buf);
        return -1;
    }

    page_id_t *part_pids = (page_id_t *)(buf + sizeof(IvfIndexMeta) + centroids_size);
    for (uint32_t i = 0; i < meta.nlist; i++) {
        if (flat_index_load(&idx->partitions[i], bp, part_pids[i]) < 0) {
            for (uint32_t j = 0; j < i; j++) {
                flat_index_destroy(&idx->partitions[j]);
            }
            free(idx->partitions);
            free(idx->centroids);
            free(buf);
            return -1;
        }
    }

    free(buf);
    return 0;
}
