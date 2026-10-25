#include "hnsw_index.h"
#include "../vec_page.h"
#include "../../pages/stream_page.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define INITIAL_NODE_CAP   1024
#define INITIAL_PAGE_CAP   16

/* ============================================================
 *  Internal helpers
 * ============================================================ */

/*
 * Generate a random level for a new node using the exponential distribution.
 * P(level = l) = (1/ml)^l * (1 - 1/ml), capped at HNSW_MAX_LEVELS-1.
 */
static uint8_t random_level(float ml) {
    float r = ((float)rand() + 1.0f) / ((float)RAND_MAX + 2.0f);
    int lvl = (int)(-logf(r) * ml);
    if (lvl < 0) lvl = 0;
    if (lvl >= HNSW_MAX_LEVELS) lvl = HNSW_MAX_LEVELS - 1;
    return (uint8_t)lvl;
}

/*
 * Retrieve the vector data for a given node by fetching its page.
 * The caller must call bp_unpin_page when done with the returned pointer.
 * Sets *out_page to the fetched Page (so caller can unpin).
 * Returns pointer to the float vector, or NULL on failure.
 */
static const float *get_node_vector(HnswIndex *idx, uint32_t node_id,
                                     Page **out_page) {
    HnswNode *node = &idx->nodes[node_id];
    Page *p = bp_fetch_page(idx->bp, node->vec_pid);
    if (!p) return NULL;
    *out_page = p;
    return vec_page_get(p, node->vec_slot);
}


/*
 * Compute distance between query and a node's vector.
 * Handles page pinning internally.
 * Returns FLT_MAX on failure.
 */
static float node_distance(HnswIndex *idx, const float *query,
                            uint32_t node_id, vec_dist_fn_t dist_fn) {
    Page *p;
    const float *vec = get_node_vector(idx, node_id, &p);
    if (!vec) return FLT_MAX;

    float d = dist_fn(query, vec, idx->dim);
    bp_unpin_page(idx->bp, idx->nodes[node_id].vec_pid, false);
    return d;
}

/* ---- Candidate list (min-heap by distance for beam search) ---- */

typedef struct {
    uint32_t node_id;
    float    distance;
} HnswCandidate;

typedef struct {
    HnswCandidate *items;
    uint32_t        count;
    uint32_t        capacity;
} HnswCandidateList;

static int cand_list_init(HnswCandidateList *cl, uint32_t cap) {
    cl->items = malloc(cap * sizeof(HnswCandidate));
    if (!cl->items) return -1;
    cl->count = 0;
    cl->capacity = cap;
    return 0;
}

static void cand_list_destroy(HnswCandidateList *cl) {
    free(cl->items);
    cl->items = NULL;
    cl->count = 0;
    cl->capacity = 0;
}

static int cand_list_push(HnswCandidateList *cl, uint32_t node_id, float dist) {
    if (cl->count >= cl->capacity) {
        uint32_t new_cap = cl->capacity * 2;
        if (new_cap < 64) new_cap = 64;
        HnswCandidate *new_items = realloc(cl->items,
                                            new_cap * sizeof(HnswCandidate));
        if (!new_items) return -1;
        cl->items = new_items;
        cl->capacity = new_cap;
    }
    cl->items[cl->count].node_id = node_id;
    cl->items[cl->count].distance = dist;
    cl->count++;
    return 0;
}

static int cmp_candidates_asc(const void *a, const void *b) {
    float da = ((const HnswCandidate *)a)->distance;
    float db = ((const HnswCandidate *)b)->distance;
    if (da < db) return -1;
    if (da > db) return  1;
    return 0;
}

static void cand_list_sort(HnswCandidateList *cl) {
    if (cl->count <= 1) return;
    qsort(cl->items, cl->count, sizeof(HnswCandidate), cmp_candidates_asc);
}

static bool cand_list_pop_min(HnswCandidateList *cl, HnswCandidate *out) {
    if (cl->count == 0) return false;
    uint32_t min_idx = 0;
    float min_dist = cl->items[0].distance;
    for (uint32_t i = 1; i < cl->count; i++) {
        if (cl->items[i].distance < min_dist) {
            min_dist = cl->items[i].distance;
            min_idx = i;
        }
    }
    *out = cl->items[min_idx];
    cl->items[min_idx] = cl->items[cl->count - 1];
    cl->count--;
    return true;
}

/* ---- Visited set (bitmap for large graphs) ---- */

typedef struct {
    uint8_t *bits;
    uint32_t num_nodes;
} VisitedSet;

static int visited_init(VisitedSet *vs, uint32_t num_nodes) {
    uint32_t bytes = (num_nodes + 7) / 8;
    if (bytes == 0) bytes = 1;
    vs->bits = calloc(bytes, 1);
    if (!vs->bits) return -1;
    vs->num_nodes = num_nodes;
    return 0;
}

static void visited_destroy(VisitedSet *vs) {
    free(vs->bits);
    vs->bits = NULL;
}

static bool visited_test(const VisitedSet *vs, uint32_t id) {
    if (id >= vs->num_nodes) return false;
    return (vs->bits[id / 8] & (1u << (id % 8))) != 0;
}

static void visited_set(VisitedSet *vs, uint32_t id) {
    if (id >= vs->num_nodes) return;
    vs->bits[id / 8] |= (1u << (id % 8));
}

/* ============================================================
 *  Core HNSW algorithms
 * ============================================================ */

/*
 * Greedy search on a single layer: starting from entry_id,
 * greedily move to the nearest neighbor until no improvement.
 * Returns the closest node found.
 */
static uint32_t search_layer_greedy(HnswIndex *idx, const float *query,
                                     uint32_t entry_id, uint8_t layer,
                                     vec_dist_fn_t dist_fn) {
    uint32_t current = entry_id;
    float current_dist = node_distance(idx, query, current, dist_fn);

    bool improved = true;
    while (improved) {
        improved = false;
        HnswNode *node = &idx->nodes[current];
        for (uint16_t i = 0; i < node->counts[layer]; i++) {
            uint32_t neighbor = node->neighbors[layer][i];
            float d = node_distance(idx, query, neighbor, dist_fn);
            if (d < current_dist) {
                current = neighbor;
                current_dist = d;
                improved = true;
            }
        }
    }
    return current;
}

/*
 * Beam search on a single layer: starting from entry_id,
 * explore neighbors using a dynamic candidate list of size ef.
 * Returns the ef closest nodes found (sorted ascending by distance).
 *
 * This is Algorithm 2 from the HNSW paper (SearchLayerBeam).
 */
static int search_layer_beam(HnswIndex *idx, const float *query,
                              uint32_t entry_id, uint8_t layer,
                              uint32_t ef, vec_dist_fn_t dist_fn,
                              HnswCandidateList *result) {
    VisitedSet visited;
    if (visited_init(&visited, idx->num_nodes) < 0)
        return -1;

    HnswCandidateList candidates;
    if (cand_list_init(&candidates, ef * 4) < 0) {
        visited_destroy(&visited);
        return -1;
    }

    float entry_dist = node_distance(idx, query, entry_id, dist_fn);

    cand_list_push(&candidates, entry_id, entry_dist);
    cand_list_push(result, entry_id, entry_dist);
    visited_set(&visited, entry_id);

    HnswCandidate c;
    while (cand_list_pop_min(&candidates, &c)) {
        cand_list_sort(result);
        float worst_result = (result->count > 0)
            ? result->items[result->count - 1].distance
            : FLT_MAX;

        if (result->count >= ef && c.distance > worst_result)
            break;

        /* Explore neighbors of c on this layer */
        HnswNode *cnode = &idx->nodes[c.node_id];
        for (uint16_t i = 0; i < cnode->counts[layer]; i++) {
            uint32_t neighbor_id = cnode->neighbors[layer][i];

            if (visited_test(&visited, neighbor_id))
                continue;
            visited_set(&visited, neighbor_id);

            float d = node_distance(idx, query, neighbor_id, dist_fn);

            cand_list_sort(result);
            worst_result = (result->count > 0)
                ? result->items[result->count - 1].distance
                : FLT_MAX;

            if (result->count < ef || d < worst_result) {
                cand_list_push(&candidates, neighbor_id, d);
                cand_list_push(result, neighbor_id, d);

                /* Trim result to ef entries */
                if (result->count > ef) {
                    cand_list_sort(result);
                    result->count = ef;
                }
            }
        }
    }

    cand_list_sort(result);
    cand_list_destroy(&candidates);
    visited_destroy(&visited);
    return 0;
}

/*
 * Heuristic neighbor selection (Algorithm 4 from HNSW paper).
 * From a candidate list, selects at most M neighbors that are both
 * close to the target AND diverse (not too close to each other).
 *
 * This prevents clustering of edges and maintains good graph connectivity.
 */
static void select_neighbors_heuristic(HnswIndex *idx,
                                        HnswCandidateList *candidates,
                                        uint32_t M_max,
                                        vec_dist_fn_t dist_fn,
                                        uint32_t *out_neighbors,
                                        uint16_t *out_count) {
    if (candidates->count == 0) {
        *out_count = 0;
        return;
    }

    cand_list_sort(candidates);

    uint16_t selected = 0;

    for (uint32_t i = 0; i < candidates->count && selected < (uint16_t)M_max; i++) {
        HnswCandidate *c = &candidates->items[i];

        bool good = true;
        for (uint16_t j = 0; j < selected; j++) {
            Page *p1, *p2;
            const float *v1 = get_node_vector(idx, c->node_id, &p1);
            const float *v2 = get_node_vector(idx, out_neighbors[j], &p2);

            if (v1 && v2) {
                float inter_dist = dist_fn(v1, v2, idx->dim);
                bp_unpin_page(idx->bp, idx->nodes[c->node_id].vec_pid, false);
                bp_unpin_page(idx->bp, idx->nodes[out_neighbors[j]].vec_pid, false);

                if (inter_dist < c->distance) {
                    good = false;
                    break;
                }
            } else {
                if (v1) bp_unpin_page(idx->bp, idx->nodes[c->node_id].vec_pid, false);
                if (v2) bp_unpin_page(idx->bp, idx->nodes[out_neighbors[j]].vec_pid, false);
            }
        }

        if (good) {
            out_neighbors[selected] = c->node_id;
            selected++;
        }
    }

    /* Fallback: keepPrunedConnections per Malkov & Yashunin (ensure connectivity) */
    if (selected < (uint16_t)M_max) {
        for (uint32_t i = 0; i < candidates->count && selected < (uint16_t)M_max; i++) {
            uint32_t cand_id = candidates->items[i].node_id;
            bool already = false;
            for (uint16_t j = 0; j < selected; j++) {
                if (out_neighbors[j] == cand_id) {
                    already = true;
                    break;
                }
            }
            if (!already) {
                out_neighbors[selected] = cand_id;
                selected++;
            }
        }
    }

    *out_count = selected;
}

/*
 * Add a bidirectional edge between two nodes on a given layer.
 * If the neighbor list is full, prune using the heuristic.
 */
static void connect_nodes(HnswIndex *idx, uint32_t node_a, uint32_t node_b,
                           uint8_t layer, vec_dist_fn_t dist_fn) {
    if (node_a == node_b) return;

    uint32_t M_max = (layer == 0) ? idx->M_max0 : idx->M;
    HnswNode *na = &idx->nodes[node_a];
    HnswNode *nb = &idx->nodes[node_b];

    for (uint16_t i = 0; i < na->counts[layer]; i++) {
        if (na->neighbors[layer][i] == node_b)
            return;
    }

    /* Add b to a's neighbor list */
    if (na->counts[layer] < (uint16_t)M_max) {
        na->neighbors[layer][na->counts[layer]] = node_b;
        na->counts[layer]++;
    } else {
        /* Need to prune: build candidate list including b */
        HnswCandidateList cands;
        cand_list_init(&cands, na->counts[layer] + 1);

        Page *pa;
        const float *va = get_node_vector(idx, node_a, &pa);
        if (va) {
            for (uint16_t i = 0; i < na->counts[layer]; i++) {
                float d = node_distance(idx, va, na->neighbors[layer][i], dist_fn);
                cand_list_push(&cands, na->neighbors[layer][i], d);
            }
            float d_new = node_distance(idx, va, node_b, dist_fn);
            cand_list_push(&cands, node_b, d_new);
            bp_unpin_page(idx->bp, idx->nodes[node_a].vec_pid, false);
        }

        uint32_t *new_neighbors = malloc(M_max * sizeof(uint32_t));
        uint16_t new_count = 0;
        if (new_neighbors) {
            select_neighbors_heuristic(idx, &cands, M_max, dist_fn,
                                       new_neighbors, &new_count);
            memcpy(na->neighbors[layer], new_neighbors,
                   new_count * sizeof(uint32_t));
            na->counts[layer] = new_count;
            free(new_neighbors);
        }
        cand_list_destroy(&cands);
    }

    /* Add a to b's neighbor list (symmetric) */
    if (nb->counts[layer] < (uint16_t)M_max) {
        nb->neighbors[layer][nb->counts[layer]] = node_a;
        nb->counts[layer]++;
    } else {
        HnswCandidateList cands;
        cand_list_init(&cands, nb->counts[layer] + 1);

        Page *pb;
        const float *vb = get_node_vector(idx, node_b, &pb);
        if (vb) {
            for (uint16_t i = 0; i < nb->counts[layer]; i++) {
                float d = node_distance(idx, vb, nb->neighbors[layer][i], dist_fn);
                cand_list_push(&cands, nb->neighbors[layer][i], d);
            }
            float d_new = node_distance(idx, vb, node_a, dist_fn);
            cand_list_push(&cands, node_a, d_new);
            bp_unpin_page(idx->bp, idx->nodes[node_b].vec_pid, false);
        }

        uint32_t *new_neighbors = malloc(M_max * sizeof(uint32_t));
        uint16_t new_count = 0;
        if (new_neighbors) {
            select_neighbors_heuristic(idx, &cands, M_max, dist_fn,
                                       new_neighbors, &new_count);
            memcpy(nb->neighbors[layer], new_neighbors,
                   new_count * sizeof(uint32_t));
            nb->counts[layer] = new_count;
            free(new_neighbors);
        }
        cand_list_destroy(&cands);
    }
}

/* ---- Vector page management (same pattern as FlatIndex) ---- */

static int grow_vec_page_array(HnswIndex *idx) {
    uint32_t new_cap = idx->vec_page_cap * 2;
    page_id_t *new_arr = realloc(idx->vec_page_ids,
                                  new_cap * sizeof(page_id_t));
    if (!new_arr) return -1;
    idx->vec_page_ids = new_arr;
    idx->vec_page_cap = new_cap;
    return 0;
}

static int alloc_new_hnsw_vec_page(HnswIndex *idx) {
    if (idx->num_vec_pages >= idx->vec_page_cap) {
        if (grow_vec_page_array(idx) < 0)
            return -1;
    }

    page_id_t new_pid;
    Page *p = bp_new_page(idx->bp, &new_pid);
    if (!p) return -1;

    vec_page_init(p, new_pid, idx->dim);
    bp_unpin_page(idx->bp, new_pid, true);

    idx->vec_page_ids[idx->num_vec_pages] = new_pid;
    idx->num_vec_pages++;
    return 0;
}

/* Store a vector on a vec page, returning the pid and slot. */
static int store_vector(HnswIndex *idx, const float *vec,
                         page_id_t *out_pid, uint16_t *out_slot) {
    if (idx->num_vec_pages == 0) {
        if (alloc_new_hnsw_vec_page(idx) < 0)
            return -1;
    }

    page_id_t last_pid = idx->vec_page_ids[idx->num_vec_pages - 1];
    Page *p = bp_fetch_page(idx->bp, last_pid);
    if (!p) return -1;

    int rc = vec_page_append(p, vec);
    if (rc == 0) {
        *out_slot = vec_page_count(p) - 1;
        *out_pid = last_pid;
        bp_unpin_page(idx->bp, last_pid, true);
        return 0;
    }

    bp_unpin_page(idx->bp, last_pid, false);

    if (alloc_new_hnsw_vec_page(idx) < 0)
        return -1;

    page_id_t new_pid = idx->vec_page_ids[idx->num_vec_pages - 1];
    p = bp_fetch_page(idx->bp, new_pid);
    if (!p) return -1;

    rc = vec_page_append(p, vec);
    if (rc < 0) {
        bp_unpin_page(idx->bp, new_pid, false);
        return -1;
    }

    *out_slot = vec_page_count(p) - 1;
    *out_pid = new_pid;
    bp_unpin_page(idx->bp, new_pid, true);
    return 0;
}

/* ---- Node allocation ---- */

static int grow_node_array(HnswIndex *idx) {
    uint32_t new_cap = idx->node_cap * 2;
    HnswNode *new_arr = realloc(idx->nodes, new_cap * sizeof(HnswNode));
    if (!new_arr) return -1;
    idx->nodes = new_arr;
    idx->node_cap = new_cap;
    return 0;
}

static int alloc_node(HnswIndex *idx, uint8_t level, page_id_t vec_pid,
                       uint16_t vec_slot, uint32_t *out_id) {
    if (idx->num_nodes >= idx->node_cap) {
        if (grow_node_array(idx) < 0)
            return -1;
    }

    uint32_t id = idx->num_nodes;
    HnswNode *node = &idx->nodes[id];
    memset(node, 0, sizeof(HnswNode));

    node->vec_pid = vec_pid;
    node->vec_slot = vec_slot;
    node->level = level;

    /* Allocate neighbor arrays for each layer up to this node's level */
    for (uint8_t l = 0; l <= level; l++) {
        uint32_t cap = (l == 0) ? idx->M_max0 : idx->M;
        node->neighbors[l] = malloc(cap * sizeof(uint32_t));
        if (!node->neighbors[l]) {
            /* Cleanup on failure */
            for (uint8_t j = 0; j < l; j++) {
                free(node->neighbors[j]);
                node->neighbors[j] = NULL;
            }
            return -1;
        }
        node->counts[l] = 0;
    }

    idx->num_nodes++;
    *out_id = id;
    return 0;
}

/* ============================================================
 *  Public API
 * ============================================================ */

int hnsw_index_create(HnswIndex *idx, BufferPool *bp,
                      uint16_t dim, VecDistanceMetric metric,
                      uint32_t M, uint32_t ef_construction) {
    if (dim == 0 || dim > VECTOR_MAX_DIM)
        return -1;

    memset(idx, 0, sizeof(HnswIndex));
    idx->bp = bp;
    idx->dim = dim;
    idx->metric = metric;

    idx->M = (M > 0) ? M : HNSW_DEFAULT_M;
    idx->M_max0 = idx->M * 2;
    idx->ef_construction = (ef_construction > 0)
        ? ef_construction : HNSW_DEFAULT_EF_CONSTRUCTION;
    idx->ef_search = HNSW_DEFAULT_EF_SEARCH;
    idx->ml = 1.0f / logf((float)idx->M);

    idx->nodes = malloc(INITIAL_NODE_CAP * sizeof(HnswNode));
    if (!idx->nodes) return -1;
    idx->node_cap = INITIAL_NODE_CAP;
    idx->num_nodes = 0;

    idx->entry_point = UINT32_MAX;
    idx->max_level = 0;

    idx->vec_page_ids = malloc(INITIAL_PAGE_CAP * sizeof(page_id_t));
    if (!idx->vec_page_ids) {
        free(idx->nodes);
        return -1;
    }
    idx->vec_page_cap = INITIAL_PAGE_CAP;
    idx->num_vec_pages = 0;

    return 0;
}

void hnsw_index_destroy(HnswIndex *idx) {
    if (idx->nodes) {
        for (uint32_t i = 0; i < idx->num_nodes; i++) {
            HnswNode *node = &idx->nodes[i];
            for (uint8_t l = 0; l <= node->level; l++) {
                free(node->neighbors[l]);
                node->neighbors[l] = NULL;
            }
        }
        free(idx->nodes);
        idx->nodes = NULL;
    }
    free(idx->vec_page_ids);
    idx->vec_page_ids = NULL;
    idx->num_nodes = 0;
    idx->node_cap = 0;
    idx->num_vec_pages = 0;
    idx->vec_page_cap = 0;
}

int hnsw_index_insert(HnswIndex *idx, const float *vec,
                      page_id_t *out_pid, uint16_t *out_slot) {
    if (!vec) return -1;

    vec_dist_fn_t dist_fn = vec_get_distance_fn(idx->metric);

    /* Store the vector on a vec page */
    page_id_t vec_pid;
    uint16_t vec_slot;
    if (store_vector(idx, vec, &vec_pid, &vec_slot) < 0)
        return -1;

    /* Assign random level */
    uint8_t new_level = random_level(idx->ml);

    /* Allocate the node */
    uint32_t new_id;
    if (alloc_node(idx, new_level, vec_pid, vec_slot, &new_id) < 0)
        return -1;

    if (out_pid)  *out_pid  = vec_pid;
    if (out_slot) *out_slot = vec_slot;

    /* First node: becomes the entry point */
    if (idx->num_nodes == 1) {
        idx->entry_point = new_id;
        idx->max_level = new_level;
        return 0;
    }

    /* Phase 1: Greedy descent from top layer to new_level+1 */
    uint32_t ep = idx->entry_point;
    for (int l = (int)idx->max_level; l > (int)new_level; l--) {
        ep = search_layer_greedy(idx, vec, ep, (uint8_t)l, dist_fn);
    }

    /* Phase 2: Beam search and connect on layers min(new_level, max_level)..0 */
    uint8_t insert_top = (new_level < idx->max_level) ? new_level : idx->max_level;

    for (int l = (int)insert_top; l >= 0; l--) {
        HnswCandidateList found;
        if (cand_list_init(&found, idx->ef_construction) < 0)
            return -1;

        if (search_layer_beam(idx, vec, ep, (uint8_t)l,
                              idx->ef_construction, dist_fn, &found) < 0) {
            cand_list_destroy(&found);
            return -1;
        }

        /* Select neighbors using heuristic */
        uint32_t M_max = (l == 0) ? idx->M_max0 : idx->M;
        uint32_t *selected = malloc(M_max * sizeof(uint32_t));
        uint16_t sel_count = 0;
        if (!selected) {
            cand_list_destroy(&found);
            return -1;
        }

        select_neighbors_heuristic(idx, &found, M_max, dist_fn,
                                   selected, &sel_count);

        /* Connect new node to selected neighbors */
        for (uint16_t i = 0; i < sel_count; i++) {
            connect_nodes(idx, new_id, selected[i], (uint8_t)l, dist_fn);
        }

        /* Update entry point for next layer */
        if (found.count > 0) {
            ep = found.items[0].node_id; /* closest found */
        }

        free(selected);
        cand_list_destroy(&found);
    }

    /* Update entry point if new node has higher level */
    if (new_level > idx->max_level) {
        idx->entry_point = new_id;
        idx->max_level = new_level;
    }

    return 0;
}

int hnsw_index_search(HnswIndex *idx, const float *query, uint32_t k,
                      uint32_t ef_search_override,
                      VecResult *results, uint32_t *num_results) {
    if (!query || !results || !num_results)
        return -1;
    if (k == 0) {
        *num_results = 0;
        return 0;
    }
    if (idx->num_nodes == 0) {
        *num_results = 0;
        return 0;
    }

    uint32_t ef = (ef_search_override > 0) ? ef_search_override : idx->ef_search;
    if (ef < k) ef = k; /* ef must be at least k */

    vec_dist_fn_t dist_fn = vec_get_distance_fn(idx->metric);

    /* Phase 1: Greedy descent from top layer to layer 1 */
    uint32_t ep = idx->entry_point;
    for (int l = (int)idx->max_level; l >= 1; l--) {
        ep = search_layer_greedy(idx, query, ep, (uint8_t)l, dist_fn);
    }

    /* Phase 2: Beam search on layer 0 */
    HnswCandidateList found;
    if (cand_list_init(&found, ef * 2) < 0)
        return -1;

    if (search_layer_beam(idx, query, ep, 0, ef, dist_fn, &found) < 0) {
        cand_list_destroy(&found);
        return -1;
    }

    /* Extract top-k from the found set */
    cand_list_sort(&found);
    uint32_t n = (found.count < k) ? found.count : k;

    for (uint32_t i = 0; i < n; i++) {
        results[i].page_id = idx->nodes[found.items[i].node_id].vec_pid;
        results[i].slot_index = idx->nodes[found.items[i].node_id].vec_slot;
        results[i].distance = found.items[i].distance;
    }
    *num_results = n;

    cand_list_destroy(&found);
    return 0;
}

uint32_t hnsw_index_count(const HnswIndex *idx) {
    return idx->num_nodes;
}

void hnsw_index_set_ef_search(HnswIndex *idx, uint32_t ef_search) {
    if (ef_search > 0)
        idx->ef_search = ef_search;
}

/* ============================================================
 *  Persistence (Save / Load via stream pages)
 * ============================================================ */

typedef struct {
    uint16_t dim;
    uint32_t metric;
    uint32_t M;
    uint32_t M_max0;
    uint32_t ef_construction;
    uint32_t ef_search;
    uint32_t num_nodes;
    uint32_t entry_point;
    uint8_t  max_level;
    uint8_t  _pad[3];
    uint32_t num_vec_pages;
    uint32_t total_bytes;
} HnswIndexMeta;

int hnsw_index_save(HnswIndex *idx, page_id_t *out_pid) {
    if (idx->num_nodes == 0) {
        size_t total = sizeof(HnswIndexMeta);
        HnswIndexMeta meta;
        memset(&meta, 0, sizeof(meta));
        meta.dim = idx->dim;
        meta.metric = (uint32_t)idx->metric;
        meta.M = idx->M;
        meta.M_max0 = idx->M_max0;
        meta.ef_construction = idx->ef_construction;
        meta.ef_search = idx->ef_search;
        meta.num_nodes = 0;
        meta.entry_point = idx->entry_point;
        meta.max_level = idx->max_level;
        meta.num_vec_pages = 0;
        meta.total_bytes = (uint32_t)total;
        return stream_write(idx->bp, &meta, total, out_pid);
    }

    /* Calculate total size */
    size_t total = sizeof(HnswIndexMeta);
    total += idx->num_vec_pages * sizeof(page_id_t);

    for (uint32_t i = 0; i < idx->num_nodes; i++) {
        HnswNode *node = &idx->nodes[i];
        total += sizeof(page_id_t) + sizeof(uint16_t) + sizeof(uint8_t) + 1; /* vec_pid, vec_slot, level, pad */
        for (uint8_t l = 0; l <= node->level; l++) {
            total += sizeof(uint16_t); /* count */
            total += node->counts[l] * sizeof(uint32_t);
        }
    }

    uint8_t *buf = malloc(total);
    if (!buf) return -1;

    uint8_t *ptr = buf;

    /* Write meta */
    HnswIndexMeta *meta = (HnswIndexMeta *)ptr;
    meta->dim = idx->dim;
    meta->metric = (uint32_t)idx->metric;
    meta->M = idx->M;
    meta->M_max0 = idx->M_max0;
    meta->ef_construction = idx->ef_construction;
    meta->ef_search = idx->ef_search;
    meta->num_nodes = idx->num_nodes;
    meta->entry_point = idx->entry_point;
    meta->max_level = idx->max_level;
    memset(meta->_pad, 0, sizeof(meta->_pad));
    meta->num_vec_pages = idx->num_vec_pages;
    meta->total_bytes = (uint32_t)total;
    ptr += sizeof(HnswIndexMeta);

    /* Write vec_page_ids */
    if (idx->num_vec_pages > 0) {
        memcpy(ptr, idx->vec_page_ids, idx->num_vec_pages * sizeof(page_id_t));
        ptr += idx->num_vec_pages * sizeof(page_id_t);
    }

    /* Write nodes */
    for (uint32_t i = 0; i < idx->num_nodes; i++) {
        HnswNode *node = &idx->nodes[i];

        memcpy(ptr, &node->vec_pid, sizeof(page_id_t)); ptr += sizeof(page_id_t);
        memcpy(ptr, &node->vec_slot, sizeof(uint16_t)); ptr += sizeof(uint16_t);
        *ptr++ = node->level;
        *ptr++ = 0; /* pad */

        for (uint8_t l = 0; l <= node->level; l++) {
            memcpy(ptr, &node->counts[l], sizeof(uint16_t)); ptr += sizeof(uint16_t);
            if (node->counts[l] > 0) {
                memcpy(ptr, node->neighbors[l],
                       node->counts[l] * sizeof(uint32_t));
                ptr += node->counts[l] * sizeof(uint32_t);
            }
        }
    }

    int rc = stream_write(idx->bp, buf, total, out_pid);
    free(buf);
    return rc;
}

int hnsw_index_load(HnswIndex *idx, BufferPool *bp, page_id_t pid) {
    HnswIndexMeta meta;
    if (stream_read(bp, pid, &meta, sizeof(HnswIndexMeta)) < 0)
        return -1;

    if (meta.total_bytes < sizeof(HnswIndexMeta))
        return -1;

    uint8_t *full_buf = malloc(meta.total_bytes);
    if (!full_buf) return -1;

    if (stream_read(bp, pid, full_buf, meta.total_bytes) < 0) {
        free(full_buf);
        return -1;
    }

    /* Initialize the index structure */
    memset(idx, 0, sizeof(HnswIndex));
    idx->bp = bp;
    idx->dim = meta.dim;
    idx->metric = (VecDistanceMetric)meta.metric;
    idx->M = meta.M;
    idx->M_max0 = meta.M_max0;
    idx->ef_construction = meta.ef_construction;
    idx->ef_search = meta.ef_search;
    idx->ml = 1.0f / logf((float)idx->M);
    idx->entry_point = meta.entry_point;
    idx->max_level = meta.max_level;

    /* Restore vec_page_ids */
    idx->num_vec_pages = meta.num_vec_pages;
    idx->vec_page_cap = meta.num_vec_pages > INITIAL_PAGE_CAP
        ? meta.num_vec_pages : INITIAL_PAGE_CAP;
    if (idx->vec_page_cap > 0) {
        idx->vec_page_ids = malloc(idx->vec_page_cap * sizeof(page_id_t));
        if (!idx->vec_page_ids) {
            free(full_buf);
            return -1;
        }
        if (meta.num_vec_pages > 0) {
            memcpy(idx->vec_page_ids, full_buf + sizeof(HnswIndexMeta),
                   meta.num_vec_pages * sizeof(page_id_t));
        }
    }

    /* Allocate nodes */
    idx->node_cap = meta.num_nodes > INITIAL_NODE_CAP
        ? meta.num_nodes : INITIAL_NODE_CAP;
    if (idx->node_cap > 0) {
        idx->nodes = malloc(idx->node_cap * sizeof(HnswNode));
        if (!idx->nodes) {
            free(idx->vec_page_ids);
            free(full_buf);
            return -1;
        }
    }
    idx->num_nodes = 0;

    /* Parse nodes from the buffer */
    uint8_t *ptr = full_buf + sizeof(HnswIndexMeta) + meta.num_vec_pages * sizeof(page_id_t);

    for (uint32_t i = 0; i < meta.num_nodes; i++) {
        HnswNode *node = &idx->nodes[i];
        memset(node, 0, sizeof(HnswNode));

        memcpy(&node->vec_pid, ptr, sizeof(page_id_t)); ptr += sizeof(page_id_t);
        memcpy(&node->vec_slot, ptr, sizeof(uint16_t)); ptr += sizeof(uint16_t);
        node->level = *ptr++;
        ptr++; /* skip pad */

        for (uint8_t l = 0; l <= node->level && l < HNSW_MAX_LEVELS; l++) {
            memcpy(&node->counts[l], ptr, sizeof(uint16_t)); ptr += sizeof(uint16_t);

            uint32_t cap = (l == 0) ? idx->M_max0 : idx->M;
            node->neighbors[l] = malloc(cap * sizeof(uint32_t));
            if (!node->neighbors[l]) {
                for (uint8_t j = 0; j < l; j++) {
                    free(node->neighbors[j]);
                }
                for (uint32_t k = 0; k < i; k++) {
                    for (uint8_t ll = 0; ll <= idx->nodes[k].level; ll++)
                        free(idx->nodes[k].neighbors[ll]);
                }
                free(full_buf);
                free(idx->nodes);
                free(idx->vec_page_ids);
                return -1;
            }

            if (node->counts[l] > 0) {
                memcpy(node->neighbors[l], ptr,
                       node->counts[l] * sizeof(uint32_t));
                ptr += node->counts[l] * sizeof(uint32_t);
            }
        }

        idx->num_nodes++;
    }

    free(full_buf);
    return 0;
}
