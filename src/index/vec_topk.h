#ifndef VEC_TOPK_H
#define VEC_TOPK_H

#include "../pages/page.h"

/*
 * A single search result: identifies a vector by its page + slot,
 * along with the computed distance from the query.
 */
typedef struct {
    page_id_t page_id;
    uint16_t  slot_index;
    float     distance;
} VecResult;

/*
 * Max-heap that maintains the k smallest distances seen so far.
 *
 * The heap root is always the *largest* distance in the set.
 * A new candidate is inserted only if it's smaller than the root
 * (or the heap isn't full yet), which keeps the heap at most k entries.
 */
typedef struct {
    VecResult *items;
    uint32_t   capacity;   /* k */
    uint32_t   count;
} VecTopK;

/* Allocate heap internals.  k = max number of results. */
int  vec_topk_init(VecTopK *tk, uint32_t k);
void vec_topk_destroy(VecTopK *tk);

/*
 * Offer a candidate to the heap.
 * Inserted only if dist < current worst, or heap isn't full yet.
 */
void vec_topk_push(VecTopK *tk, page_id_t pid, uint16_t slot, float dist);

/* Current worst (largest) distance in the heap.  +INF if empty. */
float vec_topk_threshold(const VecTopK *tk);

/* Sort results ascending by distance (call once after the scan). */
void vec_topk_sort(VecTopK *tk);

#endif
