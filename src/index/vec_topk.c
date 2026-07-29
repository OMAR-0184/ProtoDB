#include "vec_topk.h"
#include <stdlib.h>
#include <math.h>

/* ---- internal heap helpers ---- */

static void swap(VecResult *a, VecResult *b) {
    VecResult tmp = *a;
    *a = *b;
    *b = tmp;
}

static void sift_up(VecResult *items, uint32_t idx) {
    while (idx > 0) {
        uint32_t parent = (idx - 1) / 2;
        if (items[idx].distance > items[parent].distance) {
            swap(&items[idx], &items[parent]);
            idx = parent;
        } else {
            break;
        }
    }
}

static void sift_down(VecResult *items, uint32_t count, uint32_t idx) {
    while (1) {
        uint32_t largest = idx;
        uint32_t left    = 2 * idx + 1;
        uint32_t right   = 2 * idx + 2;

        if (left < count && items[left].distance > items[largest].distance)
            largest = left;
        if (right < count && items[right].distance > items[largest].distance)
            largest = right;

        if (largest == idx)
            break;

        swap(&items[idx], &items[largest]);
        idx = largest;
    }
}

/* ---- public API ---- */

int vec_topk_init(VecTopK *tk, uint32_t k) {
    if (k == 0)
        return -1;

    tk->items = malloc(k * sizeof(VecResult));
    if (!tk->items)
        return -1;

    tk->capacity = k;
    tk->count    = 0;
    return 0;
}

void vec_topk_destroy(VecTopK *tk) {
    free(tk->items);
    tk->items    = NULL;
    tk->capacity = 0;
    tk->count    = 0;
}

void vec_topk_push(VecTopK *tk, page_id_t pid, uint16_t slot, float dist) {
    if (tk->count < tk->capacity) {
        /* Heap not full yet — always insert */
        VecResult *r = &tk->items[tk->count];
        r->page_id    = pid;
        r->slot_index = slot;
        r->distance   = dist;
        sift_up(tk->items, tk->count);
        tk->count++;
    } else if (dist < tk->items[0].distance) {
        /* Better than current worst — replace root */
        VecResult *root = &tk->items[0];
        root->page_id    = pid;
        root->slot_index = slot;
        root->distance   = dist;
        sift_down(tk->items, tk->count, 0);
    }
}

float vec_topk_threshold(const VecTopK *tk) {
    if (tk->count == 0)
        return INFINITY;
    return tk->items[0].distance;
}

/* Comparison for qsort: ascending by distance */
static int cmp_by_distance(const void *a, const void *b) {
    float da = ((const VecResult *)a)->distance;
    float db = ((const VecResult *)b)->distance;
    if (da < db) return -1;
    if (da > db) return  1;
    return 0;
}

void vec_topk_sort(VecTopK *tk) {
    if (tk->count <= 1)
        return;
    qsort(tk->items, tk->count, sizeof(VecResult), cmp_by_distance);
}
