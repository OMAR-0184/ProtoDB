#ifndef VEC_PAGE_H
#define VEC_PAGE_H

#include "../pages/page.h"

/*
 * Vector page layout (within the 8 KB page data area):
 *
 *   [PageHeader]           — 12 bytes (existing, at offset 0)
 *   [VecPageHeader]        — 8 bytes  (at PAGE_DATA_OFFSET)
 *   [Vector 0: float×dim]  — dim*4 bytes
 *   [Vector 1: float×dim]  — dim*4 bytes
 *   ...
 *
 * All vectors on a page share the same dimensionality.
 * Vectors are stored contiguously with no gaps — no slot array needed.
 */

typedef struct {
  uint16_t dim;         /* vector dimensionality */
  uint16_t num_vectors; /* number of vectors currently stored */
  uint32_t _reserved;
  uint64_t
      deleted_bitmap[16]; /* 128 bytes = 1024 bits for deletion tombstones */
} VecPageHeader;

#define VEC_PAGE_HEADER_OFFSET PAGE_DATA_OFFSET
#define VEC_DATA_OFFSET                                                        \
  (VEC_PAGE_HEADER_OFFSET + (uint16_t)sizeof(VecPageHeader))

/* Initialize a page as a vector page with the given dimension. */
void vec_page_init(Page *p, page_id_t id, uint16_t dim);

/* How many vectors of this dimension can fit on one page. */
uint16_t vec_page_capacity(uint16_t dim);

/* Append a vector to the page.  Returns 0 on success, -1 if full. */
int vec_page_append(Page *p, const float *vec);

/*
 * Get a read-only pointer to vector at the given index.
 * Returns NULL if index is out of bounds.
 * The pointer is directly into the page data — zero copy.
 */
const float *vec_page_get(const Page *p, uint16_t index);

/* Number of vectors currently stored on this page (including deleted). */
uint16_t vec_page_count(const Page *p);

/* Mark the vector at the given index as deleted. Returns 0 on success, -1 on
 * bounds/already deleted. */
int vec_page_delete(Page *p, uint16_t index);

/* Check if the vector at the given index is deleted. */
bool vec_page_is_deleted(const Page *p, uint16_t index);

/* Dimension of vectors on this page. */
uint16_t vec_page_dim(const Page *p);

#endif
