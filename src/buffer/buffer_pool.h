#ifndef BUFFER_POOL_H
#define BUFFER_POOL_H

#include "../pages/page.h"
#include "../pages/storage_mgr.h"

#define BP_EMPTY_SLOT INVALID_PAGE_ID

typedef struct {
  Page *frames;
  page_id_t *pt_keys;
  int32_t *pt_vals;
  uint32_t pt_capacity;
  bool *ref_bits;
  uint32_t clock_hand;
  uint32_t num_frames;
  StorageManager *sm;
} BufferPool;

int bp_init(BufferPool *bp, StorageManager *sm, uint32_t num_frames);
void bp_destroy(BufferPool *bp);

Page *bp_fetch_page(BufferPool *bp, page_id_t page_id);
int bp_unpin_page(BufferPool *bp, page_id_t page_id, bool is_dirty);
int bp_flush_page(BufferPool *bp, page_id_t page_id);
int bp_flush_all(BufferPool *bp);

Page *bp_new_page(BufferPool *bp, page_id_t *out);
int bp_delete_page(BufferPool *bp, page_id_t page_id);

#endif
