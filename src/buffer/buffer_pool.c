#include "buffer_pool.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t hash_page_id(page_id_t id, uint32_t capacity) {
  uint32_t h = id * 2654435769u;
  return h & (capacity - 1);
}

static int32_t pt_lookup(BufferPool *bp, page_id_t page_id) {
  uint32_t idx = hash_page_id(page_id, bp->pt_capacity);

  for (uint32_t i = 0; i < bp->pt_capacity; i++) {
    uint32_t slot = (idx + i) & (bp->pt_capacity - 1);
    if (bp->pt_keys[slot] == BP_EMPTY_SLOT)
      return -1;
    if (bp->pt_keys[slot] == page_id)
      return bp->pt_vals[slot];
  }
  return -1;
}

static void pt_insert(BufferPool *bp, page_id_t page_id, int32_t frame_id) {
  uint32_t idx = hash_page_id(page_id, bp->pt_capacity);

  for (uint32_t i = 0; i < bp->pt_capacity; i++) {
    uint32_t slot = (idx + i) & (bp->pt_capacity - 1);
    if (bp->pt_keys[slot] == BP_EMPTY_SLOT) {
      bp->pt_keys[slot] = page_id;
      bp->pt_vals[slot] = frame_id;
      return;
    }
    if (bp->pt_keys[slot] == page_id) {
      bp->pt_vals[slot] = frame_id;
      return;
    }
  }
}

static void pt_remove(BufferPool *bp, page_id_t page_id) {
  uint32_t idx = hash_page_id(page_id, bp->pt_capacity);

  for (uint32_t i = 0; i < bp->pt_capacity; i++) {
    uint32_t slot = (idx + i) & (bp->pt_capacity - 1);
    if (bp->pt_keys[slot] == BP_EMPTY_SLOT)
      return;
    if (bp->pt_keys[slot] == page_id) {
      bp->pt_keys[slot] = BP_EMPTY_SLOT;
      bp->pt_vals[slot] = -1;

      uint32_t next = (slot + 1) & (bp->pt_capacity - 1);
      while (bp->pt_keys[next] != BP_EMPTY_SLOT) {
        page_id_t k = bp->pt_keys[next];
        int32_t v = bp->pt_vals[next];
        bp->pt_keys[next] = BP_EMPTY_SLOT;
        bp->pt_vals[next] = -1;
        pt_insert(bp, k, v);
        next = (next + 1) & (bp->pt_capacity - 1);
      }
      return;
    }
  }
}

static int32_t clock_find_victim(BufferPool *bp) {
  uint32_t max_sweeps = 2 * bp->num_frames;

  for (uint32_t i = 0; i < max_sweeps; i++) {
    uint32_t frame = bp->clock_hand;
    bp->clock_hand = (bp->clock_hand + 1) % bp->num_frames;

    if (bp->frames[frame].pin_count > 0)
      continue;

    if (bp->ref_bits[frame]) {
      bp->ref_bits[frame] = false;
    } else {
      return (int32_t)frame;
    }
  }

  return -1;
}

int bp_init(BufferPool *bp, StorageManager *sm, uint32_t num_frames) {
  if (num_frames == 0)
    return -1;

  memset(bp, 0, sizeof(BufferPool));
  bp->sm = sm;
  bp->num_frames = num_frames;
  bp->clock_hand = 0;

  uint32_t cap = 1;
  while (cap < num_frames * 2)
    cap <<= 1;
  bp->pt_capacity = cap;

  bp->frames = calloc(num_frames, sizeof(Page));
  bp->ref_bits = calloc(num_frames, sizeof(bool));
  bp->pt_keys = malloc(cap * sizeof(page_id_t));
  bp->pt_vals = malloc(cap * sizeof(int32_t));

  if (!bp->frames || !bp->ref_bits || !bp->pt_keys || !bp->pt_vals) {
    bp_destroy(bp);
    return -1;
  }

  for (uint32_t i = 0; i < cap; i++) {
    bp->pt_keys[i] = BP_EMPTY_SLOT;
    bp->pt_vals[i] = -1;
  }

  for (uint32_t i = 0; i < num_frames; i++) {
    bp->frames[i].id = INVALID_PAGE_ID;
    pthread_rwlock_init(&bp->frames[i].rwlatch, NULL);
  }

  pthread_mutex_init(&bp->latch, NULL);

  return 0;
}

void bp_destroy(BufferPool *bp) {
  if (bp->frames && bp->sm) {
    for (uint32_t i = 0; i < bp->num_frames; i++) {
      if (bp->frames[i].is_dirty && bp->frames[i].id != INVALID_PAGE_ID)
        storage_write_page(bp->sm, bp->frames[i].id, &bp->frames[i]);
    }
  }

  free(bp->frames);
  free(bp->ref_bits);
  free(bp->pt_keys);
  free(bp->pt_vals);

  for (uint32_t i = 0; i < bp->num_frames; i++) {
    pthread_rwlock_destroy(&bp->frames[i].rwlatch);
  }
  pthread_mutex_destroy(&bp->latch);

  memset(bp, 0, sizeof(BufferPool));
}

Page *bp_fetch_page(BufferPool *bp, page_id_t page_id) {
  pthread_mutex_lock(&bp->latch);

  int32_t frame = pt_lookup(bp, page_id);
  if (frame >= 0) {
    bp->frames[frame].pin_count++;
    bp->ref_bits[frame] = true;
    Page *p = &bp->frames[frame];
    pthread_mutex_unlock(&bp->latch);
    return p;
  }

  int32_t victim = -1;

  for (uint32_t i = 0; i < bp->num_frames; i++) {
    if (bp->frames[i].id == INVALID_PAGE_ID) {
      victim = (int32_t)i;
      break;
    }
  }

  if (victim < 0) {
    victim = clock_find_victim(bp);
    if (victim < 0) {
      pthread_mutex_unlock(&bp->latch);
      return NULL;
    }

    Page *vp = &bp->frames[victim];
    if (vp->is_dirty) {
      if (storage_write_page(bp->sm, vp->id, vp) < 0) {
        pthread_mutex_unlock(&bp->latch);
        return NULL;
      }
    }

    pt_remove(bp, vp->id);
  }

  Page *p = &bp->frames[victim];
  if (storage_read_page(bp->sm, page_id, p) < 0) {
    pthread_mutex_unlock(&bp->latch);
    return NULL;
  }

  p->pin_count = 1;
  p->is_dirty = false;
  bp->ref_bits[victim] = true;

  pt_insert(bp, page_id, victim);
  pthread_mutex_unlock(&bp->latch);
  return p;
}

int bp_unpin_page(BufferPool *bp, page_id_t page_id, bool is_dirty) {
  pthread_mutex_lock(&bp->latch);
  
  int32_t frame = pt_lookup(bp, page_id);
  if (frame < 0) {
    pthread_mutex_unlock(&bp->latch);
    return -1;
  }

  Page *p = &bp->frames[frame];
  if (p->pin_count <= 0) {
    pthread_mutex_unlock(&bp->latch);
    return -1;
  }

  p->pin_count--;
  if (is_dirty)
    p->is_dirty = true;

  pthread_mutex_unlock(&bp->latch);
  return 0;
}

int bp_flush_page(BufferPool *bp, page_id_t page_id) {
  pthread_mutex_lock(&bp->latch);

  int32_t frame = pt_lookup(bp, page_id);
  if (frame < 0) {
    pthread_mutex_unlock(&bp->latch);
    return -1;
  }

  Page *p = &bp->frames[frame];
  if (storage_write_page(bp->sm, page_id, p) < 0) {
    pthread_mutex_unlock(&bp->latch);
    return -1;
  }

  p->is_dirty = false;
  pthread_mutex_unlock(&bp->latch);
  return 0;
}

int bp_flush_all(BufferPool *bp) {
  pthread_mutex_lock(&bp->latch);
  for (uint32_t i = 0; i < bp->num_frames; i++) {
    if (bp->frames[i].id != INVALID_PAGE_ID && bp->frames[i].is_dirty) {
      if (storage_write_page(bp->sm, bp->frames[i].id, &bp->frames[i]) < 0) {
        pthread_mutex_unlock(&bp->latch);
        return -1;
      }
      bp->frames[i].is_dirty = false;
    }
  }
  pthread_mutex_unlock(&bp->latch);
  return 0;
}

Page *bp_new_page(BufferPool *bp, page_id_t *out) {
  pthread_mutex_lock(&bp->latch);
  
  page_id_t new_id;
  if (storage_allocate_page(bp->sm, &new_id) < 0) {
    pthread_mutex_unlock(&bp->latch);
    return NULL;
  }

  int32_t victim = -1;

  for (uint32_t i = 0; i < bp->num_frames; i++) {
    if (bp->frames[i].id == INVALID_PAGE_ID) {
      victim = (int32_t)i;
      break;
    }
  }

  if (victim < 0) {
    victim = clock_find_victim(bp);
    if (victim < 0) {
      storage_deallocate_page(bp->sm, new_id);
      pthread_mutex_unlock(&bp->latch);
      return NULL;
    }

    Page *vp = &bp->frames[victim];
    if (vp->is_dirty) {
      if (storage_write_page(bp->sm, vp->id, vp) < 0) {
        storage_deallocate_page(bp->sm, new_id);
        pthread_mutex_unlock(&bp->latch);
        return NULL;
      }
    }
    pt_remove(bp, vp->id);
  }

  Page *p = &bp->frames[victim];
  page_init(p, new_id, PAGE_TYPE_DATA);
  p->pin_count = 1;
  p->is_dirty = true;
  bp->ref_bits[victim] = true;

  pt_insert(bp, new_id, victim);
  *out = new_id;
  pthread_mutex_unlock(&bp->latch);
  return p;
}

int bp_delete_page(BufferPool *bp, page_id_t page_id) {
  pthread_mutex_lock(&bp->latch);
  int32_t frame = pt_lookup(bp, page_id);

  if (frame >= 0) {
    Page *p = &bp->frames[frame];
    if (p->pin_count > 0) {
      pthread_mutex_unlock(&bp->latch);
      return -1;
    }

    pt_remove(bp, page_id);
    p->id = INVALID_PAGE_ID;
    p->is_dirty = false;
    p->pin_count = 0;
    bp->ref_bits[frame] = false;
  }

  int ret = storage_deallocate_page(bp->sm, page_id);
  pthread_mutex_unlock(&bp->latch);
  return ret;
}