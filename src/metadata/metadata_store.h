#ifndef METADATA_STORE_H
#define METADATA_STORE_H

#include <stdint.h>
#include <stdbool.h>
#include "../pages/page.h"
#include "../buffer/buffer_pool.h"

typedef struct {
    page_id_t pid;
    uint16_t slot;
    char *metadata;
} MetadataEntry;

typedef struct {
    MetadataEntry *entries;
    uint32_t count;
    uint32_t capacity;
    BufferPool *bp;
} MetadataStore;

int meta_store_init(MetadataStore *ms, BufferPool *bp);
void meta_store_destroy(MetadataStore *ms);

/* Set or update metadata for a vector */
int meta_store_set(MetadataStore *ms, page_id_t pid, uint16_t slot, const char *metadata);

/* Get metadata for a vector (returns NULL if not found) */
const char *meta_store_get(const MetadataStore *ms, page_id_t pid, uint16_t slot);

/* Remove metadata for a vector */
int meta_store_delete(MetadataStore *ms, page_id_t pid, uint16_t slot);

/* Persist to disk using Stream Pages */
int meta_store_save(MetadataStore *ms, page_id_t *out_pid);

/* Load from disk */
int meta_store_load(MetadataStore *ms, BufferPool *bp, page_id_t pid);

#endif
