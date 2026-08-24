#include "metadata_store.h"
#include "../pages/stream_page.h"
#include <stdlib.h>
#include <string.h>

#define INITIAL_CAPACITY 64

int meta_store_init(MetadataStore *ms, BufferPool *bp) {
    ms->bp = bp;
    ms->count = 0;
    ms->capacity = INITIAL_CAPACITY;
    ms->entries = malloc(ms->capacity * sizeof(MetadataEntry));
    if (!ms->entries) return -1;
    return 0;
}

void meta_store_destroy(MetadataStore *ms) {
    if (ms->entries) {
        for (uint32_t i = 0; i < ms->count; i++) {
            free(ms->entries[i].metadata);
        }
        free(ms->entries);
    }
    ms->entries = NULL;
    ms->count = 0;
    ms->capacity = 0;
}

/* Helper to find entry index */
static int find_entry(const MetadataStore *ms, page_id_t pid, uint16_t slot) {
    /* Naive linear scan for simplicity. Could be a hash table for better performance. */
    for (uint32_t i = 0; i < ms->count; i++) {
        if (ms->entries[i].pid == pid && ms->entries[i].slot == slot) {
            return (int)i;
        }
    }
    return -1;
}

int meta_store_set(MetadataStore *ms, page_id_t pid, uint16_t slot, const char *metadata) {
    if (!ms || !metadata) return -1;

    int idx = find_entry(ms, pid, slot);
    if (idx >= 0) {
        char *new_meta = strdup(metadata);
        if (!new_meta) return -1;
        free(ms->entries[idx].metadata);
        ms->entries[idx].metadata = new_meta;
        return 0;
    }

    if (ms->count >= ms->capacity) {
        uint32_t new_cap = ms->capacity * 2;
        MetadataEntry *new_entries = realloc(ms->entries, new_cap * sizeof(MetadataEntry));
        if (!new_entries) return -1;
        ms->entries = new_entries;
        ms->capacity = new_cap;
    }

    ms->entries[ms->count].pid = pid;
    ms->entries[ms->count].slot = slot;
    ms->entries[ms->count].metadata = strdup(metadata);
    if (!ms->entries[ms->count].metadata) return -1;
    
    ms->count++;
    return 0;
}

const char *meta_store_get(const MetadataStore *ms, page_id_t pid, uint16_t slot) {
    if (!ms) return NULL;
    int idx = find_entry(ms, pid, slot);
    if (idx >= 0) return ms->entries[idx].metadata;
    return NULL;
}

int meta_store_delete(MetadataStore *ms, page_id_t pid, uint16_t slot) {
    if (!ms) return -1;
    int idx = find_entry(ms, pid, slot);
    if (idx < 0) return -1;

    free(ms->entries[idx].metadata);
    
    /* Swap with last to keep array contiguous */
    if ((uint32_t)idx < ms->count - 1) {
        ms->entries[idx] = ms->entries[ms->count - 1];
    }
    ms->count--;
    return 0;
}

int meta_store_save(MetadataStore *ms, page_id_t *out_pid) {
    if (ms->count == 0) {
        *out_pid = INVALID_PAGE_ID;
        return 0;
    }

    /* Calculate total size required */
    uint32_t total_size = sizeof(uint32_t); /* for count */
    for (uint32_t i = 0; i < ms->count; i++) {
        if (!ms->entries[i].metadata) return -1;
        total_size += sizeof(page_id_t) + sizeof(uint16_t);
        uint32_t len = strlen(ms->entries[i].metadata);
        total_size += sizeof(uint32_t) + len; /* length + chars (no null terminator needed) */
    }

    uint8_t *buf = malloc(total_size);
    if (!buf) return -1;

    uint32_t offset = 0;
    memcpy(buf + offset, &ms->count, sizeof(uint32_t));
    offset += sizeof(uint32_t);

    for (uint32_t i = 0; i < ms->count; i++) {
        memcpy(buf + offset, &ms->entries[i].pid, sizeof(page_id_t));
        offset += sizeof(page_id_t);
        
        memcpy(buf + offset, &ms->entries[i].slot, sizeof(uint16_t));
        offset += sizeof(uint16_t);
        
        uint32_t len = strlen(ms->entries[i].metadata);
        memcpy(buf + offset, &len, sizeof(uint32_t));
        offset += sizeof(uint32_t);
        
        memcpy(buf + offset, ms->entries[i].metadata, len);
        offset += len;
    }

    int rc = stream_write(ms->bp, buf, total_size, out_pid);
    free(buf);
    return rc;
}

int meta_store_load(MetadataStore *ms, BufferPool *bp, page_id_t pid) {
    if (meta_store_init(ms, bp) < 0) return -1;
    
    if (pid == INVALID_PAGE_ID) return 0;

    /* Calculate total stream length by traversing linked pages */
    size_t total_bytes = 0;
    page_id_t curr = pid;
    while (curr != INVALID_PAGE_ID) {
        Page *p = bp_fetch_page(bp, curr);
        if (!p) { meta_store_destroy(ms); return -1; }
        StreamPageHeader *hdr = (StreamPageHeader *)(p->data + STREAM_PAGE_HEADER_OFFSET);
        total_bytes += hdr->data_size;
        curr = hdr->next_page_id;
        bp_unpin_page(bp, p->id, false);
    }

    uint8_t *buf = malloc(total_bytes);
    if (!buf) { meta_store_destroy(ms); return -1; }

    if (stream_read(bp, pid, buf, total_bytes) < 0) {
        free(buf);
        meta_store_destroy(ms);
        return -1;
    }

    uint32_t offset = 0;
    uint32_t count = 0;
    if (offset + sizeof(uint32_t) > total_bytes) { free(buf); meta_store_destroy(ms); return -1; }
    memcpy(&count, buf + offset, sizeof(uint32_t));
    offset += sizeof(uint32_t);

    for (uint32_t i = 0; i < count; i++) {
        page_id_t entry_pid;
        uint16_t entry_slot;
        uint32_t len;
        
        if (offset + sizeof(page_id_t) > total_bytes) { free(buf); meta_store_destroy(ms); return -1; }
        memcpy(&entry_pid, buf + offset, sizeof(page_id_t));
        offset += sizeof(page_id_t);
        
        if (offset + sizeof(uint16_t) > total_bytes) { free(buf); meta_store_destroy(ms); return -1; }
        memcpy(&entry_slot, buf + offset, sizeof(uint16_t));
        offset += sizeof(uint16_t);
        
        if (offset + sizeof(uint32_t) > total_bytes) { free(buf); meta_store_destroy(ms); return -1; }
        memcpy(&len, buf + offset, sizeof(uint32_t));
        offset += sizeof(uint32_t);
        
        if (offset + len > total_bytes) { free(buf); meta_store_destroy(ms); return -1; }
        char *meta = malloc(len + 1);
        if (!meta) { free(buf); meta_store_destroy(ms); return -1; }
        
        memcpy(meta, buf + offset, len);
        meta[len] = '\0';
        offset += len;
        
        if (meta_store_set(ms, entry_pid, entry_slot, meta) < 0) {
            free(meta);
            free(buf);
            meta_store_destroy(ms);
            return -1;
        }
        free(meta);
    }

    free(buf);
    return 0;
}
