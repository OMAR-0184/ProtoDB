#ifndef STORAGE_MGR_H
#define STORAGE_MGR_H

#include "page.h"

#define PROTO_MAGIC      0xDB1234
#define SCHEMA_VERSION   4
#define META_PAGE_ID     0

typedef struct {
    uint32_t   magic;
    uint32_t   schema_version;
    page_id_t  page_count;
    page_id_t  free_list_head;
    page_id_t  root_index_pid;
    uint8_t    root_index_type; /* 0=None, 1=Flat, 2=IVF */
    uint8_t    _pad[3];
    page_id_t  root_meta_store_pid; /* Root page for MetadataStore stream */
} MetaPageHeader;

typedef struct {
    int            fd;
    char           filepath[256];
    MetaPageHeader meta;
} StorageManager;

int       storage_open(StorageManager *sm, const char *path);
int       storage_close(StorageManager *sm);
int       storage_read_page(StorageManager *sm, page_id_t page_id, Page *p);
int       storage_write_page(StorageManager *sm, page_id_t page_id, Page *p);
int       storage_allocate_page(StorageManager *sm, page_id_t *out);
int       storage_deallocate_page(StorageManager *sm, page_id_t page_id);
page_id_t storage_num_pages(StorageManager *sm);
int       storage_read_pages(StorageManager *sm, page_id_t start, uint32_t count, Page *pages);

#endif
