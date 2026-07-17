#ifndef STORAGE_MGR_H
#define STORAGE_MGR_H

#include "page.h"

#define PROTO_MAGIC      0xDB1234
#define SCHEMA_VERSION   1
#define META_PAGE_ID     0

typedef struct {
    uint32_t   magic;
    uint32_t   schema_version;
    page_id_t  page_count;
    page_id_t  free_list_head;
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

#endif
