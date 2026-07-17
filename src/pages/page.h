#ifndef PAGE_H
#define PAGE_H

#include <stdint.h>
#include <stdbool.h>
#include <string.h>

#define PAGE_SIZE        4096
#define INVALID_PAGE_ID  UINT32_MAX

#define PAGE_TYPE_FREE   0
#define PAGE_TYPE_DATA   1
#define PAGE_TYPE_INDEX  2
#define PAGE_TYPE_META   3

typedef uint32_t page_id_t;

typedef struct {
    page_id_t  page_id;
    uint16_t   num_slots;
    uint16_t   free_space_end;
    uint8_t    page_type;
    uint8_t    _pad[3];
} PageHeader;

#define PAGE_DATA_OFFSET sizeof(PageHeader)

typedef struct {
    uint16_t offset;
    uint16_t length;
} SlotEntry;

#define SLOT_SIZE sizeof(SlotEntry)

typedef struct {
    page_id_t  id;
    int        pin_count;
    bool       is_dirty;
    uint8_t    data[PAGE_SIZE];
} Page;

static inline PageHeader *page_get_header(Page *p) {
    return (PageHeader *)p->data;
}

static inline SlotEntry *page_get_slot(Page *p, uint16_t slot_index) {
    return (SlotEntry *)(p->data + PAGE_DATA_OFFSET + slot_index * SLOT_SIZE);
}

void     page_init(Page *p, page_id_t id, uint8_t page_type);
uint16_t page_free_space(Page *p);
int      page_insert_record(Page *p, const void *data, uint16_t len, uint16_t *slot_out);
int      page_get_record(Page *p, uint16_t slot_index, uint8_t **out, uint16_t *len_out);
int      page_delete_record(Page *p, uint16_t slot_index);
int      page_update_record(Page *p, uint16_t slot_index, const void *data, uint16_t len);
void     page_compact(Page *p);

#endif