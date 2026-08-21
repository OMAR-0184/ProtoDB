#include "page.h"
#include <string.h>
#include <stdlib.h>

void page_init(Page *p, page_id_t id, uint8_t page_type) {
    memset(p->data, 0, PAGE_SIZE);
    p->id        = id;
    p->pin_count = 0;
    p->is_dirty  = false;

    PageHeader *hdr = page_get_header(p);
    hdr->page_id        = id;
    hdr->num_slots      = 0;
    hdr->free_space_end = PAGE_SIZE;
    hdr->page_type      = page_type;
}

uint16_t page_free_space(Page *p) {
    PageHeader *hdr = page_get_header(p);
    uint16_t slot_array_end = (uint16_t)(PAGE_DATA_OFFSET + hdr->num_slots * SLOT_SIZE);

    if (hdr->free_space_end <= slot_array_end)
        return 0;

    return hdr->free_space_end - slot_array_end;
}

int page_insert_record(Page *p, const void *data, uint16_t len, uint16_t *slot_out) {
    if (len == 0)
        return -1;

    uint16_t needed = len + SLOT_SIZE;

    int16_t reuse_slot = -1;
    PageHeader *hdr = page_get_header(p);
    for (uint16_t i = 0; i < hdr->num_slots; i++) {
        SlotEntry *s = page_get_slot(p, i);
        if (s->offset == 0 && s->length == 0) {
            reuse_slot = (int16_t)i;
            needed = len;
            break;
        }
    }

    if (page_free_space(p) < needed)
        return -1;

    hdr->free_space_end -= len;
    memcpy(p->data + hdr->free_space_end, data, len);

    SlotEntry *slot;
    if (reuse_slot >= 0) {
        slot = page_get_slot(p, (uint16_t)reuse_slot);
        *slot_out = (uint16_t)reuse_slot;
    } else {
        slot = page_get_slot(p, hdr->num_slots);
        *slot_out = hdr->num_slots;
        hdr->num_slots++;
    }

    slot->offset = hdr->free_space_end;
    slot->length = len;
    return 0;
}

int page_get_record(Page *p, uint16_t slot_index, uint8_t **out, uint16_t *len_out) {
    PageHeader *hdr = page_get_header(p);
    if (slot_index >= hdr->num_slots)
        return -1;

    SlotEntry *s = page_get_slot(p, slot_index);
    if (s->offset == 0 && s->length == 0)
        return -1;

    *out = p->data + s->offset;
    *len_out = s->length;
    return 0;
}

int page_delete_record(Page *p, uint16_t slot_index) {
    PageHeader *hdr = page_get_header(p);
    if (slot_index >= hdr->num_slots)
        return -1;

    SlotEntry *s = page_get_slot(p, slot_index);
    if (s->offset == 0 && s->length == 0)
        return -1;

    s->offset = 0;
    s->length = 0;
    return 0;
}

void page_compact(Page *p) {
    PageHeader *hdr = page_get_header(p);
    if (hdr->num_slots == 0)
        return;

    uint16_t write_pos = PAGE_SIZE;

    for (uint16_t i = 0; i < hdr->num_slots; i++) {
        SlotEntry *s = page_get_slot(p, i);
        if (s->offset == 0 && s->length == 0)
            continue;

        write_pos -= s->length;
        if (write_pos != s->offset) {
            memmove(p->data + write_pos, p->data + s->offset, s->length);
            s->offset = write_pos;
        }
    }

    hdr->free_space_end = write_pos;
}

int page_update_record(Page *p, uint16_t slot_index, const void *data, uint16_t len) {
    PageHeader *hdr = page_get_header(p);
    if (slot_index >= hdr->num_slots)
        return -1;

    SlotEntry *s = page_get_slot(p, slot_index);
    if (s->offset == 0 && s->length == 0)
        return -1;

    if (len == s->length) {
        memcpy(p->data + s->offset, data, len);
        return 0;
    }

    /* Verify the new data will fit even after reclaiming old record's space */
    uint16_t old_len = s->length;
    if (page_free_space(p) + old_len < len)
        return -1;

    /* Tombstone old record to reclaim space */
    s->offset = 0;
    s->length = 0;

    if (page_free_space(p) < len)
        page_compact(p);

    hdr->free_space_end -= len;
    memcpy(p->data + hdr->free_space_end, data, len);
    s->offset = hdr->free_space_end;
    s->length = len;

    return 0;
}
