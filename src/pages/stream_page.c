#include "stream_page.h"
#include <string.h>
#include <stdio.h>

static StreamPageHeader *get_stream_header(Page *p) {
    return (StreamPageHeader *)(p->data + STREAM_PAGE_HEADER_OFFSET);
}

int stream_write(BufferPool *bp, const void *data, size_t size, page_id_t *out_start_pid) {
    if (size == 0) {
        *out_start_pid = INVALID_PAGE_ID;
        return 0;
    }

    const uint8_t *ptr = (const uint8_t *)data;
    size_t remaining = size;
    page_id_t first_pid = INVALID_PAGE_ID;
    page_id_t prev_pid = INVALID_PAGE_ID;

    while (remaining > 0) {
        page_id_t new_pid;
        Page *p = bp_new_page(bp, &new_pid);
        if (!p) {
            /* Rollback: delete all successfully allocated pages so far */
            page_id_t curr = first_pid;
            while (curr != INVALID_PAGE_ID) {
                Page *cp = bp_fetch_page(bp, curr);
                if (cp) {
                    StreamPageHeader *chdr = get_stream_header(cp);
                    page_id_t nxt = chdr->next_page_id;
                    bp_unpin_page(bp, curr, false);
                    bp_delete_page(bp, curr);
                    curr = nxt;
                } else {
                    break;
                }
            }
            return -1;
        }

        page_init(p, new_pid, PAGE_TYPE_STREAM);
        
        if (first_pid == INVALID_PAGE_ID) {
            first_pid = new_pid;
        }

        StreamPageHeader *hdr = get_stream_header(p);
        hdr->next_page_id = INVALID_PAGE_ID;

        size_t to_write = remaining;
        if (to_write > STREAM_MAX_DATA_PER_PAGE) {
            to_write = STREAM_MAX_DATA_PER_PAGE;
        }

        memcpy(p->data + STREAM_DATA_OFFSET, ptr, to_write);
        hdr->data_size = (uint16_t)to_write;

        ptr += to_write;
        remaining -= to_write;

        bp_unpin_page(bp, new_pid, true);

        if (prev_pid != INVALID_PAGE_ID) {
            /* Link previous page to this one */
            Page *prev_p = bp_fetch_page(bp, prev_pid);
            if (prev_p) {
                StreamPageHeader *prev_hdr = get_stream_header(prev_p);
                prev_hdr->next_page_id = new_pid;
                bp_unpin_page(bp, prev_pid, true);
            }
        }
        
        prev_pid = new_pid;
    }

    *out_start_pid = first_pid;
    return 0;
}

int stream_read(BufferPool *bp, page_id_t start_pid, void *data, size_t size) {
    if (size == 0) {
        return 0;
    }

    uint8_t *ptr = (uint8_t *)data;
    size_t remaining = size;
    page_id_t current_pid = start_pid;

    while (remaining > 0) {
        if (current_pid == INVALID_PAGE_ID) {
            /* Hit end of stream before reading requested size */
            return -1;
        }

        Page *p = bp_fetch_page(bp, current_pid);
        if (!p) {
            return -1;
        }

        StreamPageHeader *hdr = get_stream_header(p);
        
        size_t to_read = remaining;
        if (to_read > hdr->data_size) {
            to_read = hdr->data_size;
        }

        memcpy(ptr, p->data + STREAM_DATA_OFFSET, to_read);
        
        ptr += to_read;
        remaining -= to_read;
        current_pid = hdr->next_page_id;

        bp_unpin_page(bp, p->id, false);
    }

    return 0;
}
