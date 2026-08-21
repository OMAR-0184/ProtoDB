#ifndef STREAM_PAGE_H
#define STREAM_PAGE_H

#include "page.h"
#include "../buffer/buffer_pool.h"

/*
 * StreamPage is a utility for writing and reading large contiguous blocks 
 * of data across multiple buffer pool pages. This is used primarily for 
 * serializing metadata (like large arrays of centroids or page IDs).
 *
 * It uses a linked-list of pages, each with a StreamPageHeader.
 */

#define PAGE_TYPE_STREAM 6

typedef struct {
    page_id_t next_page_id; /* INVALID_PAGE_ID if last page */
    uint16_t  data_size;    /* amount of valid data on this specific page */
    uint16_t  _reserved;
} StreamPageHeader;

#define STREAM_PAGE_HEADER_OFFSET PAGE_DATA_OFFSET
#define STREAM_DATA_OFFSET        (STREAM_PAGE_HEADER_OFFSET + (uint16_t)sizeof(StreamPageHeader))
#define STREAM_MAX_DATA_PER_PAGE  (PAGE_SIZE - STREAM_DATA_OFFSET)

/*
 * Write an arbitrary byte buffer into one or more stream pages.
 * Allocates new pages using the buffer pool.
 * Returns 0 on success, and sets *out_start_pid to the first page ID.
 * Returns -1 on failure.
 */
int stream_write(BufferPool *bp, const void *data, size_t size, page_id_t *out_start_pid);

/*
 * Read a byte buffer from a linked list of stream pages.
 * Follows next_page_id until 'size' bytes are read or end of stream is reached.
 * Returns 0 on success, -1 on failure (e.g., unexpected EOF).
 */
int stream_read(BufferPool *bp, page_id_t start_pid, void *data, size_t size);

#endif
