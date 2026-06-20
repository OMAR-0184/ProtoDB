/*
 * =============================================================================
 * IMPLEMENTATION NOTES — page.h
 * =============================================================================
 *
 * MEMORY LAYOUT OF data[PAGE_SIZE]
 * ---------------------------------
 * Bytes are organised as follows inside every page:
 *
 *   [0 .............. sizeof(PageHeader)-1]  →  PageHeader (fixed, never moves)
 *   [sizeof(PageHeader) ................. →]  →  slot array grows RIGHT
 *                                    [← .]  →  record data grows LEFT
 *                       (free space lives in the gap between the two)
 *
 *   Slot N entry:  {uint16_t offset, uint16_t length}  stored at known index.
 *   To find record N: read slot[N].offset → jump to data[offset], read length bytes.
 *   To check free space: free_space_end - (PAGE_DATA_OFFSET + num_slots * slot_size)
 *
 *
 * WHAT GOES TO DISK vs WHAT STAYS IN MEMORY
 * ------------------------------------------
 *   Written to disk  →  data[PAGE_SIZE] only. The PageHeader is embedded inside
 *                        it, so it persists automatically on every flush.
 *
 *   Never written    →  id, pin_count, is_dirty. These are buffer pool state.
 *                        They are zeroed/reset each time a frame is reused.
 *
 *
 * PIN COUNT RULES (critical — violating these causes silent data loss)
 * ---------------------------------------------------------------------
 *   fetch_page()     →  pin_count++
 *   unpin_page()     →  pin_count--
 *   evict candidate  →  only frames where pin_count == 0 are eligible
 *   new frame reuse  →  reset pin_count=0, is_dirty=false, id=INVALID_PAGE_ID
 *
 *   If pin_count goes negative, a caller unpinned more than it pinned — bug.
 *   That is why pin_count is signed int, not uint: negative value is detectable.
 *
 *
 * DIRTY FLAG RULES
 * ----------------
 *   Set is_dirty=true  →  caller passes dirty=true to unpin_page()
 *   Never clear early  →  only clear AFTER a successful write_page() to disk
 *   On eviction        →  if is_dirty, flush to disk first, then overwrite frame
 *   Clean frame        →  can be overwritten directly with no disk write
 *
 *
 * PAGEHEADER STABILITY CONTRACT
 * ------------------------------
 *   PageHeader layout must never change once any data has been written to disk.
 *   Adding or reordering fields corrupts all existing database files.
 *   _pad[3] is explicit — do NOT let the compiler add implicit padding by
 *   rearranging fields. If you add a field, update _pad to compensate so the
 *   total struct size stays the same multiple of 4.
 *
 *
 * PAGE TYPES (page_type field)
 * ----------------------------
 *   Define an enum in common.h, e.g.:
 *     PAGE_TYPE_FREE  = 0   →  unallocated, safe to give to allocate_page()
 *     PAGE_TYPE_DATA  = 1   →  heap file record page
 *     PAGE_TYPE_INDEX = 2   →  B-tree node page  (Phase 7+)
 *     PAGE_TYPE_META  = 3   →  page 0 of the file, stores global state
 *
 *
 * WHAT PAGE 0 SHOULD STORE (disk_manager concern, noted here for context)
 * -----------------------------------------------------------------------
 *   Page 0 is special — it is the database file's header page (PAGE_TYPE_META).
 *   It should hold: total page count, next free page ID, schema version, and a
 *   magic number (e.g. 0xDB1234) so you can detect non-database files on open.
 *
 * =============================================================================
 */


#ifndef PAGE_H
#define PAGE_H

#include <stdint.h>
#include <stdbool.h>

#define PAGE_SIZE       4096
#define INVALID_PAGE_ID UINT32_MAX

typedef uint32_t page_id_t;


typedef struct {
    page_id_t  page_id;         
    uint16_t   num_slots;       
    uint16_t   free_space_end;  
    uint8_t    page_type;       
    uint8_t    _pad[3];         
} PageHeader;


typedef struct {
    page_id_t  id;              
    int        pin_count;       
    bool       is_dirty;        
    uint8_t    data[PAGE_SIZE]; 
} Page;


static inline PageHeader *page_get_header(Page *p) {
    return (PageHeader *)p->data;
}

#define PAGE_DATA_OFFSET  sizeof(PageHeader)

#endif