#include "vec_page.h"
#include <string.h>

static VecPageHeader *get_vec_header(Page *p) {
    return (VecPageHeader *)(p->data + VEC_PAGE_HEADER_OFFSET);
}

static const VecPageHeader *get_vec_header_const(const Page *p) {
    return (const VecPageHeader *)(p->data + VEC_PAGE_HEADER_OFFSET);
}

void vec_page_init(Page *p, page_id_t id, uint16_t dim) {
    page_init(p, id, PAGE_TYPE_VECTOR);

    VecPageHeader *vh = get_vec_header(p);
    vh->dim         = dim;
    vh->num_vectors = 0;
    vh->_reserved   = 0;
    memset(vh->deleted_bitmap, 0, sizeof(vh->deleted_bitmap));
}

uint16_t vec_page_capacity(uint16_t dim) {
    if (dim == 0)
        return 0;

    uint16_t usable = PAGE_SIZE - VEC_DATA_OFFSET;
    uint32_t vec_bytes = (uint32_t)dim * (uint32_t)sizeof(float);
    if (vec_bytes > usable)
        return 0;
    
    uint16_t cap = (uint16_t)(usable / vec_bytes);
    return (cap > 1024) ? 1024 : cap;
}

int vec_page_append(Page *p, const float *vec) {
    VecPageHeader *vh = get_vec_header(p);
    uint16_t cap = vec_page_capacity(vh->dim);

    if (vh->num_vectors >= cap)
        return -1;

    uint32_t vec_bytes = (uint32_t)vh->dim * (uint32_t)sizeof(float);
    uint32_t offset = VEC_DATA_OFFSET + (uint32_t)vh->num_vectors * vec_bytes;

    memcpy(p->data + offset, vec, vec_bytes);
    vh->num_vectors++;
    return 0;
}

const float *vec_page_get(const Page *p, uint16_t index) {
    const VecPageHeader *vh = get_vec_header_const(p);

    if (index >= vh->num_vectors)
        return NULL;

    uint32_t vec_bytes = (uint32_t)vh->dim * (uint32_t)sizeof(float);
    uint32_t offset = VEC_DATA_OFFSET + (uint32_t)index * vec_bytes;

    return (const float *)(p->data + offset);
}

uint16_t vec_page_count(const Page *p) {
    return get_vec_header_const(p)->num_vectors;
}

uint16_t vec_page_dim(const Page *p) {
    return get_vec_header_const(p)->dim;
}

int vec_page_delete(Page *p, uint16_t index) {
    VecPageHeader *vh = get_vec_header(p);
    if (index >= vh->num_vectors || index >= 1024) {
        return -1;
    }
    
    uint16_t word_idx = index / 64;
    uint16_t bit_idx = index % 64;
    
    if (vh->deleted_bitmap[word_idx] & (1ULL << bit_idx)) {
        return -1;
    }
    
    vh->deleted_bitmap[word_idx] |= (1ULL << bit_idx);
    return 0;
}

bool vec_page_is_deleted(const Page *p, uint16_t index) {
    const VecPageHeader *vh = get_vec_header_const(p);
    if (index >= vh->num_vectors || index >= 1024) {
        return true; 
    }
    
    uint16_t word_idx = index / 64;
    uint16_t bit_idx = index % 64;
    
    return (vh->deleted_bitmap[word_idx] & (1ULL << bit_idx)) != 0;
}

uint16_t vec_page_live_count(const Page *p) {
    const VecPageHeader *vh = get_vec_header_const(p);
    uint16_t live = 0;
    for (uint16_t i = 0; i < vh->num_vectors; i++) {
        uint16_t word_idx = i / 64;
        uint16_t bit_idx = i % 64;
        if (!(vh->deleted_bitmap[word_idx] & (1ULL << bit_idx)))
            live++;
    }
    return live;
}

uint16_t vec_page_compact(Page *p) {
    VecPageHeader *vh = get_vec_header(p);
    uint16_t dim = vh->dim;
    uint32_t vec_bytes = (uint32_t)dim * (uint32_t)sizeof(float);

    if (vec_bytes == 0 || vh->num_vectors == 0)
        return 0;

    uint16_t write_idx = 0;
    uint16_t original_count = vh->num_vectors;

    for (uint16_t read_idx = 0; read_idx < original_count; read_idx++) {
        uint16_t word_idx = read_idx / 64;
        uint16_t bit_idx = read_idx % 64;

        if (vh->deleted_bitmap[word_idx] & (1ULL << bit_idx))
            continue; /* skip deleted */

        if (write_idx != read_idx) {
            uint32_t dst_off = VEC_DATA_OFFSET + (uint32_t)write_idx * vec_bytes;
            uint32_t src_off = VEC_DATA_OFFSET + (uint32_t)read_idx * vec_bytes;
            memmove(p->data + dst_off, p->data + src_off, vec_bytes);
        }
        write_idx++;
    }

    uint16_t reclaimed = original_count - write_idx;

    vh->num_vectors = write_idx;
    memset(vh->deleted_bitmap, 0, sizeof(vh->deleted_bitmap));

    return reclaimed;
}

