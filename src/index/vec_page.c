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
}

uint16_t vec_page_capacity(uint16_t dim) {
    if (dim == 0)
        return 0;

    uint16_t usable = PAGE_SIZE - VEC_DATA_OFFSET;
    uint16_t vec_bytes = dim * (uint16_t)sizeof(float);
    return usable / vec_bytes;
}

int vec_page_append(Page *p, const float *vec) {
    VecPageHeader *vh = get_vec_header(p);
    uint16_t cap = vec_page_capacity(vh->dim);

    if (vh->num_vectors >= cap)
        return -1;

    uint16_t vec_bytes = vh->dim * (uint16_t)sizeof(float);
    uint16_t offset = VEC_DATA_OFFSET + vh->num_vectors * vec_bytes;

    memcpy(p->data + offset, vec, vec_bytes);
    vh->num_vectors++;
    return 0;
}

const float *vec_page_get(const Page *p, uint16_t index) {
    const VecPageHeader *vh = get_vec_header_const(p);

    if (index >= vh->num_vectors)
        return NULL;

    uint16_t vec_bytes = vh->dim * (uint16_t)sizeof(float);
    uint16_t offset = VEC_DATA_OFFSET + index * vec_bytes;

    return (const float *)(p->data + offset);
}

uint16_t vec_page_count(const Page *p) {
    return get_vec_header_const(p)->num_vectors;
}

uint16_t vec_page_dim(const Page *p) {
    return get_vec_header_const(p)->dim;
}
