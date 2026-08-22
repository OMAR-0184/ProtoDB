#include "storage_mgr.h"
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <stdio.h>
#include <sys/stat.h>
#include <errno.h>
#include <stdlib.h>

static int flush_meta(StorageManager *sm) {
    uint8_t buf[PAGE_SIZE];
    memset(buf, 0, PAGE_SIZE);

    PageHeader ph = {
        .page_id        = META_PAGE_ID,
        .num_slots      = 0,
        .free_space_end = PAGE_SIZE,
        .page_type      = PAGE_TYPE_META,
    };
    memcpy(buf, &ph, sizeof(PageHeader));
    memcpy(buf + PAGE_DATA_OFFSET, &sm->meta, sizeof(MetaPageHeader));

    ssize_t n = pwrite(sm->fd, buf, PAGE_SIZE, 0);
    if (n != PAGE_SIZE)
        return -1;

    return 0;
}

static int load_meta(StorageManager *sm) {
    uint8_t buf[PAGE_SIZE];

    ssize_t n = pread(sm->fd, buf, PAGE_SIZE, 0);
    if (n != PAGE_SIZE)
        return -1;

    memcpy(&sm->meta, buf + PAGE_DATA_OFFSET, sizeof(MetaPageHeader));

    if (sm->meta.magic != PROTO_MAGIC) {
        fprintf(stderr, "storage_open: invalid magic 0x%X (expected 0x%X)\n",
                sm->meta.magic, PROTO_MAGIC);
        return -1;
    }

    return 0;
}

int storage_open(StorageManager *sm, const char *path) {
    memset(sm, 0, sizeof(StorageManager));
    sm->fd = -1;

    size_t len = strlen(path);
    if (len >= sizeof(sm->filepath)) {
        fprintf(stderr, "storage_open: path too long\n");
        return -1;
    }
    memcpy(sm->filepath, path, len + 1);

    int fd = open(path, O_RDWR | O_CREAT, 0644);
    if (fd < 0) {
        perror("storage_open");
        return -1;
    }
    sm->fd = fd;

    struct stat st;
    if (fstat(fd, &st) < 0) {
        perror("storage_open: fstat");
        close(fd);
        sm->fd = -1;
        return -1;
    }

    if (st.st_size == 0) {
        sm->meta.magic          = PROTO_MAGIC;
        sm->meta.schema_version = SCHEMA_VERSION;
        sm->meta.page_count     = 1;
        sm->meta.free_list_head = INVALID_PAGE_ID;
        sm->meta.root_index_pid = INVALID_PAGE_ID;
        sm->meta.root_index_type= 0;
        sm->meta.root_meta_store_pid = INVALID_PAGE_ID;

        if (ftruncate(fd, PAGE_SIZE) < 0) {
            perror("storage_open: ftruncate");
            close(fd);
            sm->fd = -1;
            return -1;
        }

        if (flush_meta(sm) < 0) {
            close(fd);
            sm->fd = -1;
            return -1;
        }
    } else {
        if ((uint64_t)st.st_size < PAGE_SIZE || st.st_size % PAGE_SIZE != 0) {
            fprintf(stderr, "storage_open: corrupt file size %lld\n", (long long)st.st_size);
            close(fd);
            sm->fd = -1;
            return -1;
        }

        if (load_meta(sm) < 0) {
            close(fd);
            sm->fd = -1;
            return -1;
        }
    }

    return 0;
}

int storage_close(StorageManager *sm) {
    if (sm->fd < 0)
        return -1;

    if (flush_meta(sm) < 0) {
        close(sm->fd);
        sm->fd = -1;
        return -1;
    }

    fsync(sm->fd);
    close(sm->fd);
    sm->fd = -1;
    return 0;
}

int storage_read_page(StorageManager *sm, page_id_t page_id, Page *p) {
    if (sm->fd < 0 || page_id >= sm->meta.page_count)
        return -1;

    off_t offset = (off_t)page_id * PAGE_SIZE;
    ssize_t n = pread(sm->fd, p->data, PAGE_SIZE, offset);
    if (n != PAGE_SIZE)
        return -1;

    p->id        = page_id;
    p->pin_count = 0;
    p->is_dirty  = false;
    return 0;
}

int storage_write_page(StorageManager *sm, page_id_t page_id, Page *p) {
    if (sm->fd < 0 || page_id >= sm->meta.page_count)
        return -1;

    off_t offset = (off_t)page_id * PAGE_SIZE;
    ssize_t n = pwrite(sm->fd, p->data, PAGE_SIZE, offset);
    if (n != PAGE_SIZE)
        return -1;

    return 0;
}

int storage_allocate_page(StorageManager *sm, page_id_t *out) {
    if (sm->fd < 0)
        return -1;

    if (sm->meta.free_list_head != INVALID_PAGE_ID) {
        page_id_t reuse_id = sm->meta.free_list_head;

        uint8_t buf[PAGE_SIZE];
        off_t offset = (off_t)reuse_id * PAGE_SIZE;
        ssize_t n = pread(sm->fd, buf, PAGE_SIZE, offset);
        if (n != PAGE_SIZE)
            return -1;

        page_id_t next_free;
        memcpy(&next_free, buf + PAGE_DATA_OFFSET, sizeof(page_id_t));
        sm->meta.free_list_head = next_free;

        memset(buf, 0, PAGE_SIZE);
        PageHeader ph = {
            .page_id        = reuse_id,
            .num_slots      = 0,
            .free_space_end = PAGE_SIZE,
            .page_type      = PAGE_TYPE_FREE,
        };
        memcpy(buf, &ph, sizeof(PageHeader));
        n = pwrite(sm->fd, buf, PAGE_SIZE, offset);
        if (n != PAGE_SIZE)
            return -1;

        if (flush_meta(sm) < 0)
            return -1;

        *out = reuse_id;
        return 0;
    }

    page_id_t new_id = sm->meta.page_count;
    off_t new_size = (off_t)(new_id + 1) * PAGE_SIZE;

    if (ftruncate(sm->fd, new_size) < 0) {
        perror("storage_allocate_page: ftruncate");
        return -1;
    }

    uint8_t buf[PAGE_SIZE];
    memset(buf, 0, PAGE_SIZE);
    PageHeader ph = {
        .page_id        = new_id,
        .num_slots      = 0,
        .free_space_end = PAGE_SIZE,
        .page_type      = PAGE_TYPE_FREE,
    };
    memcpy(buf, &ph, sizeof(PageHeader));
    ssize_t n = pwrite(sm->fd, buf, PAGE_SIZE, (off_t)new_id * PAGE_SIZE);
    if (n != PAGE_SIZE)
        return -1;

    sm->meta.page_count = new_id + 1;

    if (flush_meta(sm) < 0)
        return -1;

    *out = new_id;
    return 0;
}

int storage_deallocate_page(StorageManager *sm, page_id_t page_id) {
    if (sm->fd < 0 || page_id == META_PAGE_ID || page_id >= sm->meta.page_count)
        return -1;

    uint8_t buf[PAGE_SIZE];
    memset(buf, 0, PAGE_SIZE);

    PageHeader ph = {
        .page_id        = page_id,
        .num_slots      = 0,
        .free_space_end = PAGE_SIZE,
        .page_type      = PAGE_TYPE_FREE,
    };
    memcpy(buf, &ph, sizeof(PageHeader));

    page_id_t old_head = sm->meta.free_list_head;
    memcpy(buf + PAGE_DATA_OFFSET, &old_head, sizeof(page_id_t));

    off_t offset = (off_t)page_id * PAGE_SIZE;
    ssize_t n = pwrite(sm->fd, buf, PAGE_SIZE, offset);
    if (n != PAGE_SIZE)
        return -1;

    sm->meta.free_list_head = page_id;

    return flush_meta(sm);
}

page_id_t storage_num_pages(StorageManager *sm) {
    return sm->meta.page_count;
}

int storage_read_pages(StorageManager *sm, page_id_t start, uint32_t count, Page *pages) {
    if (sm->fd < 0 || count == 0)
        return -1;

    if (start >= sm->meta.page_count || count > sm->meta.page_count - start)
        return -1;

    /* Single pread for the entire contiguous range */
    size_t total_bytes = (size_t)count * PAGE_SIZE;
    uint8_t *bulk_buf = malloc(total_bytes);
    if (!bulk_buf)
        return -1;

    off_t offset = (off_t)start * PAGE_SIZE;
    ssize_t n = pread(sm->fd, bulk_buf, total_bytes, offset);
    if (n != (ssize_t)total_bytes) {
        free(bulk_buf);
        return -1;
    }

    for (uint32_t i = 0; i < count; i++) {
        memcpy(pages[i].data, bulk_buf + i * PAGE_SIZE, PAGE_SIZE);
        pages[i].id        = start + i;
        pages[i].pin_count = 0;
        pages[i].is_dirty  = false;
    }

    free(bulk_buf);
    return 0;
}
