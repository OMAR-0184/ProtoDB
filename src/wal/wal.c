#include "wal.h"
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>

/* ---- CRC32 (ISO 3309 / ITU-T V.42) ---- */

static const uint32_t crc32_table[256] = {
    0x00000000, 0x77073096, 0xEE0E612C, 0x990951BA, 0x076DC419, 0x706AF48F,
    0xE963A535, 0x9E6495A3, 0x0EDB8832, 0x79DCB8A4, 0xE0D5E91B, 0x97D2D988,
    0x09B64C2B, 0x7EB17CBB, 0xE7B82D09, 0x90BF1D91, 0x1DB71064, 0x6AB020F2,
    0xF3B97148, 0x84BE41DE, 0x1ADAD47D, 0x6DDDE4EB, 0xF4D4B551, 0x83D385C7,
    0x136C9856, 0x646BA8C0, 0xFD62F97A, 0x8A65C9EC, 0x14015C4F, 0x63066CD9,
    0xFA0F3D63, 0x8D080DF5, 0x3B6E20C8, 0x4C69105E, 0xD56041E4, 0xA2677172,
    0x3C03E4D5, 0x4B04D447, 0xD20D85FD, 0xA50AB56B, 0x35B5A8FA, 0x42B2986C,
    0xDBBBB9D6, 0xACBCB9C0, 0x32D86CE3, 0x45DF5C75, 0xDCD60DCF, 0xABD13D59,
    0x26D930AC, 0x51DE003A, 0xC8D75180, 0xBFD06116, 0x21B4F0B5, 0x56B3C423,
    0xCFBA9599, 0xB8BDA50F, 0x2802B89E, 0x5F058808, 0xC60CD9B2, 0xB10BE924,
    0x2F6F7C87, 0x58684C11, 0xC1611DAB, 0xB6662D3D, 0x76DC4190, 0x01DB7106,
    0x98D220BC, 0xEFD5102A, 0x71B18589, 0x06B6B51F, 0x9FBFE4A5, 0xE8B8D433,
    0x7807C9A2, 0x0F00F934, 0x9609A88E, 0xE10E9818, 0x7F6A0D6B, 0x086D3D2D,
    0x91646C97, 0xE6635C01, 0x6B6B51F4, 0x1C6C6162, 0x856530D8, 0xF262004E,
    0x6C0695ED, 0x1B01A57B, 0x8208F4C1, 0xF50FC457, 0x65B0D9C6, 0x12B7E950,
    0x8BBEB8EA, 0xFCB9887C, 0x62DD1DDF, 0x15DA2D49, 0x8CD37CF3, 0xFBD44C65,
    0x4DB26158, 0x3AB551CE, 0xA3BC0074, 0xD4BB30E2, 0x4ADFA541, 0x3DD895D7,
    0xA4D1C46D, 0xD3D6F4FB, 0x4369E96A, 0x346ED9FC, 0xAD678846, 0xDA60B8D0,
    0x44042D73, 0x33031DE5, 0xAA0A4C5F, 0xDD0D7822, 0x3B6E20C8, 0x4C69105E,
    0xD56041E4, 0xA2677172, 0x3C03E4D5, 0x4B04D447, 0xD20D85FD, 0xA50AB56B,
    0x35B5A8FA, 0x42B2986C, 0xDBBBB9D6, 0xACBCB9C0, 0x32D86CE3, 0x45DF5C75,
    0xDCD60DCF, 0xABD13D59, 0x26D930AC, 0x51DE003A, 0xC8D75180, 0xBFD06116,
    0x21B4F0B5, 0x56B3C423, 0xCFBA9599, 0xB8BDA50F, 0x2802B89E, 0x5F058808,
    0xC60CD9B2, 0xB10BE924, 0x2F6F7C87, 0x58684C11, 0xC1611DAB, 0xB6662D3D,
    0x76DC4190, 0x01DB7106, 0x98D220BC, 0xEFD5102A, 0x71B18589, 0x06B6B51F,
    0x9FBFE4A5, 0xE8B8D433, 0x7807C9A2, 0x0F00F934, 0x9609A88E, 0xE10E9818,
    0x7F6A0D6B, 0x086D3D2D, 0x91646C97, 0xE6635C01, 0x6B6B51F4, 0x1C6C6162,
    0x856530D8, 0xF262004E, 0x6C0695ED, 0x1B01A57B, 0x8208F4C1, 0xF50FC457,
    0x65B0D9C6, 0x12B7E950, 0x8BBEB8EA, 0xFCB9887C, 0x62DD1DDF, 0x15DA2D49,
    0x8CD37CF3, 0xFBD44C65, 0x4DB26158, 0x3AB551CE, 0xA3BC0074, 0xD4BB30E2,
    0x4ADFA541, 0x3DD895D7, 0xA4D1C46D, 0xD3D6F4FB, 0x4369E96A, 0x346ED9FC,
    0xAD678846, 0xDA60B8D0, 0x44042D73, 0x33031DE5, 0xAA0A4C5F, 0xDD0D7822,
    0x9B64C2B0, 0xEC63F226, 0x756AA39C, 0x026D930A, 0x9C0906A9, 0xEB0E363F,
    0x72076785, 0x05005713, 0x95BF4A82, 0xE2B87A14, 0x7BB12BAE, 0x0CB61B38,
    0x92D28E9B, 0xE5D5BE0D, 0x7CDCEFB7, 0x0BDBDF21, 0x86D3D2D4, 0xF1D4E242,
    0x68DDB3F6, 0x1FDA836E, 0x81BE16CD, 0xF6B9265B, 0x6FB077E1, 0x18B74777,
    0x88085AE6, 0xFF0F6B70, 0x66063BCA, 0x11010B5C, 0x8F659EFF, 0xF862AE69,
    0x616BFFD3, 0x166CCF45, 0xA00AE278, 0xD70DD2EE, 0x4E048354, 0x3903B3C2,
    0xA7672661, 0xD06016F7, 0x4969474D, 0x3E6E77DB, 0xAED16A4A, 0xD9D65ADC,
    0x40DF0B66, 0x37D83BF0, 0xA9BCAE53, 0xDEBB9EC5, 0x47B2CF7F, 0x30B5FFE9,
    0xBDBDF21C, 0xCABAC28A, 0x53B39330, 0x24B4A3A6, 0xBAD03605, 0xCDD706FF,
    0x54DE5729, 0x23D967BF, 0xB3667A2E, 0xC4614AB8, 0x5D681B02, 0x2A6F2B94,
    0xB40BBE37, 0xC30C8EA1, 0x5A05DF1B, 0x2D02EF8D,
};

uint32_t wal_crc32(const uint8_t *data, size_t len) {
    uint32_t crc = 0xFFFFFFFF;
    for (size_t i = 0; i < len; i++) {
        crc = (crc >> 8) ^ crc32_table[(crc ^ data[i]) & 0xFF];
    }
    return crc ^ 0xFFFFFFFF;
}

/* ---- WAL file operations ---- */

static int wal_flush_header(Wal *wal) {
    WalHeader hdr = {
        .magic       = WAL_MAGIC,
        .version     = WAL_VERSION,
        .num_records = wal->num_records,
        ._reserved   = 0,
    };

    if (pwrite(wal->fd, &hdr, sizeof(WalHeader), 0) != sizeof(WalHeader))
        return -1;

    return 0;
}

static int wal_read_header(Wal *wal, WalHeader *hdr) {
    if (pread(wal->fd, hdr, sizeof(WalHeader), 0) != sizeof(WalHeader))
        return -1;
    return 0;
}

int wal_open(Wal *wal, const char *db_path, StorageManager *sm) {
    memset(wal, 0, sizeof(Wal));
    wal->fd = -1;

    /* Construct WAL path: <db_path>.wal */
    size_t path_len = strlen(db_path);
    if (path_len + 5 >= sizeof(wal->filepath)) {
        fprintf(stderr, "wal_open: path too long\n");
        return -1;
    }
    snprintf(wal->filepath, sizeof(wal->filepath), "%s.wal", db_path);

    int fd = open(wal->filepath, O_RDWR | O_CREAT, 0644);
    if (fd < 0) {
        perror("wal_open");
        return -1;
    }
    wal->fd = fd;

    struct stat st;
    if (fstat(fd, &st) < 0) {
        perror("wal_open: fstat");
        close(fd);
        wal->fd = -1;
        return -1;
    }

    if (st.st_size == 0) {
        /* New WAL file — write initial header */
        wal->num_records = 0;
        wal->next_lsn = 1;

        if (wal_flush_header(wal) < 0) {
            close(fd);
            wal->fd = -1;
            return -1;
        }
        fsync(fd);
    } else {
        /* Existing WAL — read header and attempt recovery */
        WalHeader hdr;
        if (wal_read_header(wal, &hdr) < 0 || hdr.magic != WAL_MAGIC) {
            fprintf(stderr, "wal_open: invalid WAL header, resetting\n");
            wal->num_records = 0;
            wal->next_lsn = 1;
            if (ftruncate(fd, 0) < 0) {
                close(fd);
                wal->fd = -1;
                return -1;
            }
            if (wal_flush_header(wal) < 0) {
                close(fd);
                wal->fd = -1;
                return -1;
            }
            fsync(fd);
        } else {
            wal->num_records = hdr.num_records;
            wal->next_lsn = hdr.num_records + 1;

            /* If there are records, run recovery */
            if (wal->num_records > 0) {
                if (wal_recover(wal, sm) < 0) {
                    close(fd);
                    wal->fd = -1;
                    return -1;
                }
            }
        }
    }

    return 0;
}

int wal_close(Wal *wal) {
    if (wal->fd < 0)
        return -1;

    fsync(wal->fd);
    close(wal->fd);
    wal->fd = -1;
    return 0;
}

int wal_log_page(Wal *wal, page_id_t page_id, const uint8_t *page_data) {
    if (wal->fd < 0 || !page_data)
        return -1;

    WalRecord rec;
    memset(&rec, 0, sizeof(WalRecord));
    rec.lsn = wal->next_lsn;
    rec.page_id = page_id;
    memcpy(rec.data, page_data, PAGE_SIZE);
    rec.checksum = wal_crc32(rec.data, PAGE_SIZE);

    /* Calculate offset: header + existing records */
    off_t offset = (off_t)sizeof(WalHeader) +
                   (off_t)wal->num_records * (off_t)sizeof(WalRecord);

    ssize_t n = pwrite(wal->fd, &rec, sizeof(WalRecord), offset);
    if (n != (ssize_t)sizeof(WalRecord))
        return -1;

    /* Ensure the record is durable before proceeding */
    if (fsync(wal->fd) < 0)
        return -1;

    wal->num_records++;
    wal->next_lsn++;

    /* Update the header record count */
    if (wal_flush_header(wal) < 0)
        return -1;

    if (fsync(wal->fd) < 0)
        return -1;

    return 0;
}

int wal_recover(Wal *wal, StorageManager *sm) {
    if (wal->fd < 0 || wal->num_records == 0)
        return 0;

    printf("WAL: recovering %u records...\n", wal->num_records);

    uint32_t applied = 0;
    uint32_t skipped = 0;

    for (uint32_t i = 0; i < wal->num_records; i++) {
        WalRecord rec;
        off_t offset = (off_t)sizeof(WalHeader) +
                       (off_t)i * (off_t)sizeof(WalRecord);

        ssize_t n = pread(wal->fd, &rec, sizeof(WalRecord), offset);
        if (n != (ssize_t)sizeof(WalRecord)) {
            fprintf(stderr, "WAL: truncated record %u, stopping recovery\n", i);
            break;
        }

        /* Verify checksum */
        uint32_t computed_crc = wal_crc32(rec.data, PAGE_SIZE);
        if (computed_crc != rec.checksum) {
            fprintf(stderr, "WAL: checksum mismatch on record %u "
                    "(expected 0x%08X, got 0x%08X), skipping\n",
                    i, rec.checksum, computed_crc);
            skipped++;
            continue;
        }

        /* Ensure the database is large enough for this page */
        if (rec.page_id >= sm->meta.page_count) {
            /* Extend the file */
            off_t new_size = (off_t)(rec.page_id + 1) * PAGE_SIZE;
            if (ftruncate(sm->fd, new_size) < 0) {
                perror("WAL: ftruncate during recovery");
                return -1;
            }
            sm->meta.page_count = rec.page_id + 1;
        }

        /* Write the page image directly to the database file */
        Page tmp;
        memcpy(tmp.data, rec.data, PAGE_SIZE);
        tmp.id = rec.page_id;
        if (storage_write_page(sm, rec.page_id, &tmp) < 0) {
            fprintf(stderr, "WAL: failed to write page %u during recovery\n",
                    rec.page_id);
            return -1;
        }

        applied++;
    }

    printf("WAL: recovery complete — %u applied, %u skipped\n",
           applied, skipped);

    /* Checkpoint: truncate the WAL after successful recovery */
    wal->num_records = 0;
    wal->next_lsn = applied + skipped + 1;

    if (ftruncate(wal->fd, sizeof(WalHeader)) < 0) {
        perror("WAL: ftruncate after recovery");
        return -1;
    }

    if (wal_flush_header(wal) < 0)
        return -1;

    fsync(wal->fd);
    return 0;
}

int wal_checkpoint(Wal *wal, StorageManager *sm) {
    if (wal->fd < 0)
        return -1;

    if (wal->num_records == 0)
        return 0;

    /* Replay all records into the database */
    for (uint32_t i = 0; i < wal->num_records; i++) {
        WalRecord rec;
        off_t offset = (off_t)sizeof(WalHeader) +
                       (off_t)i * (off_t)sizeof(WalRecord);

        ssize_t n = pread(wal->fd, &rec, sizeof(WalRecord), offset);
        if (n != (ssize_t)sizeof(WalRecord))
            return -1;

        uint32_t computed_crc = wal_crc32(rec.data, PAGE_SIZE);
        if (computed_crc != rec.checksum) {
            fprintf(stderr, "WAL checkpoint: checksum mismatch on record %u\n", i);
            return -1;
        }

        if (rec.page_id >= sm->meta.page_count) {
            off_t new_size = (off_t)(rec.page_id + 1) * PAGE_SIZE;
            if (ftruncate(sm->fd, new_size) < 0)
                return -1;
            sm->meta.page_count = rec.page_id + 1;
        }

        Page tmp;
        memcpy(tmp.data, rec.data, PAGE_SIZE);
        tmp.id = rec.page_id;
        if (storage_write_page(sm, rec.page_id, &tmp) < 0)
            return -1;
    }

    /* Sync the database file */
    fsync(sm->fd);

    /* Truncate the WAL */
    wal->num_records = 0;

    if (ftruncate(wal->fd, sizeof(WalHeader)) < 0)
        return -1;

    if (wal_flush_header(wal) < 0)
        return -1;

    fsync(wal->fd);
    return 0;
}
