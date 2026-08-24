#ifndef WAL_H
#define WAL_H

#include "../pages/page.h"
#include "../pages/storage_mgr.h"
#include <stdint.h>

/*
 * Write-Ahead Log (WAL) for ProtoDB.
 *
 * Provides crash recovery by logging full page images before they are
 * written to the main database file. On startup, any committed WAL
 * records that weren't checkpointed are replayed into the database.
 *
 * WAL file format:
 *   [WalHeader]          — 16 bytes (file header with magic + version)
 *   [WalRecord 0]        — sizeof(WalRecord) bytes
 *   [WalRecord 1]        — sizeof(WalRecord) bytes
 *   ...
 *
 * Each WalRecord contains a full page image with a CRC32 checksum
 * for corruption detection.
 */

#define WAL_MAGIC 0x57414C31 /* "WAL1" */
#define WAL_VERSION 1

typedef struct {
  uint32_t magic;
  uint32_t version;
  uint32_t num_records; /* number of valid records in the WAL */
  uint32_t _reserved;
} WalHeader;

typedef struct {
  uint32_t lsn;      /* Log Sequence Number */
  page_id_t page_id; /* Which page was modified */
  uint32_t checksum; /* CRC32 of the page data */
  uint32_t _reserved;
  uint8_t data[PAGE_SIZE]; /* Full page image */
} WalRecord;

typedef struct {
  int fd;               /* WAL file descriptor */
  uint32_t next_lsn;    /* Next LSN to assign */
  uint32_t num_records; /* Number of records in current WAL */
  char filepath[264];   /* Path to .wal file */
} Wal;

/* Open or create the WAL file. Automatically runs recovery if needed. */
int wal_open(Wal *wal, const char *db_path, StorageManager *sm);

/* Close the WAL file. Does NOT checkpoint first — caller must do that. */
int wal_close(Wal *wal);

/* Log a full page image before writing it to the database. */
int wal_log_page(Wal *wal, page_id_t page_id, const uint8_t *page_data);

/*
 * Checkpoint: replay all WAL records into the database, then truncate
 * the WAL. After this, the WAL is empty and all changes are durable
 * in the main database file.
 */
int wal_checkpoint(Wal *wal, StorageManager *sm);

/*
 * Recovery: read all valid WAL records and replay them into the database.
 * Called automatically by wal_open if the WAL contains uncommitted records.
 */
int wal_recover(Wal *wal, StorageManager *sm);

/* Compute CRC32 checksum for data integrity verification. */
uint32_t wal_crc32(const uint8_t *data, size_t len);

#endif
