CC      = gcc
CFLAGS  = -Wall -Wextra -Werror -std=c11 -g -O2
LDFLAGS = -lpthread
TESTDIR = tests
BUILDDIR = build

# Sources
PAGE_SRC    = src/pages/page.c
STORAGE_SRC = src/pages/storage_mgr.c
BUFFER_SRC  = src/buffer/buffer_pool.c

OBJS = $(BUILDDIR)/page.o $(BUILDDIR)/storage_mgr.o $(BUILDDIR)/buffer_pool.o

.PHONY: all clean test

all: $(OBJS)

$(BUILDDIR)/page.o: $(PAGE_SRC) src/pages/page.h
	@mkdir -p $(BUILDDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILDDIR)/storage_mgr.o: $(STORAGE_SRC) src/pages/storage_mgr.h src/pages/page.h
	@mkdir -p $(BUILDDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILDDIR)/buffer_pool.o: $(BUFFER_SRC) src/buffer/buffer_pool.h src/pages/storage_mgr.h src/pages/page.h
	@mkdir -p $(BUILDDIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Tests
$(BUILDDIR)/test_pages: $(TESTDIR)/test_pages.c $(OBJS)
	$(CC) $(CFLAGS) $< $(OBJS) -o $@ $(LDFLAGS)

$(BUILDDIR)/test_storage: $(TESTDIR)/test_storage.c $(OBJS)
	$(CC) $(CFLAGS) $< $(OBJS) -o $@ $(LDFLAGS)

$(BUILDDIR)/test_storage_bulk: $(TESTDIR)/test_storage_bulk.c $(OBJS)
	$(CC) $(CFLAGS) $< $(OBJS) -o $@ $(LDFLAGS)

$(BUILDDIR)/test_buffer_pool: $(TESTDIR)/test_buffer_pool.c $(OBJS)
	$(CC) $(CFLAGS) $< $(OBJS) -o $@ $(LDFLAGS)

test: $(BUILDDIR)/test_pages $(BUILDDIR)/test_storage $(BUILDDIR)/test_storage_bulk $(BUILDDIR)/test_buffer_pool
	@echo ""
	@$(BUILDDIR)/test_pages
	@$(BUILDDIR)/test_storage
	@$(BUILDDIR)/test_storage_bulk
	@$(BUILDDIR)/test_buffer_pool

clean:
	rm -rf $(BUILDDIR)
