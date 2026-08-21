CC      = gcc
CFLAGS  = -Wall -Wextra -Werror -std=c11 -g -O2
LDFLAGS = -lpthread -lm
TESTDIR = tests
BUILDDIR = build

# Sources
PAGE_SRC       = src/pages/page.c
STORAGE_SRC    = src/pages/storage_mgr.c
STREAM_SRC     = src/pages/stream_page.c
BUFFER_SRC     = src/buffer/buffer_pool.c
VEC_UTILS_SRC  = src/index/vec_utils.c
VEC_TOPK_SRC   = src/index/vec_topk.c
VEC_PAGE_SRC   = src/index/vec_page.c
FLAT_INDEX_SRC = src/index/flat/flat_index.c
IVF_INDEX_SRC  = src/index/ivf/ivf_index.c

OBJS = $(BUILDDIR)/page.o $(BUILDDIR)/storage_mgr.o $(BUILDDIR)/stream_page.o $(BUILDDIR)/buffer_pool.o \
       $(BUILDDIR)/vec_utils.o $(BUILDDIR)/vec_topk.o $(BUILDDIR)/vec_page.o \
       $(BUILDDIR)/flat_index.o $(BUILDDIR)/ivf_index.o

.PHONY: all clean test bench

all: $(OBJS)

$(BUILDDIR)/page.o: $(PAGE_SRC) src/pages/page.h
	@mkdir -p $(BUILDDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILDDIR)/storage_mgr.o: $(STORAGE_SRC) src/pages/storage_mgr.h src/pages/page.h
	@mkdir -p $(BUILDDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILDDIR)/stream_page.o: $(STREAM_SRC) src/pages/stream_page.h src/pages/page.h
	@mkdir -p $(BUILDDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILDDIR)/buffer_pool.o: $(BUFFER_SRC) src/buffer/buffer_pool.h src/pages/storage_mgr.h src/pages/page.h
	@mkdir -p $(BUILDDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILDDIR)/vec_utils.o: $(VEC_UTILS_SRC) src/index/vec_utils.h
	@mkdir -p $(BUILDDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILDDIR)/vec_topk.o: $(VEC_TOPK_SRC) src/index/vec_topk.h src/pages/page.h
	@mkdir -p $(BUILDDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILDDIR)/vec_page.o: $(VEC_PAGE_SRC) src/index/vec_page.h src/pages/page.h
	@mkdir -p $(BUILDDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILDDIR)/flat_index.o: $(FLAT_INDEX_SRC) src/index/flat/flat_index.h src/index/vec_page.h src/index/vec_utils.h src/index/vec_topk.h src/buffer/buffer_pool.h
	@mkdir -p $(BUILDDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILDDIR)/ivf_index.o: $(IVF_INDEX_SRC) src/index/ivf/ivf_index.h src/index/flat/flat_index.h src/index/vec_page.h src/index/vec_utils.h src/index/vec_topk.h src/buffer/buffer_pool.h
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

$(BUILDDIR)/test_flat_index: $(TESTDIR)/test_flat_index.c $(OBJS)
	$(CC) $(CFLAGS) $< $(OBJS) -o $@ $(LDFLAGS)

$(BUILDDIR)/test_ivf_index: $(TESTDIR)/test_ivf_index.c $(OBJS)
	$(CC) $(CFLAGS) $< $(OBJS) -o $@ $(LDFLAGS)

$(BUILDDIR)/test_persistence: $(TESTDIR)/test_persistence.c $(OBJS)
	$(CC) $(CFLAGS) $< $(OBJS) -o $@ $(LDFLAGS)

TEST_BINS = $(BUILDDIR)/test_pages $(BUILDDIR)/test_storage $(BUILDDIR)/test_storage_bulk \
            $(BUILDDIR)/test_buffer_pool $(BUILDDIR)/test_flat_index $(BUILDDIR)/test_ivf_index \
            $(BUILDDIR)/test_persistence

test: $(TEST_BINS)
	@echo ""
	@$(BUILDDIR)/test_pages
	@$(BUILDDIR)/test_storage
	@$(BUILDDIR)/test_storage_bulk
	@$(BUILDDIR)/test_buffer_pool
	@$(BUILDDIR)/test_flat_index
	@$(BUILDDIR)/test_ivf_index
	@$(BUILDDIR)/test_persistence

bench: $(OBJS) bench/bench_main.c
	@mkdir -p bench/results
	$(CC) $(CFLAGS) bench/bench_main.c $(OBJS) -o $(BUILDDIR)/bench_protodb $(LDFLAGS)
	@./$(BUILDDIR)/bench_protodb

clean:
	rm -rf $(BUILDDIR)
