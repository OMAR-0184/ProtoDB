CC      = gcc
CFLAGS  = -Wall -Wextra -Werror -std=c11 -g -O2
SRCDIR  = src/pages
TESTDIR = tests
BUILDDIR = build

SRCS    = $(SRCDIR)/page.c $(SRCDIR)/storage_mgr.c
OBJS    = $(BUILDDIR)/page.o $(BUILDDIR)/storage_mgr.o

.PHONY: all clean test

all: $(OBJS)

$(BUILDDIR)/page.o: $(SRCDIR)/page.c $(SRCDIR)/page.h
	@mkdir -p $(BUILDDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILDDIR)/storage_mgr.o: $(SRCDIR)/storage_mgr.c $(SRCDIR)/storage_mgr.h $(SRCDIR)/page.h
	@mkdir -p $(BUILDDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILDDIR)/test_pages: $(TESTDIR)/test_pages.c $(OBJS)
	$(CC) $(CFLAGS) $< $(OBJS) -o $@

$(BUILDDIR)/test_storage: $(TESTDIR)/test_storage.c $(OBJS)
	$(CC) $(CFLAGS) $< $(OBJS) -o $@

test: $(BUILDDIR)/test_pages $(BUILDDIR)/test_storage
	@echo ""
	@$(BUILDDIR)/test_pages
	@$(BUILDDIR)/test_storage

clean:
	rm -rf $(BUILDDIR)
