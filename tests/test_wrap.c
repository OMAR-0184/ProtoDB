#include <stdio.h>
#include <stdlib.h>
#undef assert
#define assert(x) do { if (!(x)) { printf("ASSERT FAILED at %s:%d\n", __FILE__, __LINE__); exit(1); } } while(0)
#include "tests/test_metadata.c"
