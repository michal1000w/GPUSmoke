#if defined(TINYVDBIO_USE_SYSTEM_ZLIB)
// or inclur your own zlib header here.
#include <zlib.h>
#endif

#define TINYVDBIO_IMPLEMENTATION
#define TINYVDBIO_USE_BLOSC
#include "tinyvdbio.h"