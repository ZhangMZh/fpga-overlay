// Only use OCL address spaces if in OCL flow. Otherwise use empty define.
#ifdef TARGET_OCL
#include "HLS/ocl_types.h"
#else
#define OCL_ADDRSP_GLOBAL
#endif

extern "C" float dot_prod(OCL_ADDRSP_GLOBAL float* A, OCL_ADDRSP_GLOBAL float* B, unsigned n);
