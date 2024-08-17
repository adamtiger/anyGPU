#ifndef __BINARY_OPS_CUH__
#define __BINARY_OPS_CUH__

#include "cuda_runtime.h"

void tensor_add_f32(const dim3 gs, const dim3 bs, const int length, const float* dlhs, const float* drhs, float* dout);
void tensor_add_i32(const dim3 gs, const dim3 bs, const int length, const int* dlhs, const int* drhs, int* dout);


#endif  // __BINARY_OPS_CUH__
