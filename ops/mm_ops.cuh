#ifndef __MM_OPS_CUH__
#define __MM_OPS_CUH__

#include "tensor.hpp"

/*
  Opt0 - the base kernel implementation without
    relevant optimizations.
*/
void tensor_mm_f32_opt0(
	const Tensor<float32, CUDA>& lhs,
	const Tensor<float32, CUDA>& rhs,
	const Tensor<float32, CUDA>& out);


/*
  Opt1 - optimization ideas are:
    shared memory for saving bandwidth;
	coalasced read and write;
  Assumed:
    medium, large matrices (can not fit in shared);
	size is devidable by 32;
	fixed tile size and block size (minimal flexibility);
	specified for rtx3050Ti
*/
void tensor_mm_f32_opt1(
	const Tensor<float32, CUDA>& lhs,
	const Tensor<float32, CUDA>& rhs,
	const Tensor<float32, CUDA>& out);


#endif  // __MM_OPS_CUH__
