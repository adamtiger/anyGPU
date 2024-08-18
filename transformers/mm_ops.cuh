#ifndef __MM_OPS_CUH__
#define __MM_OPS_CUH__

#include "tensor.hpp"

void tensor_mm_f32(
	const Tensor<float32, CUDA>& lhs,
	const Tensor<float32, CUDA>& rhs,
	const Tensor<float32, CUDA>& out);

#endif  // __MM_OPS_CUH__
