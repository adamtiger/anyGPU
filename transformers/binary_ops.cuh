#ifndef __BINARY_OPS_CUH__
#define __BINARY_OPS_CUH__

#include "tensor.hpp"

void tensor_add_f32(
	const KernelParameters& kpms, 
	const Tensor<float32, CUDA>& lhs, 
	const Tensor<float32, CUDA>& rhs, 
	const Tensor<float32, CUDA>& out);


void tensor_add_i32(
	const KernelParameters& kpms,
	const Tensor<int32, CUDA>& lhs,
	const Tensor<int32, CUDA>& rhs,
	const Tensor<int32, CUDA>& out);


#endif  // __BINARY_OPS_CUH__
