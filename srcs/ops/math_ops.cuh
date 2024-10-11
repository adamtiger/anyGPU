#ifndef __MATH_OPS_CUH__
#define __MATH_OPS_CUH__

#include "tensor.hpp"

void cu_tensor_silu_f32(
	const Tensor<float32, CUDA>& xt,
	Tensor<float32, CUDA>& yt);

void cu_tensor_gelu_f32(
	const Tensor<float32, CUDA>& xt,
	Tensor<float32, CUDA>& yt);

#endif  // __MATH_OPS_CUH__
