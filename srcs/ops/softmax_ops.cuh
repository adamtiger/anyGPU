#ifndef __SOFTMAX_OPS_CUH__
#define __SOFTMAX_OPS_CUH__

#include "tensor.hpp"

void cu_tensor_softmax_f32(
	const Tensor<float32, CUDA>& x,
	Tensor<float32, CUDA>& y);

void cu_tensor_softmax_bwd_f32(
	const Tensor<float32, CUDA>& x,
	const Tensor<float32, CUDA>& grad_y,
	Tensor<float32, CUDA>& grad_x);

#endif  // __SOFTMAX_OPS_CUH__
