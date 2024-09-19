#ifndef __NORM_OPS_CUH__
#define __NORM_OPS_CUH__

#include "tensor.hpp"

void cu_tensor_layer_norm_f32(
	const Tensor<float32, CUDA>& xt,
	const Tensor<float32, CUDA> wt,
	const Tensor<float32, CUDA> bt,
	const float32 eps,
	Tensor<float32, CUDA>& yt);


void cu_tensor_rms_norm_f32(
	const Tensor<float32, CUDA>& xt,
	const Tensor<float32, CUDA> wt,
	const float32 eps,
	Tensor<float32, CUDA>& yt);

#endif  // __NORM_OPS_CUH__
