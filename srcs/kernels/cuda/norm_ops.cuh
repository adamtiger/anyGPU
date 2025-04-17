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
	const bool zero_centered,
	Tensor<float32, CUDA>& yt);


void cu_tensor_group_norm_f32(
	const Tensor<float32, CUDA>& xt,
	const Tensor<float32, CUDA> wt,
	const Tensor<float32, CUDA> bt,
	const int32 num_groups,
	const float32 eps,
	Tensor<float32, CUDA>& yt);

// optimized versions (experimental)

void cu_tensor_rms_norm_f32_v1(
	const Tensor<float32, CUDA>& xt,
	const Tensor<float32, CUDA> wt,
	const float32 eps,
	const bool zero_centered,
	Tensor<float32, CUDA>& yt);

#endif  // __NORM_OPS_CUH__
