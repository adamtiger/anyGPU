#ifndef __CONV_CUH__
#define __CONV_CUH__

#include "tensor.hpp"

// baseline global
void cu_tensor_conv2d_f32(
	const Tensor<float32, CUDA>& xt,
	const Tensor<float32, CUDA> wt,
	const Tensor<float32, CUDA> bt,
	const std::array<int32, 2>& stride,
	const std::array<int32, 4>& pads,
	Tensor<float32, CUDA>& yt);


#endif  // __CONV_CUH__
