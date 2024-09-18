#ifndef __NORM_OPS_CUH__
#define __NORM_OPS_CUH__

#include "tensor.hpp"

void cu_tensor_layer_norm_f32(
	const Tensor<float32, CUDA>& xt,
	const int32 axis,
	const float32 gamma,
	const float32 beta,
	const float32 eps,
	Tensor<float32, CUDA>& yt);


#endif  // __NORM_OPS_CUH__
