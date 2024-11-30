#ifndef __CAUSAL_CONV1D_CUH__
#define __CAUSAL_CONV1D_CUH__

#include "tensor.hpp"

void cu_tensor_causal_conv1d_f32(
	const Tensor<float32, CUDA>& xt,
	const Tensor<float32, CUDA>& wt,
	const Tensor<float32, CUDA>& bt,
	Tensor<float32, CUDA>& yt);


#endif  // __CAUSAL_CONV1D_CUH__
