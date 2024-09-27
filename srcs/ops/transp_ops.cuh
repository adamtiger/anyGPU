#ifndef __TRANSP_OPS_CUH__
#define __TRANSP_OPS_CUH__

#include "tensor.hpp"

void cu_tensor_transp_f32(
	const Tensor<float32, CUDA>& x,
	const Tensor<float32, CUDA>& y);

void cu_tensor_transp_i8(
	const Tensor<int8, CUDA>& x,
	const Tensor<int8, CUDA>& y);

#endif  // __TRANSP_OPS_CUH__
