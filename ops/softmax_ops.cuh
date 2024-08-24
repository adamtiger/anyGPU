#ifndef __SOFTMAX_OPS_CUH__
#define __SOFTMAX_OPS_CUH__

#include "tensor.hpp"

void tensor_softmax_f32(
	const Tensor<float32, CUDA>& x,
	const Tensor<float32, CUDA>& y);

#endif  // __SOFTMAX_OPS_CUH__
