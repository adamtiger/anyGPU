#ifndef __QUANTIZE_OPS_CUH__
#define __QUANTIZE_OPS_CUH__

#include "tensor.hpp"

void tensor_quantize_linear_f32_i8(
	const Tensor<float32, CUDA>& x, 
	const float32 scale, 
	const int8 bias,
	Tensor<int8, CUDA>& y);


void tensor_dequantize_linear_i8_f32(
	const Tensor<int8, CUDA>& x,
	const float32 scale,
	const int8 bias,
	Tensor<float32, CUDA>& y);


#endif  // __QUANTIZE_OPS_CUH__
