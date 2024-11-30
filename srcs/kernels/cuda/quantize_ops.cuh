#ifndef __QUANTIZE_OPS_CUH__
#define __QUANTIZE_OPS_CUH__

#include "tensor.hpp"

void cu_tensor_quantize_linear_cuda_f32_i8(
	const Tensor<float32, CUDA>& x, 
	const float32 scale, 
	const int8 bias,
	Tensor<int8, CUDA>& y);


void cu_tensor_dequantize_linear_cuda_i8_f32(
	const Tensor<int8, CUDA>& x,
	const float32 scale,
	const int8 bias,
	Tensor<float32, CUDA>& y);


void cu_tensor_qmm_cuda_i8_f32(
	const Tensor<int8, CUDA>& a,
	const Tensor<int8, CUDA>& b,
	const float32 sa, const int8 zpa,
	const float32 sb, const int8 zpb,
	const float32 sy, const int8 zpy,
	Tensor<int8, CUDA>& y);

#endif  // __QUANTIZE_OPS_CUH__
