#ifndef __BINARY_OPS_CUH__
#define __BINARY_OPS_CUH__

#include "tensor.hpp"

void cu_tensor_add_f32(
	const KernelParameters& kpms, 
	const Tensor<float32, CUDA>& lhs, 
	const Tensor<float32, CUDA>& rhs, 
	const Tensor<float32, CUDA>& out);

void cu_tensor_add_i32(
	const KernelParameters& kpms,
	const Tensor<int32, CUDA>& lhs,
	const Tensor<int32, CUDA>& rhs,
	const Tensor<int32, CUDA>& out);


void cu_tensor_mul_f32(
	const KernelParameters& kpms,
	const Tensor<float32, CUDA>& lhs,
	const float32 rhs,
	const Tensor<float32, CUDA>& out);


void cu_tensor_mul_i32(
	const KernelParameters& kpms,
	const Tensor<int32, CUDA>& lhs,
	const int32 rhs,
	const Tensor<int32, CUDA>& out);


void cu_tensor_mul_f32(
	const KernelParameters& kpms,
	const Tensor<float32, CUDA>& lhs,
	const Tensor<float32, CUDA>& rhs,
	const Tensor<float32, CUDA>& out);


#endif  // __BINARY_OPS_CUH__
