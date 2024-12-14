#ifndef __MLP_GEMMA2_DP_LINEAR_CUH__
#define __MLP_GEMMA2_DP_LINEAR_CUH__

#include "tensor.hpp"

// tiled matrix multiplication, shared mem level
void cu_mlp_gemma2_dp_linear_f32_v1(
	const Tensor<float32, CUDA>& xt,     // input (batch * seq_len, 9216)
	const Tensor<float32, CUDA>& wt_dp,  // down proj weight, (9216, 2304)
	Tensor<float32, CUDA>& yt);


// tiled matrix multiplication, shared mem level
// already transposed weight
void cu_mlp_gemma2_dp_linear_f32_v2(
	const Tensor<float32, CUDA>& xt,     // input (batch * seq_len, 9216)
	const Tensor<float32, CUDA>& wt_dp,  // down proj weight, (2304, 9216)
	Tensor<float32, CUDA>& yt);


// tiled matrix multiplication, shared mem level
// transposed xt (corner turning)
void cu_mlp_gemma2_dp_linear_f32_v3(
	const Tensor<float32, CUDA>& xt,     // input (batch * seq_len, 9216)
	const Tensor<float32, CUDA>& wt_dp,  // down proj weight, (9216, 2304)
	Tensor<float32, CUDA>& yt);

#endif  // __MLP_GEMMA2_DP_LINEAR_CUH__
