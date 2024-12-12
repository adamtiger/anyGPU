#ifndef __MLP_GEMMA2_FUSED_UPPROJ_CUH__
#define __MLP_GEMMA2_FUSED_UPPROJ_CUH__

#include "tensor.hpp"

// global memory access implementation
// naive matrix multiplication, but fused
void cu_mlp_gemma2_fused_upproj_f32_v1(
	const Tensor<float32, CUDA>& xt,     // input (batch * seq_len, 2304)
	const Tensor<float32, CUDA>& wt_gp,  // gate proj weight, (2304, 9216)
	const Tensor<float32, CUDA>& wt_up,  // gate up proj weight, (2304, 9216) 
	Tensor<float32, CUDA>& yt);


// fused and tiled matrix multiplication
void cu_mlp_gemma2_fused_upproj_f32_v2(
	const Tensor<float32, CUDA>& xt,     // input (batch * seq_len, 2304)
	const Tensor<float32, CUDA>& wt_gp,  // gate proj weight, (2304, 9216)
	const Tensor<float32, CUDA>& wt_up,  // gate up proj weight, (2304, 9216) 
	Tensor<float32, CUDA>& yt);

#endif  // __MLP_GEMMA2_FUSED_UPPROJ_CUH__
