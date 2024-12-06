#ifndef __MLP_GEMMA2_FUSED_UPPROJ_CUH__
#define __MLP_GEMMA2_FUSED_UPPROJ_CUH__

#include "tensor.hpp"

// 128 horizontal tile size, tiled mm
void cu_mlp_gemma2_fused_uprpoj_f32_128(
	const Tensor<float32, CUDA>& xt,     // input (batch * seq_len, 2304)
	const Tensor<float32, CUDA>& wt_gp,  // gate proj weight, (2304, 9216)
	const Tensor<float32, CUDA>& wt_up,  // gate up proj weight, (2304, 9216) 
	Tensor<float32, CUDA>& yt);

#endif  // __MLP_GEMMA2_FUSED_UPPROJ_CUH__
