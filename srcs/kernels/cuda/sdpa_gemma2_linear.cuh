#ifndef __SDPA_GEMMA2_LINEAR_CUH__
#define __SDPA_GEMMA2_LINEAR_CUH__

#include "tensor.hpp"

// tiled matrix multiplication, shared mem level
void cu_sdpa_gemma2_linear_f32_v1(
	const Tensor<float32, CUDA>& xt,  // input (batch * seq_len, 2304|2048)
	const Tensor<float32, CUDA>& wt,  // proj weight, typical sizes (2304|1024|2048)
	Tensor<float32, CUDA>& yt);


#endif  // __SDPA_GEMMA2_LINEAR_CUH__
