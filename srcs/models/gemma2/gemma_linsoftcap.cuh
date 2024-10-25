#ifndef __GEMMA_LIN_SOFTCAP_CUH__
#define __GEMMA_LIN_SOFTCAP_CUH__

#include "tensor.hpp"

void cu_tensor_gemma_linear_softcap_f32(
	const Tensor<float32, CUDA>& xt,
	const Tensor<float32, CUDA>& wt,
	const float32 final_softcapping,
	Tensor<float32, CUDA>& yt);

#endif  // __GEMMA_LIN_SOFTCAP_CUH__
