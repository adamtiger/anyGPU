#ifndef __GEMMA_SLIDE_MASK_CUH__
#define __GEMMA_SLIDE_MASK_CUH__

#include "tensor.hpp"

void cu_tensor_gemma_slide_mask_f32(
	const Tensor<float32, CUDA>& attention_mask,
	const int32 sliding_window,
	Tensor<float32, CUDA>& yt);

#endif  // __GEMMA_SLIDE_MASK_CUH__
