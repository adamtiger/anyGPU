#ifndef __GEMMA_UPDATE_MASK_CUH__
#define __GEMMA_UPDATE_MASK_CUH__

#include "tensor.hpp"

void cu_tensor_gemma_update_mask_f32(
	const Tensor<int32, CUDA>& attention_mask,
	const Tensor<int32, CUDA>& cache_position,
	const int32 trg_len,
	Tensor<float32, CUDA>& yt);

#endif  // __GEMMA_UPDATE_MASK_CUH__
