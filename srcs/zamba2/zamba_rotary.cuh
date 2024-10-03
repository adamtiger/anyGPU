#ifndef __ZAMBA_ROTARY_CUH__
#define __ZAMBA_ROTARY_CUH__

#include "tensor.hpp"

void cu_tensor_zamba_precomp_rotary_embedding_f32(
	const Tensor<int32, CUDA>& pt,
	const int32 emb_size,
	const int32 base,
	Tensor<float32, CUDA>& freq);


void cu_tensor_apply_zamba_rotary_embedding_f32(
	const Tensor<float32, CUDA>& xt,
	const Tensor<float32, CUDA>& ft,
	Tensor<float32, CUDA>& yt);

#endif  // __ZAMBA_ROTARY_CUH__
