#ifndef __EMBEDDING_OPS_CUH__
#define __EMBEDDING_OPS_CUH__

#include "tensor.hpp"

void cu_tensor_embedding_f32(
	const Tensor<int32, CUDA>& xt, 
	const Tensor<float32, CUDA>& wt,
	Tensor<float32, CUDA>& yt);


void cu_tensor_precomp_rotary_embedding_f32(
	const int32 max_seq_len,
	const int32 emb_size,
	const int32 base,
	Tensor<float32, CUDA>& freq);


void cu_tensor_apply_rotary_embedding_f32(
	const Tensor<float32, CUDA>& xt,
	const Tensor<int32, CUDA>& pt,
	const Tensor<float32, CUDA>& ft,
	Tensor<float32, CUDA>& yt);


void cu_tensor_apply_alt_rotary_embedding_f32(
	const Tensor<float32, CUDA>& xt,
	const Tensor<int32, CUDA>& pt,
	const Tensor<float32, CUDA>& ft,
	Tensor<float32, CUDA>& yt);


#endif  // __EMBEDDING_OPS_CUH__
