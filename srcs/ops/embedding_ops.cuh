#ifndef __EMBEDDING_OPS_CUH__
#define __EMBEDDING_OPS_CUH__

#include "tensor.hpp"

void cu_tensor_embedding_f32(
	const Tensor<int32, CUDA>& xt, 
	const Tensor<float32, CUDA>& wt,
	Tensor<float32, CUDA>& yt);


#endif  // __EMBEDDING_OPS_CUH__
