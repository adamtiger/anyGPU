#ifndef __ZAMBA_GLU_CUH__
#define __ZAMBA_GLU_CUH__

#include "tensor.hpp"

void cu_tensor_zamba_glu_f32(
	const Tensor<float32, CUDA>& xt,
	Tensor<float32, CUDA>& yt);


#endif  // __ZAMBA_GLU_CUH__
