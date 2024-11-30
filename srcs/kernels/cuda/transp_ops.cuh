#ifndef __TRANSP_OPS_CUH__
#define __TRANSP_OPS_CUH__

#include "tensor.hpp"

/* transpose for 2 dim */

void cu_tensor_transp_f32(
	const Tensor<float32, CUDA>& x,
	Tensor<float32, CUDA>& y);

void cu_tensor_transp_i8(
	const Tensor<int8, CUDA>& x,
	Tensor<int8, CUDA>& y);

/* transpose with swapping two arbitrary axes */

void cu_tensor_transp_swap_f32(
	const Tensor<float32, CUDA>& x,
	const int32 ax1,
	const int32 ax2,
	Tensor<float32, CUDA>& y);

#endif  // __TRANSP_OPS_CUH__
