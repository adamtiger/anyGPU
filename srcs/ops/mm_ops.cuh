#ifndef __MM_OPS_CUH__
#define __MM_OPS_CUH__

#include "tensor.hpp"

/*
  Opt0 - the base kernel implementation without
    relevant optimizations.
*/
void cu_tensor_mm_f32_opt0(
	const Tensor<float32, CUDA>& lhs,
	const Tensor<float32, CUDA>& rhs,
	const Tensor<float32, CUDA>& out);


/*
  gemm operator on gpu, float32 support, all shapes
  Opt0 - the base kernel implementation without
	relevant optimizations.
*/
void cu_tensor_gemm_f32_opt0(
	const Tensor<float32, CUDA>& xt,
	const Tensor<float32, CUDA>& wt,
	const Tensor<float32, CUDA>& bt,
	const Tensor<float32, CUDA>& out);


namespace opt1
{
	/*
	  Opt1 - optimization ideas are:
		shared memory for saving bandwidth;
		coalasced read and write;
	  Assumed:
		medium, large matrices (can not fit in shared);
		size is devidable by 32;
		fixed tile size and block size (minimal flexibility);
		specified for rtx3050Ti
	*/
	void cu_tensor_mm_f32(
		const Tensor<float32, CUDA>& lhs,
		const Tensor<float32, CUDA>& rhs,
		const Tensor<float32, CUDA>& out);
}

namespace opt2
{
	/*
	  Opt2 - optimization ideas are:
		shared memory for saving bandwidth;
		coalasced read and write;
		fp16;
		tensor core instrinsics;
	  Assumed:
		medium, large matrices (can not fit in shared);
		size is devidable by 32;
		fixed tile size and block size (minimal flexibility);
		specified for rtx3050Ti
	*/
	void cu_tensor_mm_f16(
		const Tensor<float16, CUDA>& lhs,
		const Tensor<float16, CUDA>& rhs,
		const Tensor<float16, CUDA>& out);
}

#endif  // __MM_OPS_CUH__
