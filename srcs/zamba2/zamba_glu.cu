#include "math_ops.cuh"

__global__ void cu_tensor_zamba_glu_f32_kernel(
	const int length,
	const float32* dx,
	float32* dy)
{
	
}

void cu_tensor_zamba_glu_f32(
	const Tensor<float32, CUDA>& xt,
	Tensor<float32, CUDA>& yt)
{
	// TODO: implement this

	// calc kernel arguments
	const int length = xt.size();

	auto* dx = xt.buffer();
	auto* dy = yt.buffer();

	// kernel lauch params
	dim3 bs = { 32 * 4, 1, 1 };
	dim3 gs = { calc_req_num_blocks(length, bs.x), 1, 1 };

	cu_tensor_zamba_glu_f32_kernel<<<gs, bs >>>(length, dx, dy);
	CUDA_CHECK_LAST_ERROR();
}
